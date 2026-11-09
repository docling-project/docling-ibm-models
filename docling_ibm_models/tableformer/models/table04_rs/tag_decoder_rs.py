import contextlib
import threading
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def _attend_one(q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
    r"""
    Attention of a single query row: q (B, H, 1, hd), k and v (B, H, S, hd).

    Two batched matrix products instead of scaled_dot_product_attention, whose fused
    kernels are tuned for long queries and are several times slower for one row.
    """
    scores = torch.matmul(q, k.transpose(-1, -2)) * (q.size(-1) ** -0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v)


class TagDecoder:
    r"""
    Greedy OTSL decoding of the tag transformer of TableModel04_rs for a batch of
    tables, with the same structure corrections as the original token loop.

    Each step projects the keys and values of the new token only (KV cache) and
    attends over a cache of fixed length with a mask, and the image memory's keys
    and values are projected once per table. All steps therefore have the same
    shapes: on CUDA one step is captured as a CUDA graph and replayed, so a token
    costs one launch instead of ~150 small kernels. The tag corrections and the end
    check run on the device, and the host only checks every few steps whether all
    tables have ended, without waiting for each token.

    Not thread safe: TableModel04_rs serialises the calls.
    """

    def __init__(self, model, batch_size: int, max_steps: int, use_cuda_graph: bool):
        tt = model._tag_transformer
        self._tt = tt
        self._layers = list(tt._decoder.layers)
        word_map = model._init_data["word_map"]["word_map_tag"]
        self._start = word_map["<start>"]
        self._end = word_map["<end>"]
        self._xcel = word_map["xcel"]
        self._lcel = word_map["lcel"]
        self._fcel = word_map["fcel"]
        self._ucel = word_map["ucel"]

        device = next(tt.parameters()).device
        self._device = device
        self._B = batch_size
        self._T = max_steps
        self._E = tt._decoder_dim
        self._H = tt._n_heads
        self._hd = self._E // self._H
        self._P = tt._enc_image_size**2
        B, T, E, H, hd, P = self._B, self._T, self._E, self._H, self._hd, self._P
        n_layers = len(self._layers)

        def zeros(*shape, dtype=torch.float32):
            return torch.zeros(*shape, dtype=dtype, device=device)

        # Static buffers: the captured graph reads and writes only these
        self._mem_k = [zeros(B, H, P, hd) for _ in range(n_layers)]
        self._mem_v = [zeros(B, H, P, hd) for _ in range(n_layers)]
        self._k_cache = [zeros(B, H, T, hd) for _ in range(n_layers)]
        self._v_cache = [zeros(B, H, T, hd) for _ in range(n_layers)]
        self._x = zeros(1, B, E)  # embedded current tag
        self._t = zeros(1, dtype=torch.long)  # current step
        self._positions = torch.arange(T, device=device)
        self._hidden = zeros(T, B, E)  # last decoder layer output per step
        self._tags = zeros(T + 1, B, dtype=torch.long)
        self._finished = zeros(B, dtype=torch.bool)
        self._prev_ucel = zeros(B, dtype=torch.bool)
        self._lengths = zeros(B, dtype=torch.long)
        self._all_finished = zeros(1, dtype=torch.bool)
        self._pe = tt._positional_encoding.pe  # (max_len, 1, E)

        self._graph = None
        if use_cuda_graph and device.type == "cuda":
            self._capture()

    def _device_context(self):
        if self._device.type == "cuda":
            return torch.cuda.device(self._device)
        return contextlib.nullcontext()

    def _project(self, mha, x: Tensor, part: slice):
        return F.linear(x, mha.in_proj_weight[part], mha.in_proj_bias[part])

    def _heads(self, x: Tensor):
        # (S, B, E) -> (B, H, S, hd)
        return x.view(x.size(0), self._B, self._H, self._hd).permute(1, 2, 0, 3)

    def _out(self, mha, a: Tensor):
        # (B, H, 1, hd) -> (1, B, E)
        a = a.permute(2, 0, 1, 3).reshape(self._B, self._E)
        return F.linear(a, mha.out_proj.weight, mha.out_proj.bias).view(
            1, self._B, self._E
        )

    def _step(self):
        r"""
        One decoding step on the static buffers (same layers as TMTransformerDecoder).
        """
        E, t = self._E, self._t
        q_part, kv_part = slice(0, E), slice(E, 3 * E)
        mask = (self._positions <= t).view(1, 1, 1, self._T)
        h = self._x
        for i, layer in enumerate(self._layers):
            sa = layer.self_attn
            q = self._heads(self._project(sa, h, q_part))
            kv = self._project(sa, h, kv_part).view(1, self._B, 2, self._H, self._hd)
            self._k_cache[i].index_copy_(2, t, kv[:, :, 0].permute(1, 2, 0, 3))
            self._v_cache[i].index_copy_(2, t, kv[:, :, 1].permute(1, 2, 0, 3))
            a = _attend_one(q, self._k_cache[i], self._v_cache[i], mask)
            h = layer.norm1(h + self._out(sa, a))

            ca = layer.multihead_attn
            q = self._heads(self._project(ca, h, q_part))
            a = _attend_one(q, self._mem_k[i], self._mem_v[i])
            h = layer.norm2(h + self._out(ca, a))
            h = layer.norm3(h + layer.linear2(layer.activation(layer.linear1(h))))
        self._hidden.index_copy_(0, t, h)

        new_tag = self._tt._fc(h[0]).argmax(1)
        # Structure corrections of the original loop: xcel -> lcel (on every line,
        # since its line counter never advances) and lcel after ucel -> fcel
        new_tag = torch.where(new_tag == self._xcel, self._lcel, new_tag)
        new_tag = torch.where(
            self._prev_ucel & (new_tag == self._lcel), self._fcel, new_tag
        )
        new_tag = torch.where(self._finished, self._end, new_tag)
        ended = new_tag == self._end
        self._lengths.copy_(torch.where(~self._finished & ended, t + 1, self._lengths))
        self._finished.copy_(self._finished | ended)
        self._prev_ucel.copy_(new_tag == self._ucel)
        self._tags.index_copy_(0, t + 1, new_tag.view(1, self._B))
        next_pos = torch.clamp(t + 1, max=self._pe.size(0) - 1)
        self._x.copy_(
            self._tt._embedding(new_tag).view(1, self._B, E)
            + self._pe.index_select(0, next_pos)
        )
        self._all_finished.copy_(self._finished.all().view(1))
        self._t.add_(1)

    @torch.no_grad()
    def _capture(self):
        with self._device_context():
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):  # warm-up, as required before capture
                    self._step()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            # thread_local: other docling stages may use the GPU from other threads
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                self._step()
        self._graph = graph

    @torch.no_grad()
    def decode(self, memory: Tensor, max_steps: int, check_every: int = 8):
        r"""
        Parameters
        ----------
        memory : Tensor (P, n, E)
            Output of the tag transformer encoder for n <= batch_size tables
        max_steps : int
            Maximum number of tags generated per table (at most the buffer length)

        Returns
        -------
        tags : list of list of int
            Per table: <start>, the generated tags and <end> (if it was reached)
        hidden : Tensor (steps, n, E)
            Last decoder layer output of every step, the input of the bbox decoder
        """
        with self._device_context():
            return self._decode(memory, max_steps, check_every)

    def _decode(self, memory: Tensor, max_steps: int, check_every: int):
        n = memory.size(1)
        max_steps = min(max_steps, self._T)
        if n < self._B:  # fill the batch; the extra rows start as finished
            memory = torch.cat([memory, memory[:, :1].expand(-1, self._B - n, -1)], 1)
        E = self._E
        for i, layer in enumerate(self._layers):
            ca = layer.multihead_attn
            self._mem_k[i].copy_(
                self._heads(self._project(ca, memory, slice(E, 2 * E)))
            )
            self._mem_v[i].copy_(
                self._heads(self._project(ca, memory, slice(2 * E, 3 * E)))
            )
        self._t.zero_()
        self._tags.fill_(self._end)
        self._tags[0] = self._start
        self._finished.zero_()
        self._finished[n:] = True
        self._prev_ucel.zero_()
        self._lengths.fill_(max_steps)
        self._all_finished.zero_()
        self._x.copy_(
            self._tt._embedding(self._tags[0]).view(1, self._B, self._E) + self._pe[0]
        )

        on_cuda = self._device.type == "cuda"
        flag = torch.zeros(1, dtype=torch.bool)
        if on_cuda:
            flag = flag.pin_memory()
        event = None
        steps = max_steps
        for t in range(max_steps):
            if self._graph is not None:
                self._graph.replay()
            else:
                self._step()
            if not on_cuda:
                if bool(self._all_finished):
                    steps = t + 1
                    break
                continue
            if (t + 1) % check_every == 0:
                # Read the flag copied at the previous check: the host stays at most
                # check_every steps ahead of the device and never waits per token
                if event is not None:
                    event.synchronize()
                    if bool(flag):
                        steps = t + 1
                        break
                flag.copy_(self._all_finished, non_blocking=True)
                event = torch.cuda.Event()
                event.record()

        steps = min(steps, max_steps)
        tags = self._tags[: steps + 1, :n].T.tolist()
        lengths = self._lengths[:n].tolist()
        seqs = [row[: 1 + min(length, steps)] for row, length in zip(tags, lengths)]
        return seqs, self._hidden[:steps, :n]


class TagDecoders:
    r"""
    One TagDecoder per batch size (powers of two up to max_batch_size), built on
    first use. A failed CUDA graph capture falls back to running the steps eagerly.
    """

    def __init__(self, model, max_steps: int, max_batch_size: int = 4):
        self._model = model
        self._max_steps = max_steps
        self.max_batch_size = max_batch_size
        self._decoders: dict[int, TagDecoder] = {}
        self.lock = threading.Lock()

    def get(self, n: int) -> TagDecoder:
        size = 1
        while size < n:
            size *= 2
        if size not in self._decoders:
            try:
                decoder = TagDecoder(self._model, size, self._max_steps, True)
            except RuntimeError:
                decoder = TagDecoder(self._model, size, self._max_steps, False)
            self._decoders[size] = decoder
        return self._decoders[size]
