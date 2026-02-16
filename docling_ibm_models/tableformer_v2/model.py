#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
r"""
TableFormerV2:
This module provides a self-contained implementation of the TableFormerV2
model for table structure recognition. It uses a lightweight architecture optimized for
CPU inference / GPU batch inference while maintaining high accuracy.

Architecture overview:
- Image encoder: EfficientNetV2-S backbone with Squeeze-and-Excitation
- Spatial mixer: Depthwise separable convolutions (no self-attention in encoder)
- Decoder: Cache-aware Transformer decoder with cross-attention to image features
- Bbox head: Multi-layer attention decoder for cell bounding box prediction
"""
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, cast

import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s
from transformers import (
    AutoConfig,
    AutoModel,
    GenerationMixin,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import ModelOutput

_log = logging.getLogger(__name__)

from docling_ibm_models.tableformer.utils.app_profiler import AggProfiler

# =============================================================================
# Custom Output Classes
# =============================================================================


@dataclass
class TableFormerV2Output(ModelOutput):
    r"""
    Output class for TableFormerV2 inference.

    Attributes
    ----------
    logits : torch.Tensor, optional
        Token prediction logits of shape (B, L, vocab_size)
    hidden_states : torch.Tensor, optional
        Decoder hidden states of shape (B, L, D)
    predicted_bboxes : torch.Tensor, optional
        Predicted bounding boxes in xyxy format [0,1], shape (N, 4)
    past_key_values : tuple, optional
        Cached key-value pairs for autoregressive generation
    """

    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    predicted_bboxes: Optional[torch.Tensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None


# =============================================================================
# Model Building Blocks
# =============================================================================


class SqueezeExcitation(nn.Module):
    r"""
    Squeeze-and-Excitation block for channel-wise attention.

    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class DepthwiseSeparableBlock(nn.Module):
    r"""
    Depthwise separable convolution block for lightweight spatial mixing.
    """

    def __init__(self, channels: int, expansion: float = 1.0):
        super().__init__()
        hidden = int(channels * expansion)
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            SqueezeExcitation(hidden),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# =============================================================================
# Feature Pyramid Network (FPN)
# =============================================================================


class SimpleFPN(nn.Module):
    r"""
    Simple Feature Pyramid Network to fuse multi-scale features from EfficientNet.

    EfficientNetV2-S feature map sizes at 448x448 input:
    - Stage 2: 112x112, 48 channels  (fine details: grid lines, small text)
    - Stage 4: 28x28, 128 channels   (medium structures)
    - Final:   14x14, 1280 channels  (after conv head)

    This FPN fuses stages 2, 4, and final to capture multi-scale information.
    """

    def __init__(self, out_channels: int = 1280):
        super().__init__()
        # EfficientNetV2-S actual channel sizes: idx2=48, idx4=128, final=1280
        self.in_channels = [48, 128, 1280]  # stages 2, 4, final

        # Lateral connections (1x1 conv to match channels)
        self.lateral_conv_s2 = nn.Conv2d(48, out_channels, kernel_size=1)
        self.lateral_conv_s4 = nn.Conv2d(128, out_channels, kernel_size=1)
        # Final already has out_channels

        # Smooth convolutions after upsampling
        self.smooth_s2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.smooth_s4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        # Final fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(
        self, feat_s2: torch.Tensor, feat_s4: torch.Tensor, feat_final: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse multi-scale features.

        Args:
            feat_s2: (B, 48, H/4, W/4) - stage 2 features
            feat_s4: (B, 160, H/16, W/16) - stage 4 features
            feat_final: (B, 1280, H/32, W/32) - final features

        Returns:
            fused: (B, 1280, H/32, W/32) - fused features at final resolution
        """
        target_size = feat_final.shape[2:]  # (H/32, W/32)

        # Lateral connections
        p2 = self.lateral_conv_s2(feat_s2)
        p4 = self.lateral_conv_s4(feat_s4)
        p_final = feat_final

        # Resize all to final resolution
        p2 = nn.functional.interpolate(
            p2, size=target_size, mode="bilinear", align_corners=False
        )
        p4 = nn.functional.interpolate(
            p4, size=target_size, mode="bilinear", align_corners=False
        )

        # Smooth
        p2 = self.smooth_s2(p2)
        p4 = self.smooth_s4(p4)

        # Concatenate and fuse
        fused = torch.cat([p2, p4, p_final], dim=1)
        fused = self.fusion(fused)

        return fused


# =============================================================================
# Transformer Decoder Components
# =============================================================================


class CachedTransformerDecoderLayer(nn.Module):
    r"""
    Cache-aware Transformer decoder layer for efficient autoregressive generation.

    Parameters
    ----------
    d_model : int
        Model dimension
    nhead : int
        Number of attention heads
    dim_feedforward : int
        Feedforward network hidden dimension
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.0)
        self.activation = nn.ReLU()

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        k = v = x
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        attn_output, _ = self.self_attn(
            x,
            k,
            v,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        present = (k, v) if use_cache else None
        return attn_output, present

    def _ca_block(
        self,
        x: torch.Tensor,
        mem: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        attn_output, _ = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return attn_output

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        sa_out, present_kv = self._sa_block(
            tgt,
            tgt_mask,
            tgt_key_padding_mask,
            past_kv=past_key_value,
            use_cache=use_cache,
        )
        tgt = self.norm1(tgt + self.dropout(sa_out))
        ca_out = self._ca_block(tgt, memory, memory_mask, memory_key_padding_mask)
        tgt = self.norm2(tgt + self.dropout(ca_out))
        ff = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout(ff))
        return tgt, present_kv


class CachedTransformerDecoder(nn.Module):
    r"""
    Stack of cache-aware Transformer decoder layers.

    Parameters
    ----------
    d_model : int
        Model dimension
    nhead : int
        Number of attention heads
    dim_feedforward : int
        Feedforward network hidden dimension
    num_layers : int
        Number of decoder layers
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CachedTransformerDecoderLayer(d_model, nhead, dim_feedforward)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]]:
        next_past: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = (
            [] if use_cache else None
        )
        for i, layer in enumerate(self.layers):
            past = past_key_values[i] if past_key_values is not None else None
            tgt, present = layer(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                past_key_value=past,
                use_cache=use_cache,
            )
            if use_cache and present is not None and next_past is not None:
                next_past.append(present)
        return tgt, (tuple(next_past) if use_cache and next_past is not None else None)


# =============================================================================
# Bounding Box Decoder Components
# =============================================================================


def _cxcywh_to_xyxy(cxcywh: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = cxcywh.unbind(-1)
    x1 = (cx - 0.5 * w).clamp(0.0, 1.0)
    y1 = (cy - 0.5 * h).clamp(0.0, 1.0)
    x2 = (cx + 0.5 * w).clamp(0.0, 1.0)
    y2 = (cy + 0.5 * h).clamp(0.0, 1.0)
    return torch.stack([x1, y1, x2, y2], dim=-1)


class BboxDecoderLayer(nn.Module):
    r"""
    Single decoder layer for bounding box prediction.
    """

    def __init__(
        self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        batch_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_unsq = x.unsqueeze(0)

        if batch_mask is not None:
            attn_mask = ~batch_mask
        else:
            attn_mask = None

        sa_out, _ = self.self_attn(x_unsq, x_unsq, x_unsq, attn_mask=attn_mask)
        x = self.self_attn_norm(x + sa_out.squeeze(0))

        x_unsq = x.unsqueeze(1)
        ca_out, _ = self.cross_attn(x_unsq, memory, memory)
        x = self.cross_attn_norm(x + ca_out.squeeze(1))

        x = self.ffn_norm(x + self.ffn(x))
        return x


class BboxHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        ff_dim = embed_dim * 4

        self.input_proj = nn.Linear(embed_dim, embed_dim)
        self.input_norm = nn.LayerNorm(embed_dim)

        self.kv_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_norm = nn.LayerNorm(embed_dim)

        self.layers = nn.ModuleList(
            [
                BboxDecoderLayer(embed_dim, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )

        self.bbox_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 4),
        )

    def forward(
        self,
        cell_embeddings: torch.Tensor,
        encoder_hidden: torch.Tensor,
        cell_batch_indices: torch.Tensor,
        spatial_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        r"""
        Predict bounding boxes for cell embeddings.

        Parameters
        ----------
        cell_embeddings : torch.Tensor
            Decoder hidden states at cell positions, shape (N, D)
        encoder_hidden : torch.Tensor
            Encoder outputs (image features), shape (B, S, D)
        cell_batch_indices : torch.Tensor
            Batch index for each cell, shape (N,)
        spatial_size : tuple, optional
            Unused, kept for API compatibility

        Returns
        -------
        torch.Tensor
            Predicted bboxes in xyxy format [0, 1], shape (N, 4)
        """
        if cell_embeddings.numel() == 0:
            return cell_embeddings.new_empty(0, 4)

        batch_mask = cell_batch_indices.unsqueeze(0) == cell_batch_indices.unsqueeze(1)
        encoder_for_cells = encoder_hidden[cell_batch_indices]

        x = self.input_norm(self.input_proj(cell_embeddings))
        memory = self.kv_norm(self.kv_proj(encoder_for_cells))

        for layer in self.layers:
            x = layer(x, memory, batch_mask)

        bbox_cxcywh = torch.sigmoid(self.bbox_mlp(x))
        bbox_xyxy = _cxcywh_to_xyxy(bbox_cxcywh)
        return bbox_xyxy


# =============================================================================
# Configuration Class
# =============================================================================


class TableFormerV2Config(PretrainedConfig):
    model_type = "TableFormerV2"

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        ff_dim: int = 2048,
        num_decoder_layers: int = 2,
        vocab_size: int = 13,
        conv_mixer_expansion: float = 1.0,
        data_cells: Optional[List[int]] = None,
        use_fpn: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_decoder_layers = num_decoder_layers
        self.vocab_size = vocab_size
        self.conv_mixer_expansion = conv_mixer_expansion
        self.data_cells = data_cells or []
        self.use_fpn = use_fpn


# =============================================================================
# Main Model Class
# =============================================================================


class TableFormerV2(PreTrainedModel, GenerationMixin):
    r"""
    TableFormerV2: CPU-optimized model for table structure recognition (inference only).

    This model uses:
    - EfficientNetV2-S backbone for image encoding
    - Depthwise separable convolutions instead of Transformer encoder
    - Cache-aware Transformer decoder for token generation
    - Attention-based bbox head for cell localization

    Parameters
    ----------
    config : TableFormerV2Config
        Model configuration
    """

    config_class = TableFormerV2Config

    def __init__(self, config: TableFormerV2Config):
        super().__init__(config)

        # Vision encoder
        self.feature_extractor = efficientnet_v2_s()
        self.se_module = SqueezeExcitation(in_channels=1280)
        self.conv_mixer = DepthwiseSeparableBlock(
            1280, expansion=config.conv_mixer_expansion
        )
        self.feature_to_embedding = nn.Linear(1280, config.embed_dim)

        # Optional FPN for multi-scale features
        self.use_fpn = getattr(config, "use_fpn", False)
        if self.use_fpn:
            self.fpn = SimpleFPN(out_channels=1280)

        # embeddings
        self.input_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, 512, config.embed_dim))

        # decoder with caching
        self.transformer_decoder = CachedTransformerDecoder(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            num_layers=config.num_decoder_layers,
        )

        # Output heads
        self.output_projection = nn.Linear(config.embed_dim, config.vocab_size)
        self.bbox_head = BboxHead(config.embed_dim, config.num_heads)

        self.data_cells = config.data_cells

    def _is_profiling_enabled(self) -> bool:
        r"""
        Check if profiling is enabled by checking if AggProfiler has cycles.

        Returns
        -------
        bool
            True if profiling is enabled, False otherwise
        """
        try:
            profiler = AggProfiler()
            # Check if profiler has cycles (profiling has been started)
            return len(profiler._cycles) > 0
        except Exception:
            return False

    def _positional_encoding(
        self, batch_size: int, seq_len: int, offset: int = 0
    ) -> torch.Tensor:
        pos_enc_size = self.positional_encoding.size(1)
        total_len = offset + seq_len
        if total_len <= pos_enc_size:
            return self.positional_encoding[:, offset : offset + seq_len, :].expand(
                batch_size, seq_len, -1
            )
        num_repeats = (total_len + pos_enc_size - 1) // pos_enc_size
        repeated = self.positional_encoding.repeat(1, num_repeats, 1)
        return repeated[:, offset : offset + seq_len, :].expand(batch_size, seq_len, -1)

    def encode_images(self, images: torch.Tensor) -> dict:
        prof_enabled = self._is_profiling_enabled()
        if prof_enabled:
            AggProfiler().begin("model_encoder", prof_enabled)

        if self.use_fpn:
            # Extract multi-scale features for FPN
            # EfficientNetV2-S stages: 0-1 (stem), 2, 3, 4, 5, 6, 7 (head)
            x = images
            feat_s2 = None
            feat_s4 = None

            for idx, layer in enumerate(self.feature_extractor.features):
                x = layer(x)
                if idx == 2:  # Stage 2: 48 channels
                    feat_s2 = x
                elif idx == 4:  # Stage 4: 160 channels
                    feat_s4 = x

            feat_final = x  # Final: 1280 channels

            # Fuse with FPN
            features = self.fpn(feat_s2, feat_s4, feat_final)
            features = self.se_module(features)
            features = self.conv_mixer(features)
        else:
            # Original single-scale path
            features = self.feature_extractor.features(images)
            features = self.se_module(features)
            features = self.conv_mixer(features)

        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        encoded = self.feature_to_embedding(features)

        if prof_enabled:
            AggProfiler().end("model_encoder", prof_enabled)

        return {"last_hidden_state": encoded, "spatial_size": (H, W)}

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[dict] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: Optional[bool] = None,
        return_dict: bool = True,
    ) -> TableFormerV2Output:
        r"""
        Forward pass for inference.

        Parameters
        ----------
        images : torch.Tensor, optional
            Input images of shape (B, 3, H, W)
        input_ids : torch.Tensor
            Input token IDs of shape (B, L)
        attention_mask : torch.Tensor, optional
            Attention mask of shape (B, L)
        encoder_outputs : dict, optional
            Pre-computed encoder outputs
        past_key_values : tuple, optional
            Cached key-values for autoregressive decoding
        use_cache : bool, optional
            Whether to use KV caching (default: True)
        return_dict : bool
            Whether to return a ModelOutput (default: True)

        Returns
        -------
        TableFormerV2Output
            Model outputs including logits and predicted bboxes
        """
        use_cache = True if use_cache is None else use_cache

        if encoder_outputs is None:
            if images is None:
                raise ValueError("Either images or encoder_outputs must be provided")
            encoder_outputs = self.encode_images(images)

        if input_ids is None:
            raise ValueError("input_ids must be provided")
        batch_size, seq_len = input_ids.shape

        past_length = 0
        if (
            past_key_values is not None
            and len(past_key_values) > 0
            and past_key_values[0] is not None
        ):
            past_length = past_key_values[0][0].shape[1]

        tgt = self.input_embedding(input_ids) + self._positional_encoding(
            batch_size, seq_len, offset=past_length
        )

        if past_length > 0:
            causal_mask = None
        else:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=input_ids.device)
            ).T
            causal_mask = causal_mask.masked_fill(
                causal_mask == 0, float("-inf")
            ).masked_fill(causal_mask == 1, 0.0)

        prof_enabled = self._is_profiling_enabled()

        if prof_enabled:
            AggProfiler().begin("model_tag_transformer_decoder", prof_enabled)
        decoded, present_kv = self.transformer_decoder(
            tgt=tgt,
            memory=encoder_outputs["last_hidden_state"],
            tgt_mask=causal_mask,
            tgt_key_padding_mask=None,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        if prof_enabled:
            AggProfiler().end("model_tag_transformer_decoder", prof_enabled)

        if prof_enabled:
            AggProfiler().begin("model_tag_transformer_fc", prof_enabled)
        logits = self.output_projection(decoded)
        if prof_enabled:
            AggProfiler().end("model_tag_transformer_fc", prof_enabled)

        # Identify cell positions and predict bboxes
        cell_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for cell_id in self.data_cells:
            cell_mask |= input_ids == cell_id

        cell_positions = torch.nonzero(cell_mask, as_tuple=False)
        cell_embeddings = decoded[cell_mask]
        cell_batch_indices = (
            cell_positions[:, 0]
            if cell_positions.numel() > 0
            else cell_positions.new_empty(0)
        )

        pred_bboxes = self.bbox_head(
            cell_embeddings,
            encoder_outputs["last_hidden_state"],
            cell_batch_indices,
            spatial_size=encoder_outputs.get("spatial_size", None),
        )

        if not return_dict:
            return (logits, decoded, pred_bboxes, present_kv)  # type: ignore[return-value]

        return TableFormerV2Output(
            logits=logits,
            hidden_states=decoded,
            predicted_bboxes=pred_bboxes,
            past_key_values=present_kv,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past: Optional[Tuple] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[dict] = None,
        **kwargs,
    ) -> dict:
        return {
            "input_ids": input_ids,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "past_key_values": past,
            "use_cache": True,
        }

    def generate(
        self,
        images: torch.Tensor,
        tokenizer,
        max_length: int = 512,
        generation_config=None,
        **kwargs,
    ) -> dict:
        r"""
        Autoregressive generation with bounding box prediction.

        Parameters
        ----------
        images : torch.Tensor
            Input images of shape (B, 3, H, W)
        tokenizer : PreTrainedTokenizer
            Tokenizer with bos_token_id and eos_token_id
        max_length : int
            Maximum sequence length (default: 512)
        generation_config : GenerationConfig, optional
            HuggingFace generation configuration

        Returns
        -------
        dict
            Dictionary containing:
            - generated_ids: (B, L) token IDs
            - predicted_bboxes: (B, num_cells, 4) bboxes in xyxy [0, 1]
        """
        if generation_config is not None and hasattr(generation_config, "max_length"):
            max_length = generation_config.max_length

        prof_enabled = self._is_profiling_enabled()
        if prof_enabled:
            AggProfiler().begin("predict_total", prof_enabled)

        self.eval()  # type: ignore[attr-defined]
        with torch.no_grad():
            # Image encoding (profiling handled inside encode_images)
            encoder_outputs = self.encode_images(images)
            batch_size = images.size(0)
            device = images.device

            generated_ids = torch.full(
                (batch_size, 1), tokenizer.bos_token_id, dtype=torch.long, device=device
            )
            current_input = generated_ids
            past_key_values = None

            # Autoregressive generation loop
            for step in range(max_length):
                outputs = self.forward(
                    input_ids=current_input,
                    attention_mask=None,
                    encoder_outputs=encoder_outputs,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )

                if outputs.logits is None:
                    raise ValueError("Model forward pass returned None logits")
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                past_key_values = outputs.past_key_values
                current_input = next_token

                if torch.all(next_token == tokenizer.eos_token_id):
                    break

            # Final forward pass to get hidden states for bbox prediction
            final_outputs = self.forward(
                input_ids=generated_ids,
                attention_mask=torch.ones_like(generated_ids),
                encoder_outputs=encoder_outputs,
                past_key_values=None,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = final_outputs.hidden_states

            # Find cell positions and predict bboxes
            pred_bboxes = None
            max_cells = 0
            cell_positions_per_batch = []

            for b in range(batch_size):
                seq = generated_ids[b]
                positions = []
                for pos, tok in enumerate(seq.tolist()):
                    if tok in self.data_cells:
                        positions.append(pos)
                cell_positions_per_batch.append(positions)
                max_cells = max(max_cells, len(positions))

            if max_cells > 0:
                pred_bboxes = torch.zeros(batch_size, max_cells, 4, device=device)
                spatial_size = encoder_outputs.get("spatial_size", None)

                for b in range(batch_size):
                    positions = cell_positions_per_batch[b]
                    if positions:
                        if hidden_states is None:
                            raise ValueError(
                                "Model forward pass returned None hidden_states"
                            )
                        cell_embs = hidden_states[b, positions, :]
                        batch_indices = torch.zeros(
                            len(positions), dtype=torch.long, device=device
                        )
                        enc_out = encoder_outputs["last_hidden_state"][b : b + 1]

                        if prof_enabled:
                            AggProfiler().begin("model_bbox_decoder", prof_enabled)
                        bboxes = self.bbox_head(
                            cell_embs, enc_out, batch_indices, spatial_size=spatial_size
                        )
                        if prof_enabled:
                            AggProfiler().end("model_bbox_decoder", prof_enabled)
                        pred_bboxes[b, : len(bboxes)] = bboxes

        if prof_enabled:
            AggProfiler().end("predict_total", prof_enabled)

        return {
            "generated_ids": generated_ids,
            "predicted_bboxes": pred_bboxes,
        }


AutoConfig.register("TableFormerV2", TableFormerV2Config)
AutoModel.register(TableFormerV2Config, TableFormerV2)
