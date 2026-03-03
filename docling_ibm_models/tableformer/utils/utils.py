import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models.resnet import BasicBlock, conv1x1
from torchvision.ops.boxes import box_area


def remove_padding(seq):
    r"""
    Remove the trailing zeros from the provided input

    Parameters
    ----------
    list: List of integers
        Predicted sequence

    Returns
    -------
    list: List of integers
        The part of the input before the zero padding

    """
    pad_len = 0
    for x in reversed(seq):
        if x != 0:
            break
        pad_len += 1
    if pad_len == 0:
        return seq, 0

    un_padded = seq[:-pad_len]
    return un_padded, pad_len


def probabilities_to_predictions(probabilities):
    r"""
    Convert probabilities to predictions

    Parameters
    ----------
    probabilities : Tensor[batch_size, vocab_size, seq_len]
        All log probabilities coming out at the last stage of the decoder

    Returns
    -------
    predictions : tensor [batch_size, output_sequence_length]
        The prediceted trags

    """
    # max_idx: [batch_size, seq_len]
    max_idx = torch.argmax(probabilities, dim=1)
    return max_idx


def print_target_predict(target, predictions, filenames=None, batch_idx=0):
    r"""
    For the Tags, print the target and predicted tensors for the specified batch index

    We expect to have the batch size as the first dimension.
    Only the specified batch is extractred and the remaining dimenions are flattened.
    The results are printed as 2 lists with the target on top and the predictions below underlined

    Parameters
    ---------
    target : tensor [batch_size, output_sequence_length]
        The ground truth tags

    predictions : tensor [batch_size, output_sequence_length]
        The prediceted trags

    filenames : list of string
        The actual filename that provides the data

    batch_idx : int
        Which index in the batch dimension will be printed
    """
    target_flat = target[batch_idx].flatten()
    predictions_flat = predictions[batch_idx].flatten()
    target_label = "target"
    predict_label = "predict"
    if filenames is not None:
        target_label = filenames[batch_idx]
    label_len = max(len(target_label), len(predict_label))
    print("{}: {}".format(target_label.ljust(label_len, " "), target_flat.tolist()))
    print(
        "{}: {}".format(predict_label.ljust(label_len, " "), predictions_flat.tolist())
    )


def load_image(full_fn):
    r"""
    Load an image from the disk as a numpy array

    Parameters
    ----------
    full_fn : string
        The full path filename of the image

    Results
    -------
    img : numpy array: (channels, width, height)
        The loaded image as a numpy array
    """
    with Image.open(full_fn) as f:
        img = np.asarray(f)  # (width, height, channels)
        img = img.transpose(2, 0, 1)  # (channels, width, height)
    return img


def resnet_block(stride=1):
    layers = []
    downsample = nn.Sequential(
        conv1x1(256, 512, stride),
        nn.BatchNorm2d(512),
    )
    layers.append(BasicBlock(256, 512, stride, downsample))
    layers.append(BasicBlock(512, 512, 1))
    return nn.Sequential(*layers)


def repackage_hidden(h):
    r"""
    Wraps hidden states in new Tensors, to detach them from their history.
    """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@torch.no_grad()
def bip_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def generate_square_subsequent_mask(sz: int, device: str = "cpu") -> torch.Tensor:
    """Generate the attention mask for causal decoding"""
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    ).to(device=device)
    return mask


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    Source from: https://github.com/Bjarten/early-stopping-pytorch
    """

    def __init__(self, patience=2, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self._patience = patience
        self._verbose = verbose
        self._counter = 0
        self._best_score = None
        self._early_stop = False
        self._val_loss_min = np.Inf
        self._delta = delta
        self._trace_func = trace_func

    def __call__(self, val_loss):
        score = -val_loss
        save_checkpoint = True
        if self._best_score is None:
            self._best_score = score
            save_checkpoint = True
            if self._verbose:
                verb = f"Validation loss decreased ({self._val_loss_min:.6f} --> {val_loss:.6f})."
                self._trace_func(verb)
            self._val_loss_min = val_loss
        elif score < self._best_score + self._delta:
            self._counter += 1
            self._trace_func(
                f"EarlyStopping counter: {self._counter} out of {self._patience}"
            )
            if self._counter >= self._patience:
                self._early_stop = True
                save_checkpoint = False
        else:
            self._best_score = score
            save_checkpoint = True
            self._counter = 0
            if self._verbose:
                verb = f"Validation loss decreased ({self._val_loss_min:.6f} --> {val_loss:.6f})."
                self._trace_func(verb)
            self._val_loss_min = val_loss
        return save_checkpoint


def print_dict(m: dict):
    r"""
    Print dict elements in separate lines sorted by keys
    """
    if len(m) == 0:
        return

    # Check if the key is a stringified integer
    first_key = next(iter(m))
    is_numeric = isinstance(first_key, str) and first_key.isnumeric()
    if is_numeric:
        keys = sorted([int(k) for k in m.keys()])
    else:
        keys = sorted([k for k in m.keys()])

    for k in keys:
        if is_numeric:
            v = m[str(k)]
        else:
            v = m[k]
        print("{}: {}".format(k, v))


def print_list(lst: list):
    r"""
    Print list elements in separate lines
    """
    for i, elm in enumerate(lst):
        if isinstance(elm, list):
            print("{}: ({}) - {}".format(i, len(elm), elm))
        else:
            print("{}: {}".format(i, elm))
