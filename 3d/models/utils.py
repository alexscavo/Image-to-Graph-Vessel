"""Utils for deformable transformer."""

import torch
import torch.distributed as dist


def nested_tensor_from_tensor_list(tensor_list):
    # Supports:
    #  - 2D: [C, H, W]  -> batch [B, C, H, W], mask [B, H, W]
    #  - 3D: [C, D, H, W] -> batch [B, C, D, H, W], mask [B, D, H, W]

    if len(tensor_list) == 0:
        raise ValueError("empty tensor_list")

    ndim = tensor_list[0].ndim

    if ndim == 3:
        # [C, H, W]
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape

        dtype = tensor_list[0].dtype
        device = tensor_list[0].device

        padded = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)

        for img, pad_img, m in zip(tensor_list, padded, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False

        return NestedTensor(padded, mask)

    elif ndim == 4:
        # [C, D, H, W]
        max_size = _max_by_axis([list(vol.shape) for vol in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, d, h, w = batch_shape

        dtype = tensor_list[0].dtype
        device = tensor_list[0].device

        padded = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, d, h, w), dtype=torch.bool, device=device)

        for vol, pad_vol, m in zip(tensor_list, padded, mask):
            pad_vol[: vol.shape[0], : vol.shape[1], : vol.shape[2], : vol.shape[3]].copy_(vol)
            m[: vol.shape[1], : vol.shape[2], : vol.shape[3]] = False

        return NestedTensor(padded, mask)

    else:
        raise ValueError(f"not supported ndim={ndim}")


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

class NestedTensor(object):
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def is_main_process():
    return get_rank() == 0

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)
