import torch
import math
from crypten.cuda.cuda_tensor import CUDALongTensor

def bitpack(t, bitwidth):
    t &= (2 ** bitwidth - 1)
    shape = t.shape
    t = t.flatten()
    total_size = t.shape[0]
    pack_ratio = math.floor(64 / bitwidth)
    size_after_bitpack = math.ceil(total_size / pack_ratio)
    padding_size = size_after_bitpack * pack_ratio - total_size
    if isinstance(t, CUDALongTensor):
        t = CUDALongTensor.cat([t, CUDALongTensor(torch.zeros(padding_size, dtype=torch.int).to(t.device))])
        t = t.view(pack_ratio, -1)
        t2 = CUDALongTensor.stack([t[i] << (bitwidth * i) for i in range(pack_ratio)])
    else:
        t = torch.cat([t, torch.zeros(padding_size, dtype=t.dtype).to(t.device)])
        t = t.view(pack_ratio, -1)
        t2 = torch.stack([t[i] << (bitwidth * i) for i in range(pack_ratio)])
    return t2.sum(dim=0), shape

def bitunpack(t, bitwidth, shape):
    pack_ratio = math.floor(64 / bitwidth)
    total_size = math.prod(shape)
    size_after_bitpack = math.ceil(total_size / pack_ratio)
    padding_size = size_after_bitpack * pack_ratio - total_size
    if isinstance(t, CUDALongTensor):
        t2 = CUDALongTensor.stack([(t >> (bitwidth * i)) & ((2 ** bitwidth) - 1) for i in range(pack_ratio)])
    else:
        t2 = torch.stack([(t >> (bitwidth * i)) & ((2 ** bitwidth) - 1) for i in range(pack_ratio)])
    if padding_size > 0:
        return t2.view(-1)[:-padding_size].view(shape)
    else:
        return t2.view(shape)
