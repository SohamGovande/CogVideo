import torch
import torch.distributed as dist
import os
import torch.nn as nn
from typing import List
from torch.distributed.tensor import Shard, Replicate
from torch.distributed.device_mesh import init_device_mesh

device_mesh = None

def all_gather_sp(tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor, dtype=tensor.dtype, device=tensor.device) for _ in range(world_size)]
    tensor = tensor.contiguous()
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=dim)

def shard(x, dim):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert x.size(dim=dim) % world_size == 0, f"x.size(dim={dim}) = {x.size(dim=dim)} is not divisible by world_size = {world_size}"
    return torch.tensor_split(x, world_size, dim=dim)[rank]


def gather_multiple_tensors_sequence_parallel(tensors: List[torch.Tensor], dim: int = 1) -> List[torch.Tensor]:
    gathered_tensors_list = []
    for tensor in tensors:
        gathered_tensors = [torch.zeros_like(tensor, dtype=tensor.dtype, device=tensor.device) for _ in range(dist.get_world_size())]
        tensor = tensor.contiguous()
        dist.all_gather(gathered_tensors, tensor)
        gathered_tensors_list.append(torch.cat(gathered_tensors, dim=dim))
    return gathered_tensors_list

def build_device_mesh():
    global device_mesh
    device_mesh = init_device_mesh("cuda", (dist.get_world_size(),))


def is_local():
    return dist.get_rank() == 0

def _apply_tp_linear(linear: nn.Linear, style: str) -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Linear's weight matrix is transposed, and is of shape
    # (linear.out_features, linear.in_features)
    dim_lookup = {
        "colwise": (0, "out_features"),
        "rowwise": (1, "in_features")
    }
    assert style in dim_lookup
    shard_dim, size_attr = dim_lookup[style]

    # ensure we can shard evenly
    assert getattr(linear, size_attr) % world_size == 0
    
    linear.weight = nn.Parameter(shard(linear.weight, shard_dim), requires_grad=False)
    setattr(linear, size_attr, getattr(linear, size_attr) // world_size)

    if style == "colwise":
        if hasattr(linear, "scales") and linear.scales is not None:
            linear.scales = shard(linear.scales, 0)
        if hasattr(linear, "bias") and linear.bias is not None:
            linear.bias = nn.Parameter(shard(linear.bias, shard_dim), requires_grad=False)
    
    # shape info should still be synced
    assert linear.weight.shape == (linear.out_features, linear.in_features)


def _apply_tp_ffn(mlp):
    assert hasattr(mlp, "net")
    assert len(mlp.net) == 4
    _apply_tp_linear(mlp.net[0].proj, 'colwise')
    _apply_tp_linear(mlp.net[2], 'rowwise')

    if not hasattr(mlp, '_tp_ffn_hook_registered'):
        def hook(module, _input, output):
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
        mlp.register_forward_hook(hook)
        mlp._tp_ffn_hook_registered = True


def _apply_tp_attn(attn):
    _apply_tp_linear(attn.to_q, 'colwise')
    _apply_tp_linear(attn.to_k, 'colwise')
    _apply_tp_linear(attn.to_v, 'colwise')
    _apply_tp_linear(attn.to_out[0], 'rowwise')

    world_size = dist.get_world_size()
    attn.heads //= world_size
    attn.query_dim //= world_size
    # attn.kv_dim //= world_size
    attn.inner_kv_dim //= world_size
    attn.cross_attention_dim //= world_size

    def hook(module, _input, output):
        if isinstance(output, tuple) and len(output) >= 2:
            dist.all_reduce(output[0][0], op=dist.ReduceOp.SUM)
            dist.all_reduce(output[0][1], op=dist.ReduceOp.SUM)
        else:
            dist.all_reduce(output[0], op=dist.ReduceOp.SUM)

    if not hasattr(attn, '_tp_attn_hook_registered'):
        attn.register_forward_hook(hook)
        attn._tp_attn_hook_registered = True

def apply_tp(transformer: nn.Module):
    dist.barrier()
    for block in transformer.transformer_blocks:
        _apply_tp_attn(block.attn1)
        _apply_tp_ffn(block.ff)
    dist.barrier()