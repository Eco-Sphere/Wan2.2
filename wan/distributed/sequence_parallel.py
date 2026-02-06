# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import torch
import torch.cuda.amp as amp
from .parallel_mgr import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
from ..modules.attn_layer import xFuserLongContextAttention

from ..modules.model import sinusoidal_embedding_1d
from wan.utils.rainfusion import Rainfusion
from mindiesd import rotary_position_embedding

import torch.distributed as dist
from .comm import all_to_all_4D

import os


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    if pad_size == 0:
        return original_tensor
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=torch.float32,
        device=original_tensor.device
        ).to(original_tensor.dtype)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


@torch.amp.autocast('npu', enabled=False)
def rope_apply(x, grid_sizes, freqs_list):
    cos, sin = freqs_list[0]
    return rotary_position_embedding(x, cos, sin, rotated_mode="rotated_interleaved", fused=True)


def sp_dit_forward(
    self,
    x,
    t,
    context,
    seq_len,
    y=None,
    t_idx=None,
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    if self.rainfusion_config and self.rainfusion_config["atten_mask_all"] is None:
        self.rainfusion_config["grid_size"] = Rainfusion.get_grid_size(x[0].shape, self.patch_size)
        logging.info(f"Rainfusion grid size: {self.rainfusion_config['grid_size']}")
        self.rainfusion_config["atten_mask_all"] = Rainfusion.get_atten_mask(
            grid_size=self.rainfusion_config["grid_size"],
            sparsity=self.rainfusion_config["sparsity"]
        )
    if self.model_type == 'i2v':
        assert y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    # time embeddings
    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim,
                                    t).unflatten(0, (bt, seq_len)).float())
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    # Context Parallel
    x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    e = torch.chunk(e, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    e0 = torch.chunk(e0, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]


    if self.freqs_list is None:
        c = (self.dim // self.num_heads) // 2
        s = x.shape[1]
        freqs = self.freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        freqs_list = []

        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            seq_len = f * h * w
            freqs_i = torch.cat([
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ],
            dim=-1).reshape(seq_len, 1, -1)

            # apply rotary embedding
            sp_size = get_sequence_parallel_world_size()
            sp_rank = get_sequence_parallel_rank()
            freqs_i = pad_freqs(freqs_i, s * sp_size)
            s_per_rank = s
            freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) * s_per_rank), :, :]
            cos, sin = torch.chunk(torch.view_as_real(freqs_i_rank.to(torch.complex64)), 2, dim=-1)
            cos = cos.unsqueeze(0).expand(-1, -1, -1, -1, 2).flatten(-2).to(x.dtype)
            sin = sin.unsqueeze(0).expand(-1, -1, -1, -1, 2).flatten(-2).to(x.dtype)
            freqs_i_rank = (cos, sin)
            freqs_list.append(freqs_i_rank)
        self.freqs_list = freqs_list

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs_list,
        context=context,
        context_lens=context_lens,
        rainfusion_config=self.rainfusion_config,
        t_idx=t_idx,
    )

    block_idx = 0
    for block in self.blocks:
        kwargs['block_idx'] = block_idx
        x = block(x, **kwargs)
        block_idx += 1

    # head
    x = self.head(x, e)

    # Context Parallel
    x = get_sp_group().all_gather(x, dim=1)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]

def proj_alltoall(x, proj_func, grid_sizes, freqs, shape, async_op=True, norm_func=None):
    b, s, n, d = shape
    _input = None

    # QK
    if norm_func:
        _input = norm_func(proj_func(x)).view(b, s, n, d)
        _input = rope_apply(_input, grid_sizes, freqs)
    # V
    else:
        _input = proj_func(x).view(b, s, n, d)

    ulysses_pg = get_sp_group().ulysses_group
    seq_world_size = dist.get_world_size(ulysses_pg)
    bs, shard_seqlen, hc, hs = _input.shape
    seqlen = shard_seqlen * seq_world_size
    shard_hc = hc // seq_world_size

    if async_op:
        output, async_handler = all_to_all_4D(input_=_input, scatter_idx=2, gather_idx=1, group=ulysses_pg, async_op=async_op)
    else:
        output = all_to_all_4D(input_=_input, scatter_idx=2, gather_idx=1, group=ulysses_pg, async_op=async_op)
    def wait():
        nonlocal output, async_handler, async_op, seqlen, bs, shard_hc, hs
        if async_op:
            async_handler.wait()
            # if scattering the seq-dim, transpose the heads back to the original dimension
            output = output.reshape(seqlen, bs, shard_hc, hs)

            # (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
            output = output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)

        return output
    
    return wait

def sp_attn_forward(self, x, seq_lens, grid_sizes, freqs, args, dtype=torch.bfloat16, rainfusion_config=None, t_idx=None, **kwargs):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    async_op = int(os.getenv("ENABLE_ASYNC_QKV", 0))
    if not async_op:
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)
    else:
        q = proj_alltoall(x, self.q, grid_sizes=grid_sizes, freqs=freqs, shape=(b, s, n, d), norm_func=self.norm_q, async_op=async_op)
        k = proj_alltoall(x, self.k, grid_sizes=grid_sizes, freqs=freqs, shape=(b, s, n, d), norm_func=self.norm_k, async_op=async_op)
        q = q()
        v = proj_alltoall(x, self.v, grid_sizes=grid_sizes, freqs=freqs, shape=(b, s, n, d), async_op=async_op)
        k = k()
        v = v()

    kwargs['async_op'] = async_op

    x = xFuserLongContextAttention(args, rainfusion_config=rainfusion_config, fa_quant=getattr(self, 'fa_quant', None))(
        None,
        query=half(q),
        key=half(k),
        value=half(v),
        seq_lens=seq_lens,
        window_size=self.window_size,
        t_idx=t_idx,
        **kwargs
    )

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x
