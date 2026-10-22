"""Bound Janus vision SDPA memory without changing its attention boundaries."""
from __future__ import annotations

from collections import defaultdict
from types import MethodType

import torch
from torch.nn import functional as F


def segmented_vision_sdpa(
    self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None
):
    from qwen_vl.model.modeling_qwen2_5_vl import apply_rotary_pos_emb_vision

    length = hidden_states.shape[0]
    q, k, v = (
        self.qkv(hidden_states)
        .reshape(length, 3, self.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    if position_embeddings is None:
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
    q, k = apply_rotary_pos_emb_vision(q, k, *position_embeddings)

    # Upstream's dense mask permits attention only inside these segments.
    # Equal-length segments can be a batch, enabling fused 4D SDPA without
    # allocating a quadratic mask across all historical frames or windows.
    sizes = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    query = torch.split(q, sizes)
    key = torch.split(k, sizes)
    value = torch.split(v, sizes)
    groups = defaultdict(list)
    for index, size in enumerate(sizes):
        groups[size].append(index)
    output = [None] * len(sizes)
    for indices in groups.values():
        batch_q, batch_k, batch_v = [
            torch.stack([chunks[index] for index in indices]).transpose(1, 2)
            for chunks in (query, key, value)
        ]
        attended = F.scaled_dot_product_attention(
            batch_q, batch_k, batch_v, dropout_p=0.0
        ).transpose(1, 2)
        for offset, index in enumerate(indices):
            output[index] = attended[offset]
    return self.proj(torch.cat(output, dim=0).reshape(length, -1))


def install_segmented_vision_sdpa(model):
    for block in model.visual.blocks:
        if type(block.attn).__name__ == "Qwen2_5_VLVisionSdpaAttention":
            block.attn.forward = MethodType(segmented_vision_sdpa, block.attn)
