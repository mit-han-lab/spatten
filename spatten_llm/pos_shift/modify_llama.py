import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.utils.checkpoint

import torch.nn.functional as F

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
)
from spatten_llm.kv_cache_token_pruning import SpAttenCache
# from transformers.cache_utils import Cache
import types

__all__ = ["enable_llama_spatten"]

def llama_spatten_forward(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[SpAttenCache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,):

    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # print(f"key_states.shape={key_states.shape} value_states.shape={value_states.shape} query_states.shape={query_states.shape}")

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

    if past_key_value is not None:
        past_key_value.update_attn_score(self.layer_idx, attn_weights[:, :, :, :cache_position+1])
        attn_weights = past_key_value.apply_mask(self.layer_idx, attn_weights)
    
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, -1)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def enable_llama_spatten(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_llama_spatten(
                module,
            )

        if isinstance(module, LlamaAttention):
            model._modules[name].forward = types.MethodType(
                llama_spatten_forward, model._modules[name]
            )



# def apply_rotary_pos_emb_single(x, cos, sin, unsqueeze_dim=1):
#     # # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
#     # cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
#     # sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
#     # cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
#     # sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
#     # # print(cos.shape)
#     # x_embed = (x * cos) + (rotate_half(x) * sin)

#     cos = cos.unsqueeze(unsqueeze_dim)
#     sin = sin.unsqueeze(unsqueeze_dim)
#     x_embed = (x * cos) + (rotate_half(x) * sin)
#     return x_embed


# def llama_pos_shift_attention_forward(
#     self: LlamaAttention,
#     hidden_states: torch.Tensor,
#     attention_mask: Optional[torch.Tensor] = None,
#     position_ids: Optional[torch.LongTensor] = None,
#     past_key_value: Optional[SpAttenCache] = None,
#     output_attentions: bool = False,
#     use_cache: bool = False,
#     cache_position: Optional[torch.LongTensor] = None,
# ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#     bsz, q_len, _ = hidden_states.size()

#     if self.config.pretraining_tp > 1:
#         key_value_slicing = (
#             self.num_key_value_heads * self.head_dim
#         ) // self.config.pretraining_tp
#         query_slices = self.q_proj.weight.split(
#             (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
#         )
#         key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
#         value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

#         query_states = [
#             F.linear(hidden_states, query_slices[i])
#             for i in range(self.config.pretraining_tp)
#         ]
#         query_states = torch.cat(query_states, dim=-1)

#         key_states = [
#             F.linear(hidden_states, key_slices[i])
#             for i in range(self.config.pretraining_tp)
#         ]
#         key_states = torch.cat(key_states, dim=-1)

#         value_states = [
#             F.linear(hidden_states, value_slices[i])
#             for i in range(self.config.pretraining_tp)
#         ]
#         value_states = torch.cat(value_states, dim=-1)

#     else:
#         query_states = self.q_proj(hidden_states)
#         key_states = self.k_proj(hidden_states)
#         value_states = self.v_proj(hidden_states)

#     query_states = query_states.view(
#         bsz, q_len, self.num_heads, self.head_dim
#     ).transpose(1, 2)
#     key_states = key_states.view(
#         bsz, q_len, self.num_key_value_heads, self.head_dim
#     ).transpose(1, 2)
#     value_states = value_states.view(
#         bsz, q_len, self.num_key_value_heads, self.head_dim
#     ).transpose(1, 2)

#     kv_seq_len = key_states.shape[-2]
#     if past_key_value is not None:
#         kv_seq_len += past_key_value.get_seq_length(self.layer_idx)
#     cos, sin = self.rotary_emb(value_states, position_ids)
#     ### Shift Pos: query pos is min(cache_size, idx)
#     # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
#     query_states = apply_rotary_pos_emb_single(query_states, cos, sin)
#     ###

#     # print(f"position_ids.shape={position_ids.shape}")
#     # print(f"key_states.shape={key_states.shape} value_states.shape={value_states.shape} query_states.shape={query_states.shape}")
#     # print(f"cos.shape={cos.shape}")

#     if past_key_value is not None:
#         # reuse k, v, self_attention
#         cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
#         key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
#         # key_states = torch.cat([past_key_value[0], key_states], dim=2)
#         # value_states = torch.cat([past_key_value[1], value_states], dim=2)

#     # past_key_value = (key_states, value_states) if use_cache else None
#     # print(f"key_states.shape={key_states.shape} value_states.shape={value_states.shape}")

#     ### Shift Pos: key pos is the pos in cache
#     key_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0)
#     cos, sin = self.rotary_emb(value_states, key_position_ids)
#     key_states = apply_rotary_pos_emb_single(key_states, cos, sin)
#     ###

#     # repeat k/v heads if n_kv_heads < n_heads
#     key_states = repeat_kv(key_states, self.num_key_value_groups)
#     value_states = repeat_kv(value_states, self.num_key_value_groups)

#     # print(f"key_states.shape={key_states.shape} value_states.shape={value_states.shape} query_states.shape={query_states.shape}")

#     attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
#         self.head_dim
#     )

#     print(f"attn_weights={attn_weights.shape}")

#     # store attention scores for deciding which token to prune
#     if hasattr(self, "attn_scores"):
#         self.attn_scores = attn_weights.detach().clone()
#     else:
#         setattr(self, "attn_scores", attn_weights.detach().clone())

#     if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
#         raise ValueError(
#             f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
#             f" {attn_weights.size()}"
#         )

#     if attention_mask is not None:
#         # if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
#         #     raise ValueError(
#         #         f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
#         #     )
#         causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
#         attn_weights = attn_weights + causal_mask
#         # attn_weights = attn_weights + attention_mask

#     # upcast attention to fp32
#     attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
#         query_states.dtype
#     )
#     attn_output = torch.matmul(attn_weights, value_states)

#     if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
#         raise ValueError(
#             f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
#             f" {attn_output.size()}"
#         )

#     attn_output = attn_output.transpose(1, 2).contiguous()
#     attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

#     if self.config.pretraining_tp > 1:
#         attn_output = attn_output.split(
#             self.hidden_size // self.config.pretraining_tp, dim=2
#         )
#         o_proj_slices = self.o_proj.weight.split(
#             self.hidden_size // self.config.pretraining_tp, dim=1
#         )
#         attn_output = sum(
#             [
#                 F.linear(attn_output[i], o_proj_slices[i])
#                 for i in range(self.config.pretraining_tp)
#             ]
#         )
#     else:
#         attn_output = self.o_proj(attn_output)

#     if not output_attentions:
#         attn_weights = None

#     return attn_output, attn_weights, past_key_value


# def enable_llama_pos_shift_attention(model):
#     # return
#     for name, module in reversed(model._modules.items()):
#         if len(list(module.children())) > 0:
#             enable_llama_pos_shift_attention(
#                 module,
#             )

#         if isinstance(module, LlamaAttention):
#             model._modules[name].forward = types.MethodType(
#                 llama_pos_shift_attention_forward, model._modules[name]
#             )


