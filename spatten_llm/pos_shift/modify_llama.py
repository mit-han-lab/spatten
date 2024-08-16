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
from transformers.utils import logging

from spatten_llm.kv_cache_token_pruning import SpAttenCache, SpAttenStaticCache, SpAttenDynamicCache

import types
from dataclasses import dataclass
from typing import Callable, Optional

from tqdm import tqdm

from transformers.modeling_outputs import CausalLMOutputWithPast

import gc


__all__ = ["enable_llama_spatten"]

logger = logging.get_logger(__name__)

@dataclass
class SpAttenConfig:
    sparsity: float         # for "ratio" policy
    target_ntokens: int     # for "spatten" and "streaming" policy
    policy: str
    per_layer: bool
    threshold: float
    visualizer: Callable = None
    context_stage_step_by_step: bool = True


@torch.no_grad()
def spatten_logits(config, 
                   spatten_config: SpAttenConfig, 
                   forward: Callable, 
                   input_ids: torch.Tensor, 
                   past_key_values: SpAttenCache | None = None, 
                   labels: Optional[torch.LongTensor] = None,
                   output_attentions: Optional[bool] = None,
                   output_hidden_states: Optional[bool] = None,
                   return_dict: Optional[bool] = None,
                   use_cache: Optional[bool] = None,
                   **kwargs):

    print(f"input_ids.shape={input_ids.shape}")
    print(f"input_ids={input_ids}")
    print(f"config={spatten_config}")
    print(f"kwargs={kwargs}", flush=True)

    (bsz, ntokens) = input_ids.shape

    if not spatten_config.context_stage_step_by_step and ntokens > 1:
        return forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
            **kwargs
        )

    assert labels is None
    assert not output_attentions
    assert not output_hidden_states
    assert return_dict is None or return_dict

    # raise 233
    
    past_ntokens = 0

    if past_key_values is None:
        past_key_values = SpAttenStaticCache(config, bsz, ntokens, device=input_ids.device, dtype=torch.float16)
        # past_key_values = SpAttenDynamicCache()
        past_key_values.init_spatten_cache(per_layer=spatten_config.per_layer)
        past_key_values.value_pruning_threshold = spatten_config.threshold
    else:
        assert isinstance(past_key_values, SpAttenCache)
        past_ntokens = past_key_values.get_num_tokens(0)
    
    logits = None
    print(torch.cuda.memory_allocated(0))
    for step in tqdm(range(ntokens)):
        outputs = forward(
            input_ids=input_ids[:, step:step+1],
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        if logits is None:
            logits = outputs.logits
        else:
            assert logits.ndim == 3
            assert logits.shape[1] == step
            logits = torch.cat([logits, outputs.logits], dim=1)

        for layer, score in past_key_values.importance_scores.items():
            if spatten_config.policy == "spatten":
                # target_ntokens = math.ceil(ntokens * (1 - math.sqrt(1 - sparsity)))
                past_key_values.apply_token_pruning(layer, 0, 0, spatten_config.target_ntokens)
            elif spatten_config.policy == "streaming":
                # target_ntokens = math.ceil(ntokens * (1 - math.sqrt(1 - sparsity)))
                past_key_values.apply_token_pruning(layer, 4, spatten_config.target_ntokens - 4, 0)
            elif spatten_config.policy == "spatten_ratio_size":
                assert past_ntokens == 0
                target_ntokens = math.ceil(ntokens * spatten_config.sparsity)
                past_key_values.apply_token_pruning(layer, 0, 0, target_ntokens)
            elif spatten_config.policy == "streaming_ratio_access":
                assert past_ntokens == 0
                target_ntokens = math.ceil(ntokens * spatten_config.sparsity)
                past_key_values.apply_token_pruning(layer, 4, target_ntokens - 4, 0)
            elif spatten_config.policy == "ratio":
                target_ntokens = max(4, math.ceil((past_ntokens + step) * spatten_config.sparsity))
                if past_key_values.get_num_kept_tokens(layer) > target_ntokens:
                    past_key_values.apply_token_pruning(layer, 0, 0, target_ntokens)
            elif spatten_config.policy == "none":
                pass
            else:
                assert False
            if step % 64 == 0 and not spatten_config.visualizer is None:
                if layer in past_key_values.pruned:
                    print(f"Step {step} Layer {layer}")
                    spatten_config.visualizer(input_ids[0].tolist(), past_key_values.pruned[layer][0].tolist())
        # if step % 50 == 0:
        #     print(torch.cuda.memory_allocated(0))
        #     gc.collect()
        #     torch.cuda.empty_cache()
        #     print(torch.cuda.memory_allocated(0))

    print(f"KVCache key accessed = {past_key_values.key_access_count} ({past_key_values.key_access_count / past_key_values.kv_dense_access_count})")
    print(f"KVCache val accessed = {past_key_values.value_access_count} ({past_key_values.value_access_count / past_key_values.kv_dense_access_count})")

    return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

def forward_wrapper(self, input_ids: torch.Tensor, *args, **kwargs):
    for arg in args:
        print(f"Warning: ignore arg: {arg}")
    # for name, arg in kwargs.items():
    #     print(f"Warning: ignore arg {name}: {arg}")
    return spatten_logits(
        config=self.config, 
        spatten_config=self.spatten_config,
        forward=self._forward_orig, 
        input_ids=input_ids, 
        **kwargs)


def llama_spatten_forward(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[SpAttenCache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
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

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    ### SpAtten ###
    if past_key_value is not None and cache_position.numel() == 1:
        attn_weights[:, :, :, :cache_position+1] = past_key_value.apply_key_mask(self.layer_idx, attn_weights[:, :, :, :cache_position+1])
    ### SpAtten ###

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

    ### SpAtten ###
    if past_key_value is not None and cache_position.numel() == 1:
        past_key_value.update_attn_score(self.layer_idx, attn_weights[:, :, :, :cache_position+1])
        attn_weights[:, :, :, :cache_position+1] = past_key_value.apply_value_mask(self.layer_idx, attn_weights[:, :, :, :cache_position+1], threshold=past_key_value.value_pruning_threshold) # , num_fetch_values=max(4, math.ceil(cache_position * 0.5)))
    ### SpAtten ###

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

    # bsz, q_len, _ = hidden_states.size()

    # if self.config.pretraining_tp > 1:
    #     key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
    #     query_slices = self.q_proj.weight.split(
    #         (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
    #     )
    #     key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
    #     value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

    #     query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
    #     query_states = torch.cat(query_states, dim=-1)

    #     key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
    #     key_states = torch.cat(key_states, dim=-1)

    #     value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
    #     value_states = torch.cat(value_states, dim=-1)

    # else:
    #     query_states = self.q_proj(hidden_states)
    #     key_states = self.k_proj(hidden_states)
    #     value_states = self.v_proj(hidden_states)

    # query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    # value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # cos, sin = self.rotary_emb(value_states, position_ids)
    # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # if past_key_value is not None:
    #     # sin and cos are specific to RoPE models; cache_position needed for the static cache
    #     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    #     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # key_states = repeat_kv(key_states, self.num_key_value_groups)
    # value_states = repeat_kv(value_states, self.num_key_value_groups)

    # # print(f"key_states.shape={key_states.shape} value_states.shape={value_states.shape} query_states.shape={query_states.shape}")

    # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    # if attention_mask is not None:  # no matter the length, we just slice it
    #     causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
    #     attn_weights = attn_weights + causal_mask

    # ### SpAtten ###
    # if past_key_value is not None:
    #     attn_weights[:, :, :, :cache_position+1] = past_key_value.apply_key_mask(self.layer_idx, attn_weights[:, :, :, :cache_position+1])
    # ### SpAtten ###

    # # upcast attention to fp32
    # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

    # ### SpAtten ###
    # if past_key_value is not None:
    #     past_key_value.update_attn_score(self.layer_idx, attn_weights[:, :, :, :cache_position+1])
    #     attn_weights[:, :, :, :cache_position+1] = past_key_value.apply_value_mask(self.layer_idx, attn_weights[:, :, :, :cache_position+1], threshold=past_key_value.value_pruning_threshold) # , num_fetch_values=max(4, math.ceil(cache_position * 0.5)))
    # ### SpAtten ###
    
    # attn_output = torch.matmul(attn_weights, value_states)

    # if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
    #     raise ValueError(
    #         f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
    #         f" {attn_output.size()}"
    #     )

    # attn_output = attn_output.transpose(1, 2).contiguous()

    # attn_output = attn_output.reshape(bsz, q_len, -1)

    # if self.config.pretraining_tp > 1:
    #     attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
    #     o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
    #     attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    # else:
    #     attn_output = self.o_proj(attn_output)

    # if not output_attentions:
    #     attn_weights = None

    # return attn_output, attn_weights, past_key_value

def enable_llama_spatten(model, config: SpAttenConfig):
    def patch_module(m):
        for name, module in reversed(m._modules.items()):
            if len(list(module.children())) > 0:
                patch_module(
                    module
                )

            if isinstance(module, LlamaAttention):
                m._modules[name].forward = types.MethodType(
                    llama_spatten_forward, m._modules[name]
                )
    
    patch_module(model)

    model.spatten_config = config
    model._forward_orig = model.forward
    model.forward = types.MethodType(forward_wrapper, model)



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


