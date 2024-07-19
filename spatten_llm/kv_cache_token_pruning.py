import torch
import torch.nn.functional as F
from transformers.cache_utils import StaticCache, DynamicCache
from typing import Any, Dict, List, Optional, Tuple, Union


def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end):
    return x[:, start:end, ...]


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}

class SpAttenCache(object):
    def init_spatten_cache(
        self,
        per_layer=False,
    ):
        # print("SpAttenCache init")
        # print(f"SpAttenKVCache: keep start: {start_size}, keep recent: {recent_size}, keep important: {important_size}")
        self.per_layer = per_layer

        # layer_idx -> [bsz, tokens]
        self.importance_scores: Dict[int, torch.Tensor] = {}
        self.importance_scores_update_cnt: Dict[int, torch.Tensor] = {}
        self.pruned: Dict[int, torch.Tensor] = {}
        # layer_idx -> int
        self.n_pruned: Dict[int, int] = {}

    @torch.no_grad()
    def update_attn_score(self, layer_idx: int, attn_score: torch.Tensor):
        if not self.per_layer and layer_idx != 0:
            return self.update_attn_score(0, attn_score)
        
        assert attn_score.ndim == 4
        (bsz, nheads, qlen, ntokens) = attn_score.shape

        # print(attn_score.shape)
        attn_score = attn_score.sum(2).sum(1)
        # print(attn_score.shape)
        attn_score_cnt = torch.full([bsz, ntokens], fill_value=nheads*qlen, dtype=torch.int32, device=attn_score.device)
        # print(attn_score_cnt.shape)
        assert attn_score.shape == attn_score_cnt.shape

        if not layer_idx in self.importance_scores:
            self.importance_scores[layer_idx] = attn_score.to(torch.float32)
            self.importance_scores_update_cnt[layer_idx] = attn_score_cnt
        else:
            assert bsz == self.importance_scores[layer_idx].shape[0]
            old_ntokens = self.importance_scores[layer_idx].shape[1]
            assert ntokens >= old_ntokens

            # print(self.importance_scores[layer_idx].shape)
            # print(f"{ntokens} {old_ntokens}")

            if ntokens > old_ntokens:
                self.importance_scores[layer_idx] = F.pad(
                    self.importance_scores[layer_idx], 
                    pad=(0, ntokens - old_ntokens), 
                    value=0)
                self.importance_scores_update_cnt[layer_idx] = F.pad(
                    self.importance_scores_update_cnt[layer_idx], 
                    pad=(0, ntokens - old_ntokens), 
                    value=0)
            assert self.importance_scores[layer_idx].shape == self.importance_scores_update_cnt[layer_idx].shape

            self.importance_scores[layer_idx] += attn_score
            self.importance_scores_update_cnt[layer_idx] += attn_score_cnt
    
    @torch.no_grad()
    def apply_token_pruning(self, layer_idx: int, start_size: int, recent_size: int, importance_size: int):
        assert start_size >= 0 and recent_size >= 0 and importance_size >= 0

        num_kept_tokens = start_size + recent_size + importance_size
        (bsz, ntokens) = self.importance_scores[layer_idx].shape
        if ntokens <= num_kept_tokens:
            return
        if layer_idx in self.n_pruned and ntokens - self.n_pruned[layer_idx] <= num_kept_tokens:
            return
        
        assert not torch.isinf(self.importance_scores[layer_idx]).any()
        assert not torch.isnan(self.importance_scores[layer_idx]).any()
        assert not (self.importance_scores_update_cnt[layer_idx] == 0).any()
        
        scores = self.importance_scores[layer_idx] / self.importance_scores_update_cnt[layer_idx]

        assert not torch.isinf(scores).any()
        assert not torch.isnan(scores).any()
        
        mask = torch.zeros_like(scores)
        if layer_idx in self.pruned:
            pruned = F.pad(self.pruned[layer_idx], pad=(0, ntokens - self.pruned[layer_idx].shape[-1]), value=False)
            mask[pruned] = float("-Inf")

        scores += mask
        scores[:, 0:start_size] += float("Inf")
        scores[:, scores.shape[-1] - recent_size:scores.shape[-1]] += float("Inf")
        # print(scores)
        assert not torch.isnan(scores).any()

        _, indexes = torch.topk(scores, num_kept_tokens, dim=-1, largest=True)
        indexes = indexes.sort().values.to(scores.device)

        self.pruned[layer_idx] = torch.ones_like(scores, dtype=torch.bool).scatter_(-1, indexes, 0)
        self.n_pruned[layer_idx] = ntokens - num_kept_tokens

        assert self.pruned[layer_idx].sum() == bsz * self.n_pruned[layer_idx]

        # print(f"Layer {layer_idx} n_pruned={self.n_pruned[layer_idx]} pruned={self.pruned[layer_idx]}")

    # attn_weights after softmax [bsz, nheads, qlen, ntokens]
    @torch.no_grad()
    def apply_mask(self, layer_idx: int, attn_weights: torch.Tensor, num_fetch_values: int = -1) -> torch.Tensor:
        (bsz, nheads, qlen, ntokens) = attn_weights.shape
        if not layer_idx in self.pruned:
            return attn_weights

        (old_bsz, old_ntokens) = self.pruned[layer_idx].shape
        assert bsz == old_bsz
        assert ntokens >= old_ntokens

        pruned = F.pad(self.pruned[layer_idx], pad=(0, ntokens - old_ntokens), value=0)
        attn_weights *= (1 - pruned.unsqueeze(1).unsqueeze(2).int())

        # Prune values
        if num_fetch_values >= 0 and qlen == 1:
            _, indexes = torch.topk(attn_weights, ntokens - num_fetch_values, dim=-1, largest=False)
            indexes = indexes.sort().values.to(attn_weights.device)
            attn_weights = attn_weights.scatter_(-1, indexes, 0)
        
        return attn_weights
    
    @torch.no_grad()
    def reset_scores(self, layer_idx: int):
        if layer_idx in self.importance_scores:
            self.importance_scores[layer_idx].fill_(0)
            self.importance_scores_update_cnt[layer_idx].fill_(0)
    
    def get_num_kept_tokens(self, layer_idx: int) -> int:
        if not layer_idx in self.importance_scores:
            return 0
        (bsz, ntokens) = self.importance_scores[layer_idx].shape
        if not layer_idx in self.n_pruned:
            return ntokens
        return ntokens - self.n_pruned[layer_idx]

class SpAttenStaticCache(StaticCache, SpAttenCache):
    pass

class SpAttenDynamicCache(DynamicCache, SpAttenCache):
    pass
# class SpAttenKVCache:
#     def __init__(
#         self,
#         start_size=4,
#         recent_size=128,
#         important_size=128,
#         k_seq_dim=2,
#         v_seq_dim=2,
#     ):
#         print(f"SpAttenKVCache: keep start: {start_size}, keep recent: {recent_size}, keep important: {important_size}")
#         self.start_size = start_size
#         self.recent_size = recent_size
#         self.important_size = important_size
#         self.cache_size = start_size + important_size + recent_size
#         self.k_seq_dim = k_seq_dim
#         self.v_seq_dim = v_seq_dim
#         self.k_slice = DIM_TO_SLICE[k_seq_dim]
#         self.v_slice = DIM_TO_SLICE[v_seq_dim]

#     def apply_token_pruning(self, past_key_values, num_coming, attn_score_all):
#         if past_key_values is None:
#             return None
#         seq_len = past_key_values[0][0].size(self.k_seq_dim)
#         if seq_len + num_coming <= self.cache_size:
#             return past_key_values

#         print(f"attn_score_all[0].shape={attn_score_all[0].shape}")
            
#         # apply SpAtten importance_score based token pruning
#         self.importance_score = [item.sum(0).sum(1) for item in attn_score_all]
        
#         bsz, num_heads, _, head_dim = past_key_values[0][0].shape
#         kv_important = []
#         for layer_idx, score in enumerate(self.importance_score):
#             keep_important = None
#             if self.important_size > 0:
#                 # select the important ones for tokens in the middle
#                 important_score_candidates = score[:, self.start_size:seq_len-self.recent_size+num_coming]
#                 _, keep_important = torch.topk(important_score_candidates, self.important_size, dim=-1)
#                 keep_important = keep_important.sort().values
            
#             keep_important += self.start_size
#             keep_important = keep_important.to(past_key_values[0][0].device)
#             mask = torch.zeros(score.shape, dtype=torch.bool).to(past_key_values[0][0].device)
#             mask = mask.scatter(-1, keep_important, 1).cpu()

#             k_important = past_key_values[layer_idx][0].squeeze()[mask].view(bsz, num_heads, -1, head_dim)
#             v_important = past_key_values[layer_idx][1].squeeze()[mask].view(bsz, num_heads, -1, head_dim)
#             kv_important.append((k_important, v_important))
                
#         return [
#             [
#                 torch.cat(
#                     [
#                         self.k_slice(k, 0, self.start_size),
#                         k_important,
#                         self.k_slice(
#                             k, seq_len - self.recent_size + num_coming, seq_len
#                         ),
#                     ],
#                     dim=self.k_seq_dim,
#                 ),
#                 torch.cat(
#                     [
#                         self.v_slice(v, 0, self.start_size),
#                         v_important,
#                         self.v_slice(
#                             v, seq_len - self.recent_size + num_coming, seq_len
#                         ),
#                     ],
#                     dim=self.v_seq_dim,
#                 ),
#             ]
#             for (k, v), (k_important, v_important) in zip(past_key_values, kv_important)
#         ]
