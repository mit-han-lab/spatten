import torch


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


class SpAttenKVCache:
    def __init__(
        self,
        start_size=4,
        recent_size=128,
        important_size=128,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        print(f"SpAttenKVCache: keep start: {start_size}, keep recent: {recent_size}, keep important: {important_size}")
        self.start_size = start_size
        self.recent_size = recent_size
        self.important_size = important_size
        self.cache_size = start_size + important_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

    def apply_token_pruning(self, past_key_values, num_coming, attn_score_all):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values

            
        # apply SpAtten importance_score based token pruning
        self.importance_score = [item.sum(0).sum(1) for item in attn_score_all]
        
        bsz, num_heads, _, head_dim = past_key_values[0][0].shape
        kv_important = []
        for layer_idx, score in enumerate(self.importance_score):
            keep_important = None
            if self.important_size > 0:
                # select the important ones for tokens in the middle
                important_score_candidates = score[:, self.start_size:seq_len-self.recent_size+num_coming]
                _, keep_important = torch.topk(important_score_candidates, self.important_size, dim=-1)
                keep_important = keep_important.sort().values
            
            keep_important += self.start_size
            keep_important = keep_important.to(past_key_values[0][0].device)
            mask = torch.zeros(score.shape, dtype=torch.bool).to(past_key_values[0][0].device)
            mask = mask.scatter(-1, keep_important, 1).cpu()

            k_important = past_key_values[layer_idx][0].squeeze()[mask].view(bsz, num_heads, -1, head_dim)
            v_important = past_key_values[layer_idx][1].squeeze()[mask].view(bsz, num_heads, -1, head_dim)
            kv_important.append((k_important, v_important))
                
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        k_important,
                        self.k_slice(
                            k, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        v_important,
                        self.v_slice(
                            v, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for (k, v), (k_important, v_important) in zip(past_key_values, kv_important)
        ]
