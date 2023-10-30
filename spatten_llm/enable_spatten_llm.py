from spatten_llm.kv_cache_token_pruning import SpAttenKVCache

__all__ = ["enable_spatten_llm"]

def enable_spatten_llm(model, start_size, important_size, recent_size):
    if "llama" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
        from spatten_llm.pos_shift.modify_llama import (
            enable_llama_pos_shift_attention,
        )

        enable_llama_pos_shift_attention(model)
    else:
        raise ValueError(f"got {model.config.model_type}")
    
    kv_cache = SpAttenKVCache(
        start_size=start_size,
        important_size=important_size,
        recent_size=recent_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
    )
    return kv_cache
