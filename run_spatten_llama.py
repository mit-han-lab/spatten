import warnings

warnings.filterwarnings("ignore")

import torch
import argparse
import json
import os
import time
import re
import sys

from tqdm import tqdm
from spatten_llm.utils import load, download_url, load_jsonl
from spatten_llm.enable_spatten_llm import enable_spatten_llm
from transformers.models.llama.modeling_llama import LlamaAttention

@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len):
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    pos = 0
    for _ in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())
        generated_text = (
            tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
            .strip()
            .split(" ")
        )

        now = len(generated_text) - 1
        if now > pos:
            print(" ".join(generated_text[pos:now]), end=" ", flush=True)
            pos = now

        if pred_token_idx == tokenizer.eos_token_id:
            break
    print(" ".join(generated_text[pos:]), flush=True)
    return past_key_values


@torch.no_grad()
def spatten_inference(model, tokenizer, prompts, kv_cache=None, max_gen_len=64):
    past_key_values = None
    n_token_pruned_cumulative = 0
    for idx, prompt in enumerate(prompts):
        prompt = "USER: " + prompt + "\n\nASSISTANT: "
        print("\n" + prompt, end="")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)
        seq_len = input_ids.shape[1]

        if kv_cache is not None and idx > 0:
            space_needed = seq_len + max_gen_len
            
            attn_score_all = []
            for m in model.modules():
                if isinstance(m, LlamaAttention):
                    attn_score_all.append(m.attn_scores)
            n_token_prev = past_key_values[0][0].size(2)
            past_key_values = kv_cache.apply_token_pruning(past_key_values, space_needed, attn_score_all)
            n_token_remained = past_key_values[0][0].size(2)
            n_token_pruned = n_token_prev - n_token_remained
            n_token_pruned_cumulative += n_token_pruned
            print(f"N pruned token this round: {n_token_pruned}, cumulative: {n_token_pruned_cumulative}")

        past_key_values = greedy_generate(
            model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len
        )



def main(args):
    model_name_or_path = args.model_name_or_path
    model, tokenizer = load(model_name_or_path)
    test_filepath = os.path.join(args.data_root, "mt_bench.jsonl")
    print(f"Loading data from {test_filepath} ...")

    if not os.path.exists(test_filepath):
        download_url(
            "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
            args.data_root,
        )
        os.rename(os.path.join(args.data_root, "question.jsonl"), test_filepath)

    list_data = load_jsonl(test_filepath)
    prompts = []
    for sample in list_data:
        prompts += sample["turns"]

    if 1:
        kv_cache = enable_spatten_llm(
            model, 
            start_size=args.start_size, 
            important_size=args.important_size,
            recent_size=args.recent_size
        )
    else:
        kv_cache = None

    spatten_inference(
        model,
        tokenizer,
        prompts,
        kv_cache,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="lmsys/vicuna-13b-v1.3"
    )
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--enable_spatten", action="store_true")
    parser.add_argument("--start_size", type=int, default=0)
    parser.add_argument("--important_size", type=int, default=150)
    parser.add_argument("--recent_size", type=int, default=150)
    parser.add_argument("--pdb", action="store_true")
    args = parser.parse_args()
    
    if args.pdb:
        import pdb
        pdb.set_trace()
    
    main(args)
