import torch
from torch import nn

import argparse
import math
import json
import types

from typing import List
from tqdm import tqdm

import lm_eval
import lm_eval.models

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.tokenization_utils import PreTrainedTokenizer

from spatten_llm.utils import load
from spatten_llm.kv_cache_token_pruning import SpAttenStaticCache, SpAttenDynamicCache, SpAttenCache
from spatten_llm.pos_shift.modify_llama import enable_llama_spatten, SpAttenConfig

from lmquant.llm.eval import LlmEvalConfig

from dataclasses import dataclass
from typing import Callable, Optional




def visualize(tokenizer: PreTrainedTokenizer, input_ids: list, pruned: list):
    last_pruned = False
    tokens = []
    result = ""
    def decode():
        nonlocal tokens
        nonlocal result
        if len(tokens) == 0:
            return
        print(f"{last_pruned} => {tokens}")
        s = tokenizer.decode(
            tokens, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            spaces_between_special_tokens=False,)
        if last_pruned:
            result += '<span style="color:gray">' + s + '</span>'
        else:
            result += s
        tokens = []
    print(input_ids)
    print(pruned)
    for i, token in enumerate(input_ids):
        current_pruned = i < len(pruned) and pruned[i]
        if current_pruned != last_pruned:
            decode()
            last_pruned = current_pruned
        tokens.append(token)
    decode()
    print(result)


def main(args):
    model_name_or_path = args.model_name_or_path
    model, tokenizer = load(model_name_or_path)

    max_seq_length = args.max_seq_length

    config = SpAttenConfig(
        sparsity=args.sparsity,
        target_ntokens=(args.sparsity * max_seq_length if args.target_ntokens is None else args.target_ntokens),
        policy=args.policy,
        per_layer=not args.cascade,
        threshold=args.threshold,
    )
    if args.visualize:
        config.spatten_visualizer = lambda input_ids, pruned: visualize(tokenizer, input_ids, pruned)

    if args.baseline:
        print("Evaluating Baseline")
    else:
        print(f"Evaluating SpAtten with config={config}")
        enable_llama_spatten(model, config)

    if args.lmquant_tasks and len(args.lmquant_tasks) >= 1:
        print(f"Evaluating perplexity on {args.lmquant_tasks} using LMQuant, max_seq_length={max_seq_length}")
        LlmEvalConfig(max_seq_length=max_seq_length, tasks=args.lmquant_tasks).evaluate(model, tokenizer=tokenizer, model_name="spatten")
    else:
        print(f"Evaluating accuracy on {args.lmeval_tasks} using lm_eval")
        lm = lm_eval.models.huggingface.HFLM(pretrained=model, tokenizer=tokenizer)
        result = lm_eval.simple_evaluate(lm, tasks=args.lmeval_tasks, bootstrap_iters=0, verbosity="DEBUG", num_fewshot=args.shots)
        print(json.dumps(result["results"], sort_keys=True, indent=4, default=str))
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="lmsys/vicuna-13b-v1.3"
    )
    parser.add_argument(
        "--sparsity", type=float, default=0.5
    )
    parser.add_argument("--target_ntokens", type=int)
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--cascade", action="store_true")
    parser.add_argument("--threshold", type=float, default=1e-5)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--lmeval_tasks", type=str, nargs='+', default=["piqa", "copa", "openbookqa", "winogrande", "mathqa", "hellaswag", "arc_easy", "arc_challenge"])
    parser.add_argument("--lmquant_tasks", type=str, nargs='+')

    args = parser.parse_args()
    main(args)
