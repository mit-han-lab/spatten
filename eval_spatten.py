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

    if False:
        lm_ref = lm_eval.models.huggingface.HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)
        # lm_ref = SpAttenEval(model=model, tokenizer=tokenizer, batch_size=1, use_token_pruning=False)
        result_ref = lm_eval.simple_evaluate(lm_ref, tasks=["wikitext"], limit=1, bootstrap_iters=0, verbosity="DEBUG")
        print(json.dumps(result_ref["results"], sort_keys=True, indent=4, default=str))

    max_seq_length = args.max_seq_length
    print(f"max_seq_length={max_seq_length}")

    # if args.baseline:
    #     LlmEvalConfig(max_seq_length=max_seq_length).evaluate(model, tokenizer=tokenizer, model_name="llama2-7b")

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
        lm = lm_eval.models.huggingface.HFLM(pretrained=model, tokenizer=tokenizer)
        result = lm_eval.simple_evaluate(lm, tasks=args.lmeval_tasks, bootstrap_iters=0, verbosity="DEBUG", num_fewshot=args.shots)
        print(json.dumps(result["results"], sort_keys=True, indent=4, default=str))
        # return

    enable_llama_spatten(model, config)

    if False:
        # lm = lm_eval.models.huggingface.HFLM(pretrained=model, tokenizer=tokenizer)
        lm = SpAttenLMEval(model=model, tokenizer=tokenizer, batch_size=2)
        result = lm_eval.simple_evaluate(lm, tasks=["wikitext"], limit=1, bootstrap_iters=0, verbosity="DEBUG")
        print(json.dumps(result["results"], sort_keys=True, indent=4, default=str))
        return
    
    if True:
        lm = lm_eval.models.huggingface.HFLM(pretrained=model, tokenizer=tokenizer)
        result = lm_eval.simple_evaluate(lm, tasks=args.lmeval_tasks, bootstrap_iters=0, verbosity="DEBUG", num_fewshot=args.shots)
        print(json.dumps(result["results"], sort_keys=True, indent=4, default=str))
        return

    # wrapper = SpAttenLMQuantWrapper(model)
    
    

    past_key_values = SpAttenDynamicCache()
    past_key_values.init_spatten_cache(config.per_layer)
    past_key_values.value_pruning_threshold = config.threshold
    print(model.generate(
            torch.tensor([[128000]], dtype=torch.int).cuda(),
            max_new_tokens=128,
            use_cache=True,
            return_dict_in_generate=True,
            past_key_values=past_key_values))
    
    LlmEvalConfig(max_seq_length=max_seq_length).evaluate(model, tokenizer=tokenizer, model_name="llama2-7b")


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

    args = parser.parse_args()
    main(args)
