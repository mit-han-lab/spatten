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

from spatten_llm.utils import load
from spatten_llm.kv_cache_token_pruning import SpAttenStaticCache
from spatten_llm.pos_shift.modify_llama import enable_llama_spatten

from lmquant.llm.eval import LlmEvalConfig

@torch.no_grad()
def spatten_logits(config, forward, input_ids):
    print(input_ids.shape)
    print(input_ids)
    (bsz, ntokens) = input_ids.shape
    past_key_values = SpAttenStaticCache(config, bsz, ntokens, device=input_ids.device, dtype=torch.float16)
    # past_key_values = SpAttenDynamicCache()
    past_key_values.init_spatten_cache()
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
            target_ntokens = max(4, math.ceil(step * 0.5))
            if past_key_values.get_num_kept_tokens(layer) > target_ntokens:
                past_key_values.apply_token_pruning(layer, 0, 0, target_ntokens)
                # past_key_values.reset_scores(layer)
        # if step % 20 == 0:
        #     # print(torch.cuda.memory_allocated(0))
        #     gc.collect()
        #     torch.cuda.empty_cache()
        #     # print(torch.cuda.memory_allocated(0))
    return logits

class SpAttenLMEval(lm_eval.models.huggingface.HFLM):
    def __init__(self, model, tokenizer, batch_size=1, max_length=None):
        model.eval()
        super().__init__(
            pretrained=model,
            tokenizer=tokenizer,
            max_length=max_length,
            batch_size=batch_size)
    
    def _model_call(self, inps, attn_mask=None, labels=None):
        return spatten_logits(self.model.config, self.model.forward, inps)
    
def forward_wrapper(self, *args, **kwargs):
    for arg in args[1:]:
        print(f"Warning: ignore arg: {arg}")
    for name, arg in kwargs.items():
        print(f"Warning: ignore arg {name}: {arg}")
    logits = spatten_logits(self.config, self._forward_orig, args[0])
    return CausalLMOutputWithPast(logits=logits)

def main(args):
    model_name_or_path = args.model_name_or_path
    model, tokenizer = load(model_name_or_path)

    # lm_ref = lm_eval.models.huggingface.HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)
    # # lm_ref = SpAttenEval(model=model, tokenizer=tokenizer, batch_size=1, use_token_pruning=False)
    # result_ref = lm_eval.simple_evaluate(lm_ref, tasks=["wikitext"], bootstrap_iters=0, verbosity="DEBUG")
    # print(json.dumps(result_ref["results"], sort_keys=True, indent=4, default=str))

    # LlmEvalConfig().evaluate(model, tokenizer=tokenizer, model_name="llama2-7b")

    enable_llama_spatten(model)

    # lm = lm_eval.models.huggingface.HFLM(pretrained=model, tokenizer=tokenizer)
    # lm = SpAttenEval(model=model, tokenizer=tokenizer, batch_size=2)
    # result = lm_eval.simple_evaluate(lm, tasks=["wikitext"], bootstrap_iters=0, verbosity="DEBUG")
    # print(json.dumps(result["results"], sort_keys=True, indent=4, default=str))

    # wrapper = SpAttenLMQuantWrapper(model)
    model._forward_orig = model.forward
    model.forward = types.MethodType(forward_wrapper, model)
    LlmEvalConfig(max_seq_length=4096).evaluate(model, tokenizer=tokenizer, model_name="llama2-7b")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="lmsys/vicuna-13b-v1.3"
    )

    args = parser.parse_args()
    main(args)
