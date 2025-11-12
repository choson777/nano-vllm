from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.request import Request, RequestStatus
from nanovllm.engine.draft_engine import DraftEngine
from nanovllm.engine.target_engine import TargetEngine
from nanovllm.config import Config, DraftConfig, TargetConfig
from nanovllm import SamplingParams
from dataclasses import fields, asdict
from collections import defaultdict

from transformers import AutoTokenizer
import torch
import tqdm
from time import perf_counter

class SpeculativeEngine:
    
    def __init__(
        self,
        draft_model: str,
        target_model: str,
        num_speculative_tokens: int = 4,
        random_seed: int = 42,
        **kwargs
    ):
        if random_seed:
            torch.manual_seed(random_seed)
        # 提取 draft_ 和 target_ 开头的参数
        draft_kwargs = {}
        target_kwargs = {}
        for k, v in kwargs.items():
            if k.startswith("draft_"):
                draft_kwargs[k[6:]] = v  # 去掉 "draft_" 前缀
            elif k.startswith("target_"):
                target_kwargs[k[7:]] = v  # 去掉 "target_" 前缀
            else:
                pass
        self.draft_config = DraftConfig(**draft_kwargs)
        self.target_config = TargetConfig(**target_kwargs)
        self.num_speculative_tokens = num_speculative_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(target_model, use_fast=True)
        self.eos = self.tokenizer.eos_token_id
        self.draft_config.eos = self.tokenizer.eos_token_id
        self.target_config.eos = self.tokenizer.eos_token_id
        self.target_engine = TargetEngine(target_model, **asdict(self.target_config))
        self.draft_engine = DraftEngine(draft_model, num_turn_spec_tokens=4, **asdict(self.draft_config))
        
        
    def add_request(self, prompt, sampling_param):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        target_req = Request(prompt, sampling_param)
        draft_req = Request(prompt, sampling_param, target_req.request_id)
        self.target_engine.add_request(target_req) 
        self.draft_engine.add_request(draft_req)
    
    
    def verify(self, draft_req_logits, target_req_logits, draft_outputs, target_outputs):
        common_req_ids = set(draft_req_logits.keys()) & set(target_req_logits.keys())
        verbose = False
        outputs = []
        for req_id in common_req_ids:
            draft_logits = draft_req_logits[req_id]
            target_logits = target_req_logits[req_id]
            draft_token_ids = draft_outputs[req_id]
            target_token_ids = target_outputs[req_id]
            num_round_generate = len(draft_token_ids)
            num_unaccept_tokens = num_round_generate
            # print(f"req {req_id} draft logit shape {draft_logits.shape} target_logit shape {target_logits.shape} {draft_token_ids} {target_token_ids}")
            for i in range(num_round_generate):    
                r = torch.rand(1, device=draft_logits.device)
                draft_logit = draft_logits[i]
                target_logit = target_logits[i]
                draft_token_id = draft_token_ids[i]
                if r > (target_logit[draft_token_id] / draft_logit[draft_token_id]):
                    break
                
                if verbose:
                    print(f"req id{req_id}, r is {r}, accpeted token{draft_token_ids[i]}")
                
                if draft_token_id == self.eos:
                    break
                num_unaccept_tokens -= 1
            new_token_id = target_token_ids[num_round_generate - num_unaccept_tokens]
            self.draft_engine.verify_process(req_id, num_unaccept_tokens, new_token_id)
            req = self.target_engine.verify_process(req_id, num_unaccept_tokens, new_token_id)
            outputs.append(req)
            # print(f"req id {req_id} unaccept token {num_unaccept_tokens}")
        return outputs        
    
    def one_round(self):
        # print("========================draft=======================")
        draft_outputs, draft_req_logits_map = self.draft_engine.run()
        # print(draft_outputs)
        # print("========================target=======================")
        self.target_engine.integrate_draft_output(draft_outputs)
        target_outputs, target_req_logits_map = self.target_engine.run()
        # print(target_outputs)
        # print("========================verify=======================")
        outputs = self.verify(draft_req_logits_map, target_req_logits_map, draft_outputs, target_outputs)
        return outputs
    
    def is_finished(self):
        return self.target_engine.is_finished() and self.draft_engine.is_finished()
    
    def reset_block_manager(self):
        self.target_engine.reset_block_manager()
        self.draft_engine.reset_block_manager()
    
    
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        ):
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        while not self.is_finished():
            output = self.one_round()
            for req in output:
                outputs[req.request_id] = req.token_ids
        outputs = [outputs[req_id] for req_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs