from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.draft_engine import DraftEngine
from nanovllm.engine.target_engine import TargetEngine
from nanovllm.config import Config, DraftConfig, TargetConfig
from dataclasses import fields, asdict
from collections import defaultdict

from transformers import AutoTokenizer
from nanovllm.engine.sequence import Sequence
import torch

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
        
        
    def add_request(self, prompts, sampling_params):
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            if isinstance(prompt, str):
                prompt = self.tokenizer.encode(prompt)
            target_seq = Sequence(prompt, sp)
            draft_seq = Sequence(prompt, sp, target_seq.seq_id)
            self.target_engine.add_request(target_seq) 
            self.draft_engine.add_request(draft_seq)
    
    
    def verify(self, draft_seq_logits, target_seq_logits, draft_outputs, target_outputs):
        common_seq_ids = set(draft_seq_logits.keys()) & set(target_seq_logits.keys())
        verbose = False
        outputs = []
        for seq_id in common_seq_ids:
            draft_logits = draft_seq_logits[seq_id]
            target_logits = target_seq_logits[seq_id]
            draft_token_ids = draft_outputs[seq_id]
            target_token_ids = target_outputs[seq_id]
            num_round_generate = len(draft_token_ids)
            num_unaccept_tokens = num_round_generate
            # print(f"seq {seq_id} draft logit shape {draft_logits.shape} target_logit shape {target_logits.shape} {draft_token_ids} {target_token_ids}")
            for i in range(num_round_generate):    
                r = torch.rand(1, device=draft_logits.device)
                draft_logit = draft_logits[i]
                target_logit = target_logits[i]
                draft_token_id = draft_token_ids[i]
                if r > (target_logit[draft_token_id] / draft_logit[draft_token_id]):
                    break
                
                if verbose:
                    print(f"seq id{seq_id}, r is {r}, accpeted token{draft_token_ids[i]}")
                
                if draft_token_id == self.eos:
                    break
                num_unaccept_tokens -= 1
            new_token_id = target_token_ids[num_round_generate - num_unaccept_tokens]
            self.draft_engine.verify_process(seq_id, num_unaccept_tokens, new_token_id)
            seq = self.target_engine.verify_process(seq_id, num_unaccept_tokens, new_token_id)
            outputs.append(seq)
            print(f"seq id {seq_id} unaccept token {num_unaccept_tokens}")
        return outputs        
    
    def one_round(self):
        print("========================draft=======================")
        draft_outputs, draft_seq_logits_map = self.draft_engine.run()
        print(draft_outputs)
        print("========================target=======================")
        self.target_engine.integrate_draft_output(draft_outputs)
        target_outputs, target_seq_logits_map = self.target_engine.run()
        print(target_outputs)
        print("========================verify=======================")
        outputs = self.verify(draft_seq_logits_map, target_seq_logits_map, draft_outputs, target_outputs)
        return outputs
    
    def is_finished(self):
        return self.target_engine.is_finished() and self.draft_engine.is_finished()
    
    
    def generate(self):
        outputs = {}
        while not self.is_finished():
            output = self.one_round()
            for seq in output:
                outputs[seq.seq_id] = seq.token_ids
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs