from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.draft_engine import DraftEngine
from nanovllm.engine.target_engine import TargetEngine
from nanovllm.config import Config, DraftConfig, TargetConfig
from dataclasses import fields, asdict

from transformers import AutoTokenizer
from nanovllm.engine.sequence import Sequence
import copy

class SpeculativeEngine:
    
    def __init__(
        self,
        draft_model: str,
        target_model: str,
        num_speculative_tokens: int = 4,
        **kwargs
    ):
        
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
        print(self.tokenizer.eos_token_id)
        self.draft_config.eos = self.tokenizer.eos_token_id
        self.target_config.eos = self.tokenizer.eos_token_id
        self.target_engine = TargetEngine(target_model, **asdict(self.target_config))
        self.draft_engine = DraftEngine(draft_model, num_turn_spec_tokens=4, **asdict(self.draft_config))
        self.seq_id_map = {}
        
    def add_request(self, prompts, sampling_params):
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            if isinstance(prompt, str):
                prompt = self.tokenizer.encode(prompt)
            target_seq = Sequence(prompt, sp)
            draft_seq = Sequence(prompt, sp)
            self.target_engine.add_request(target_seq)
            self.seq_id_map[target_seq.seq_id] = draft_seq.seq_id 
            self.draft_engine.add_request(draft_seq)
    
    
    def one_turn(self):
        outputs, seq_logits = self.draft_engine.run()
        self.target_engine.integrate_draft_output(outputs, self.seq_id_map)
        outputs, seq_logits = self.target_engine.run()