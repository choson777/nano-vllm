from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.config import Config, DraftConfig, TargetConfig
from dataclasses import fields, asdict


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
        self.target_engine = LLMEngine(target_model, **asdict(self.target_config))
        self.draft_engine = LLMEngine(draft_model,  **asdict(self.draft_config))
        
        
    