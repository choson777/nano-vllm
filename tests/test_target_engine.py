import os
from nanovllm import SpeculativeEngine, LLM, DraftEngine, Sequence, SamplingParams, TargetEngine, TargetConfig
from dataclasses import asdict
from transformers import AutoTokenizer
import torch


import torch._dynamo
torch._dynamo.config.suppress_errors = True

def main():
    target_model_path = os.path.expanduser("/data3/lqc/models/qwen3-0.6b/")
    tokenizer = AutoTokenizer.from_pretrained(target_model_path)
    target_config = TargetConfig()
    target_engine = TargetEngine(target_model_path, **asdict(target_config))
    
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
        "write a short poem about the ocean",
        "explain the concept of machine learning in simple terms"
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    sampling_params = sampling_params = [sampling_params] * len(prompts)
    
    for prompt, sp in zip(prompts, sampling_params):
        if isinstance(prompt, str):
            prompt = tokenizer.encode(prompt)
        target_seq = Sequence(prompt, sp)
        draft_seq = Sequence(prompt, sp)
        target_engine.add_request(target_seq)
    
    outputs = {
        5: [151667, 198, 32313, 11],
        7: [151667, 198, 32313, 11],
        9: [151667, 198, 32313, 11],
        11: [151667, 198, 32313, 11]
    }
    
    
    
    target_engine.run()
    
if __name__ == '__main__':
    main()