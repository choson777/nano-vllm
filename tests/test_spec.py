import os
from nanovllm import SpeculativeEngine, LLM, SamplingParams
from transformers import AutoTokenizer
import torch
import time

def main():
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    draft_model_path = os.path.expanduser("/data3/lqc/models/qwen3-0.6b/")
    # /data3/szf_hf/huggingface/model/Qwen3-8B/
    # /data3/lqc/models/qwen3-0.6b/
    target_model_path = os.path.expanduser("/data3/szf_hf/huggingface/model/Qwen3-8B/")
    tokenizer = AutoTokenizer.from_pretrained(target_model_path)
    spec_llm = SpeculativeEngine(target_model_path, draft_model_path)
    print("✅ SpeculativeEngine initialized successfully!")
    sampling_params = SamplingParams(temperature=0.6, max_tokens=50)

    prompts = [
        "introduce yourself",
        "list all prime numbers less than 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    
    outputs = spec_llm.generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")

    
if __name__ == '__main__':
    main()
