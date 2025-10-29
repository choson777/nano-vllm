import os
from nanovllm import SpeculativeEngine, LLM, DraftEngine, DraftConfig, SamplingParams, Sequence
from dataclasses import asdict
from transformers import AutoTokenizer
import torch

def main():
    draft_model_path = os.path.expanduser("/data3/lqc/models/qwen3-0.6b/")
    tokenizer = AutoTokenizer.from_pretrained(draft_model_path)
    draft_config = DraftConfig()
    draft_engine = DraftEngine(draft_model_path, num_turn_spec_tokens=4, **asdict(draft_config))

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
    seq_id_map = {}
    for prompt, sp in zip(prompts, sampling_params):
        if isinstance(prompt, str):
            prompt = tokenizer.encode(prompt)
        target_seq = Sequence(prompt, sp)
        draft_seq = Sequence(prompt, sp)
        seq_id_map[target_seq.seq_id] = draft_seq.seq_id
        draft_engine.add_request(draft_seq)

        
    outputs = {}
    for i in range(10):
        print(i)
        output, seq_logits = draft_engine.run()
        print(output)
        for seq_id in output:
            if seq_id not in outputs:
                outputs[seq_id] = []
            outputs[seq_id].extend(output[seq_id])
    
    for seq_id in outputs:
        print(tokenizer.decode(outputs[seq_id]))
    
if __name__ == '__main__':
    main()