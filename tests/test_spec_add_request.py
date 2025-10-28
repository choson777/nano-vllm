import os
from nanovllm import SpeculativeEngine, LLM, SamplingParams
from transformers import AutoTokenizer

def main():
    draft_model_path = os.path.expanduser("/data3/lqc/models/qwen3-0.6b/")
    target_model_path = os.path.expanduser("/data3/szf/huggingface/model/Qwen3-8B/")
    tokenizer = AutoTokenizer.from_pretrained(target_model_path)
    spec_llm = SpeculativeEngine(draft_model_path, target_model_path)
    print("✅ SpeculativeEngine initialized successfully!")
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
    
    spec_llm.add_request(prompts, sampling_params)
    spec_llm.one_turn()
    
    
    for seq in spec_llm.target_engine.scheduler.suspend:
        print(seq.seq_id, seq.token_ids, seq.block_table)
    for seq in spec_llm.draft_engine.scheduler.suspend:
        print(seq.seq_id, seq.token_ids, seq.block_table)
    

if __name__ == '__main__':
    main()
