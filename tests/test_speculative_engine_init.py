import os
from nanovllm import SpeculativeEngine, LLM

def main():
    draft_model_path = os.path.expanduser("/data3/lqc/models/qwen3-0.6b/")
    target_model_path = os.path.expanduser("/data3/szf/huggingface/model/Qwen3-8B/")
    spec_llm = SpeculativeEngine(draft_model_path, target_model_path)
    print("✅ SpeculativeEngine initialized successfully!")

if __name__ == '__main__':
    main()
