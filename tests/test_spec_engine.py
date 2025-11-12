import os
from nanovllm import SpeculativeEngine, LLM, SamplingParams
from transformers import AutoTokenizer
import torch
import time

def main():
    draft_model_path = os.path.expanduser("/data3/lqc/models/qwen3-0.6b/")
    # /data3/szf_hf/huggingface/model/Qwen3-8B/
    # /data3/lqc/models/qwen3-0.6b/
    target_model_path = os.path.expanduser("/data3/szf_hf/huggingface/model/Qwen3-8B/")
    tokenizer = AutoTokenizer.from_pretrained(target_model_path)
    spec_llm = SpeculativeEngine(draft_model_path, target_model_path)
    print("✅ SpeculativeEngine initialized successfully!")
    sampling_params = SamplingParams(temperature=0.6, max_tokens=512)

    prompts = [
        # "introduce yourself",
        # "list all prime numbers less than 100",
        """Consider the following scenario: You are an advisor to the CEO of a large, established technology company (let's call it 'TechCorp') that has been dominant in the consumer software market for over two decades. TechCorp is known for its productivity suites, operating systems, and cloud services. However, the landscape is rapidly shifting with the rise of Artificial Intelligence, particularly large language models (LLMs) and generative AI, as well as increasing competition from agile startups and tech giants investing heavily in these new fields.

            The CEO has come to you seeking a comprehensive strategic analysis and a multi-year roadmap. Your task is to provide a detailed, multi-faceted strategic plan addressing the following points:

            Current State Analysis: Analyze TechCorp's current position. What are its core strengths (e.g., user base, brand, resources, existing tech stack) and potential vulnerabilities in the face of the AI revolution? How does its current product portfolio align or conflict with the trends in AI and generative technologies?
            Competitive Landscape: Identify the key players. This includes direct competitors developing AI features, potential disruptors, open-source initiatives, and the major tech giants (Google, Microsoft, Meta, Amazon). Analyze their strategies, strengths, and weaknesses relative to TechCorp.
            Strategic Options: Propose 2-3 distinct, high-level strategic options for TechCorp. For example:
            Option A: Aggressive internal development of proprietary, state-of-the-art foundational AI models and integration across all products.
            Option B: Strategic partnerships and licensing of advanced AI models from leading AI labs (e.g., OpenAI, Anthropic) and focusing on superior application and user experience.
            Option C: Acquisition spree targeting promising AI startups to rapidly gain talent, technology, and market share.
            Option D: A hybrid approach focusing on specific niche applications of AI within its existing product strengths (e.g., AI for productivity, AI for security).
            For each option, detail the potential benefits, required investments (financial, talent, time), and significant risks.
            Recommended Path: Based on your analysis, recommend one primary strategic path (or a hybrid approach) and justify your choice. Explain why it is the best fit for TechCorp's profile and the current market dynamics.
            Implementation Plan: Outline a high-level, phased implementation plan for your recommended strategy over the next 3-5 years. Include key milestones, potential internal challenges (e.g., culture shift, restructuring), and external challenges (e.g., regulatory scrutiny on AI, data privacy).
            Risk Mitigation: Identify the critical risks associated with your recommended path (e.g., technical failure, talent loss, ethical AI issues, market rejection) and propose mitigation strategies.
            Success Metrics: Define clear, measurable KPIs to track the success of the strategy (e.g., AI feature adoption rates, revenue from new AI products, market share in AI-enabled services, developer engagement with AI APIs).
            Please structure your response clearly, addressing each point sequentially, providing a thorough and well-reasoned analysis. Consider the long-term implications for TechCorp's market position, profitability, and relevance in a world increasingly driven by artificial intelligence."""
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    # warm up
    with torch.no_grad():
        for _ in range(5):
            _ = spec_llm.generate(prompts, SamplingParams(temperature=0.6, max_tokens=4))
    total_time = 0
    total_tokens = 0    
    with torch.no_grad():
        for i in range(1):
            print(i)
            start_time = time.time()
            outputs = spec_llm.generate(prompts, sampling_params)
            end_time = time.time()
            total_time += end_time - start_time
            total_tokens += len(outputs[0]['token_ids'])
    
    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


    print(f"推理速度为{(total_tokens) / total_time}")
if __name__ == '__main__':
    main()
