import os
from nanovllm import SpeculativeEngine, LLM, SamplingParams
from transformers import AutoTokenizer

def main():
    draft_model_path = os.path.expanduser("/data3/lqc/models/qwen3-0.6b/")
    # /data3/szf_hf/huggingface/model/Qwen3-8B/
    target_model_path = os.path.expanduser("/data3/lqc/models/qwen3-0.6b/")
    tokenizer = AutoTokenizer.from_pretrained(target_model_path)
    spec_llm = SpeculativeEngine(draft_model_path, target_model_path)
    print("✅ SpeculativeEngine initialized successfully!")
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

    prompts = [
    """
    In the heart of a vast digital landscape, a new frontier of artificial intelligence emerges, reshaping the very fabric of human interaction with technology. This frontier, often referred to as the era of large language models (LLMs), is characterized by systems of unprecedented scale and capability. These models, trained on immense corpora of text, have demonstrated an uncanny ability to understand, generate, and reason about human language in ways that were once thought to be exclusively human domains. They can compose poetry, answer complex questions, translate languages, and even generate code, blurring the lines between human creativity and machine capability. The implications of this technological leap are profound, influencing fields as diverse as education, science, art, and business. However, this power comes with significant challenges and responsibilities. Issues of bias, misinformation, privacy, and the potential displacement of human roles are at the forefront of discussions surrounding their deployment. As we stand on the precipice of this new age, the balance between harnessing the immense potential of LLMs and mitigating their risks becomes a critical task for technologists, ethicists, policymakers, and society at large. 
    """,
    """
    In the heart of a vast digital landscape, a new frontier of artificial intelligence emerges, reshaping the very fabric of human interaction with technology. This frontier, often referred to as the era of large language models (LLMs), is characterized by systems of unprecedented scale and capability. These models, trained on immense corpora of text, have demonstrated an uncanny ability to understand, generate, and reason about human language in ways that were once thought to be exclusively human domains. They can compose poetry, answer complex questions, translate languages, and even generate code, blurring the lines between human creativity and machine capability.
    The implications of this technological leap are profound, influencing fields as diverse as education, science, art, and business. In education, LLMs can personalize learning experiences, providing instant feedback and tutoring tailored to individual needs. In science, they assist in analyzing vast datasets, identifying patterns, and generating hypotheses that might take human researchers significantly longer to formulate. In art, they open new avenues for creative expression, enabling collaboration between human artists and AI, or even generating entirely new forms of art. In business, they streamline operations, enhance customer service through sophisticated chatbots, and drive innovation by processing and analyzing market trends and customer feedback at scale.
    However, this power comes with significant challenges and responsibilities. Issues of bias, inherent in the training data, must be carefully addressed to ensure fairness. The potential for generating misinformation poses a serious threat to information integrity. Privacy concerns arise as models process and potentially store sensitive data. The potential displacement of human roles across various sectors is a major societal concern. These issues are at the forefront of discussions surrounding the deployment and regulation of LLMs.
    As we stand on the precipice of this new age, the balance between harnessing the immense potential of LLMs and mitigating their risks becomes a critical task for technologists, ethicists, policymakers, and society at large. The future will likely be shaped by how effectively we navigate this complex interplay between human intelligence and its artificial counterpart. It is crucial to ensure that these powerful tools augment human capabilities rather than diminish them, and contribute positively to the collective well-being of humanity. Transparency, accountability, and robust ethical frameworks are paramount. The journey into this digital frontier is just beginning, and its destination remains to be written by the choices we make today, focusing on responsible development and ethical use.
    """
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
    spec_llm.one_round()
    
    
    for seq in spec_llm.target_engine.scheduler.suspend:
        print(seq.seq_id, seq.token_ids, seq.block_table, seq.num_tokens, seq.num_checked_logit_generated_tokens, seq.num_prompt_tokens)
    for seq in spec_llm.draft_engine.scheduler.suspend:
        print(seq.seq_id, seq.token_ids, seq.block_table, seq.num_tokens, seq.num_checked_logit_generated_tokens, seq.num_prompt_tokens)


if __name__ == '__main__':
    main()
