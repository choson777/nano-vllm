from nanovllm.engine.llm_engine import LLMEngine


class AsyncLLM:
    """A Large Language Model (LLM) for online inference."""

    def __init__(
        self,
        model: str,
        enforce_eager: bool = True,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        **kwargs,
    ):
        self.engine = LLMEngine(
            model, 
            enforce_eager=enforce_eager, 
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            **kwargs)
        self.request_events = {}
        