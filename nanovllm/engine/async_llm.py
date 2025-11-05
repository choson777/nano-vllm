from nanovllm.engine.llm_engine import LLMEngine
from nanovllm import SamplingParams

from typing import Optional
import asyncio

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
        self.step_outputs = {}
        self.is_engine_running = False
        
    async def engine_step(self):
        self.is_engine_running = True
        step_outputs = self.engine.step()
        self.is_engine_running = False
        for step_output in step_outputs:
            seq_id = step_output.seq_id
            self.request_events[seq_id].set()
            self.step_outputs[seq_id] = step_output
    
    async def generate(
        self,
        request_id: str,
        prompt: Optional[str] = None,
        sampling_params: SamplingParams = SamplingParams(),
    ):
        
        request_events = asyncio.Event()
        self.request_events[request_id] = request_events
        
        self.engine.add_request(request_id, prompt, sampling_params)
        
        if not self.is_engine_running:
            await self.engine_step()
        
        return self.step_outputs[request_id]