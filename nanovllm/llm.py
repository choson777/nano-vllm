from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams

from typing import Optional
import asyncio


class LLM(LLMEngine):
    pass

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
        self.kicking_request_id: Optional[str] = None
        self.timeout_interval = 1
        
    async def engine_step(self, kicking_request_id: Optional[str] = None):
        self.is_engine_running = True
        self.kicking_request_id = kicking_request_id
        await asyncio.sleep(0)
        step_outputs = self.engine.step()
        self.is_engine_running = False
        self.kicking_request_id = None
        for step_output in step_outputs:
            request_id = step_output.request_id
            if request_id in self.request_events:
                self.request_events[request_id].set()
                self.step_outputs[request_id] = step_output
    
    async def generate(
        self,
        request_id: str,
        prompt: Optional[str] = None,
        sampling_params: SamplingParams = SamplingParams(),
    ):
        request_event = asyncio.Event()
        self.request_events[request_id] = request_event
        self.engine.add_request(prompt, sampling_params, request_id)
        outputs = ""
        while True:
            if request_id not in self.request_events:
                return
            
            if not self.is_engine_running:
                try:
                    await self.engine_step()
                except RuntimeError as e:
                    await self.abort(request_id)
                    raise e
            
            try:
                await asyncio.wait_for(request_event.wait(), timeout=self.timeout_interval)
            except asyncio.TimeoutError:
                continue
            request_event.clear()
            step_output = self.step_outputs[request_id]
            outputs += step_output.new_token
            print(outputs)
            yield outputs
            
            if step_output.is_finished:
                del self.request_events[request_id]
                del self.step_outputs[request_id]
                
                if not self.is_engine_running:
                    await self.engine_step()
                break
    
    async def abort(self, request_id: str):
        if request_id not in self.request_events:
            # The request has already finished or been aborted.
            return

        self.engine.abort_request(request_id)

        if request_id in self.request_events:
            del self.request_events[request_id]
        if request_id in self.step_outputs:
            del self.step_outputs[request_id]

        # To prevent deadlock when a request is aborted while the engine is
        # running.
        if self.kicking_request_id == request_id:
            self.is_engine_running = False
            self.kicking_request_id = None