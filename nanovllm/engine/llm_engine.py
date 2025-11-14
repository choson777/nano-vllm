import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
from typing import Optional

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.request import Request
from nanovllm.scheduler.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class StepOutput:
    def __init__(self, req: Request, new_token: str, new_token_id: int):
        self.request_id = req.request_id
        self.request = req
        self.new_token = new_token
        self.new_token_id = new_token_id
        self.is_finished = req.is_finished
    
    def __repr__(self) -> str:
        return (
            f"StepOutput(request_id={self.request_id}, "
            f"request={self.request},"
            f"new_token={self.new_token}, "
            f"new_token_id={self.new_token_id}, "
            f"is_finished={self.is_finished})"
        )

class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams, request_id: Optional[int] = None ):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        req = Request(prompt, sampling_params, request_id)
        self.scheduler.add(req)

    def step(self):
        reqs, is_prefill = self.scheduler.schedule()
        print(f"这一轮调用的request的数量是{len(reqs)}, waiting队列长度为{self.scheduler.waiting}, running 队列长度为{self.scheduler.running}")
        if not reqs:
            print("No sequences scheduled, engine waiting for new requests or cache availability.")
            return []        
        token_ids, _ = self.model_runner.call("run", reqs, is_prefill)
        self.scheduler.postprocess(reqs, token_ids)
        outputs = [StepOutput(req, self.tokenizer.decode(req.last_token), req.last_token) for req in reqs]
        return outputs

    def is_finished(self):
        return self.scheduler.is_finished()

    def abort_request(self, request_id: int):
        self.scheduler.abort(request_id)
    
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for req_id, token_ids in output:
                outputs[req_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[req_id] for req_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs