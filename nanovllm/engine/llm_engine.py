import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner

class StepOutput:
    def __init__(self, seq: Sequence, new_token: str, new_token_id: int):
        self.seq_id = seq.seq_id
        self.new_token = new_token
        self.new_token_id = new_token_id
        self.is_finished = seq.is_finished
    
    def __repr__(self) -> str:
        return (
            f"StepOutput(request_id={self.seq_id}, "
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

    def add_request(self, request_id: int, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(request_id, prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        print(f"这一轮调用的request的数量是{len(seqs)}, waiting队列长度为{self.scheduler.waiting}, running 队列长度为{self.scheduler.running}")
        if not seqs:
            print("No sequences scheduled, engine waiting for new requests or cache availability.")
            return []        
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [StepOutput(seq, self.tokenizer.decode(seq.last_token), seq.last_token) for seq in seqs]
        return outputs

    def is_finished(self):
        return self.scheduler.is_finished()
