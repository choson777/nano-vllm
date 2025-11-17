import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.request import Request
from nanovllm.scheduler.spec_scheduler import TargetScheduler
from nanovllm.engine.model_runner import ModelRunner


class TargetEngine:

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
        self.scheduler = TargetScheduler(config)
        atexit.register(self.exit)
        
        
    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()
            
    def add_request(self, req: Request):
        self.scheduler.add(req)
        
    def integrate_draft_output(self, outputs):
        self.scheduler.add_token_ids(outputs)

    def is_round_finished(self):
        return self.scheduler.is_round_finished()

    def is_finished(self):
        return self.scheduler.is_finished()
    
    def step(self):
        reqs, is_prefill = self.scheduler.schedule()
        print(f"这轮调用的reqs是:{reqs} {is_prefill}")
        token_ids, logits = self.model_runner.call("run", reqs, is_prefill)
        req_logits = {}
        req_outputs = {}
        start_index = 0
        for req in reqs:
            if is_prefill:
                end_index = start_index + req.num_tokens - (req.num_prompt_tokens - 1)
            else:
                end_index = start_index + req.num_tokens - req.num_consume_tokens
            req_logits[req.request_id] = logits[start_index: end_index]
            req_outputs[req.request_id] = token_ids[start_index: end_index]
            start_index = end_index
        self.scheduler.postprocess(reqs)
        return req_outputs, req_logits
    
    
    def run(self):
        self.scheduler.resume_from_suspend()
        req_logits = {}
        req_outputs = {}
        while not self.is_round_finished():
            req_output, req_logit = self.step()
            for req_id in req_logit:
                req_logits[req_id] = req_logit[req_id]
            for req_id in req_output:
                req_outputs[req_id] = req_output[req_id]
        return req_outputs, req_logits
    
    def verify_process(self, req_id, num_unaccpet_tokens, new_token_id):
        return self.scheduler.verify_process(req_id, num_unaccpet_tokens, new_token_id)
    
    def reset_block_manager(self):
        self.scheduler.reset_hash_map()
        
    def abort(self, request_id:int):
        self.scheduler.abort(request_id)