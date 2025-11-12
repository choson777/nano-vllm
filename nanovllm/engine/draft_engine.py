import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
import torch

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.request import Request
from nanovllm.scheduler.spec_scheduler import DraftScheduler
from nanovllm.engine.model_runner import ModelRunner


class DraftEngine:

    def __init__(self, model, num_turn_spec_tokens, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.model_runner = ModelRunner(config, 0, [])
        self.scheduler = DraftScheduler(config, num_turn_spec_tokens)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner

    def add_request(self, req):
        self.scheduler.add(req)
    

    def step(self):
        reqs, is_prefill = self.scheduler.schedule()
        token_ids, logits = self.model_runner.call("run", reqs, is_prefill)
        self.scheduler.postprocess(reqs, token_ids)
        outputs = [(req.request_id, req.one_round_generated_token_ids) for req in reqs if (req.is_suspend or req.is_finished)]
        req_id_to_logits = [(req.request_id, logit) for req, logit in zip(reqs, logits)]
        return outputs, req_id_to_logits

    def is_round_finished(self):
        return self.scheduler.is_round_finished()
    
    def is_finished(self):
        return self.scheduler.is_finished()
    
    def run(self):
        self.scheduler.resume_from_suspend()
        req_logits = {}
        outputs = {}
        i = 0
        while not self.is_round_finished():
            output, logits = self.step()
            for req_id, logit in logits:
                if req_id not in req_logits:
                    req_logits[req_id] = []
                req_logits[req_id].append(logit)
                
            for req_id, token_ids in output:
                outputs[req_id] = token_ids
            i += 1
        for req_id in req_logits:
            req_logits[req_id] = torch.stack(req_logits[req_id])
        return outputs, req_logits

    
    def verify_process(self, req_id: int, unaccept_tokens: int, new_token_id: int):
        self.scheduler.verify_process(req_id, unaccept_tokens, new_token_id)
        
        
    def reset_block_manager(self):
        self.scheduler.reset_hash_map()