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
        print("进来了 有这个进程")
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.model_runner = ModelRunner(config, 0, [])
        self.scheduler = DraftScheduler(config, num_turn_spec_tokens)
        atexit.register(self.exit)
        print(f"初始化了draft model")

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
        req_id_to_selected_logit = []
        for req, logit_tensor, selected_token_id in zip(reqs, logits, token_ids):
            selected_logit_val = logit_tensor[selected_token_id].item()  # float
            req_id_to_selected_logit.append((req.request_id, selected_logit_val))

        return outputs, req_id_to_selected_logit

    def is_round_finished(self):
        return self.scheduler.is_round_finished()
    
    def is_finished(self):
        return self.scheduler.is_finished()
    
    def run(self):
        self.scheduler.resume_from_suspend()
        req_logits = {}
        outputs = {}
        while not self.is_round_finished():
            output, logits = self.step()
            for req_id, logit in logits:
                if req_id not in req_logits:
                    req_logits[req_id] = []
                req_logits[req_id].append(logit)
                
            for req_id, token_ids in output:
                outputs[req_id] = token_ids
        return outputs, req_logits

    
    def verify_process(self, req_id: int, unaccept_tokens: int, new_token_id: int):
        self.scheduler.verify_process(req_id, unaccept_tokens, new_token_id)
        
        
    def reset_block_manager(self):
        self.scheduler.reset_hash_map()