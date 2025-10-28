import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.scheduler.target_scheduler import TargetScheduler
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
        print(f"target engine eos{config.eos}")
        self.scheduler = TargetScheduler(config)
        atexit.register(self.exit)
        
        
    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()
            
    def add_request(self, seq: Sequence):
        self.scheduler.add(seq)
        
    def integrate_draft_output(self, outputs, seq_id_map):
        self.scheduler.add_token_ids(outputs, seq_id_map)
                
            