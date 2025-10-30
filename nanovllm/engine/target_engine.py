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
        self.scheduler = TargetScheduler(config)
        atexit.register(self.exit)
        
        
    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()
            
    def add_request(self, seq: Sequence):
        self.scheduler.add(seq)
        
    def integrate_draft_output(self, outputs):
        self.scheduler.add_token_ids(outputs)

    def is_round_finished(self):
        return self.scheduler.is_round_finished()

    
    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids, logits = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs)
        seq_logits = {}
        seq_outputs = {}
        start_index = 0
        for seq in seqs:
            end_index = start_index + seq.num_verify_tokens
            seq_logits[seq.seq_id] = logits[start_index: end_index]
            seq_outputs[seq.seq_id] = token_ids[start_index: end_index]
            start_index = end_index
        return seq_outputs, seq_logits
    
    
    def run(self):
        self.scheduler.resume_from_suspend()
        seq_logits = {}
        seq_outputs = {}
        while not self.is_round_finished():
            seq_output, seq_logit = self.step()
            for seq_id in seq_logit:
                seq_logits[seq_id] = seq_logit[seq_id]
            for seq_id in seq_output:
                seq_outputs[seq_id] = seq_output[seq_id]
        return seq_outputs, seq_logits
    
    def verify_process(self, seq_id, num_unaccpet_tokens, new_token_id):
        self.scheduler.verify_process(seq_id, num_unaccpet_tokens, new_token_id)