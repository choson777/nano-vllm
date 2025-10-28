import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.scheduler.draft_scheduler import DraftScheduler
from nanovllm.engine.model_runner import ModelRunner


class DraftEngine:

    def __init__(self, model, num_turn_spec_tokens, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.model_runner = ModelRunner(config, 0, [])
        print(f"draft engine eos{config.eos}")
        self.scheduler = DraftScheduler(config, num_turn_spec_tokens)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner

    def add_request(self, seq):
        self.scheduler.add(seq)
    

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids, logits = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.increase_token_ids) for seq in seqs if (seq.is_suspend or seq.is_finished)]
        seq_id_to_logits = [(seq.seq_id, logit) for seq, logit in zip(seqs, logits)]
        return outputs, seq_id_to_logits

    def is_turn_finished(self):
        return self.scheduler.is_turn_finished()
    
    def run(self):
        self.scheduler.resume_from_suspend()
        seq_logits = {}
        outputs = {}
        while not self.is_turn_finished():
            output, logits = self.step()
            for seq_id, logit in logits:
                if seq_id not in seq_logits:
                    seq_logits[seq_id] = []
                seq_logits[seq_id].append(logit)
                
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
        return outputs, seq_logits
