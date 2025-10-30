from collections import deque
from typing import Optional

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class TargetScheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.suspend: deque[Sequence] = deque()
        self.seq_id_map = {}
    
    def is_finished(self):
        return not self.waiting and not self.running and not self.suspend
    
    def is_round_finished(self):
        return not self.waiting and not self.running
    
    def add(self, seq: Sequence):
        self.waiting.append(seq)
        self.seq_id_map[seq.seq_id] = seq
        
    def add_token_ids(self, token_ids_map):
        for seq in self.waiting:
            if seq.seq_id in token_ids_map:
                seq.reset_for_new_round()
                seq.extend_token(token_ids_map[seq.seq_id])
                
        for seq in self.suspend:
            if seq.seq_id in token_ids_map:
                seq.reset_for_new_round()
                seq.extend_token(token_ids_map[seq.seq_id])

    def resume_from_suspend(self):
        while self.suspend:
            seq = self.suspend.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
    
    def schedule(self) -> tuple[list[Sequence], bool]:

        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True
    
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append_mul_tokens(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append_mul_tokens(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False    
    
    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        
    def postprocess(self, seqs: list[Sequence]):
        for seq in seqs:
            seq.status = SequenceStatus.SUSPEND
            self.running.remove(seq)
            self.suspend.append(seq)
            
            
    def verify_process(self, seq_id, num_unaccept_tokens, new_token_id):
        seq = self.seq_id_map[seq_id]
        if num_unaccept_tokens > 0:
            self.block_manager.reclaim_tokens(seq, num_unaccept_tokens)
            seq.delete_tokens(num_unaccept_tokens)
        seq.reset_for_new_round()
        seq.append_token(new_token_id)