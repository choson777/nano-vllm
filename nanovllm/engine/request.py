from copy import copy
from enum import Enum, auto
from itertools import count
from typing import Optional

from nanovllm.sampling_params import SamplingParams


class RequestStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
    SUSPEND = auto()


class Request:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams(), request_id: Optional[int] = None):
        self.request_id = request_id if request_id else next(Request.counter)
        self.status = RequestStatus.WAITING
        
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_checked_tokens = self.num_prompt_tokens - 1
        self.num_consume_tokens = 0
        self.one_round_generated_tokens = 0

        
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == RequestStatus.FINISHED
    
    @property
    def is_suspend(self):
        return self.status == RequestStatus.SUSPEND
    
    @property
    def num_unchecked_tokens(self):
        return self.num_tokens - self.num_checked_tokens
    
    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens    
    
    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def one_round_generated_token_ids(self):
        return self.token_ids[-self.one_round_generated_tokens:]
    
    @property
    def unchecked_token_ids(self):
        return self.token_ids[self.num_checked_tokens:]
    
    @property
    def unconsume_token_ids(self):
        return self.token_ids[self.num_consume_tokens:]
    
    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size        
    
    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
    
    def extend_token(self, token_ids: list[int]):
        self.token_ids.extend(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens += len(token_ids)
        
    def delete_tokens(self, num_remove_tokens: int):
        del self.token_ids[-num_remove_tokens:]
        self.num_tokens -= num_remove_tokens
        self.last_token = self.token_ids[-1]

    def __getstate__(self):
        return (self.num_tokens, self.last_token, self.num_prompt_tokens, self.num_cached_tokens, self.num_checked_tokens, self.num_consume_tokens, self.block_table, self.token_ids)

    def __setstate__(self, state):
        self.num_tokens, self.last_token, self.num_prompt_tokens, self.num_cached_tokens, self.num_checked_tokens, self.num_consume_tokens, self.block_table, self.token_ids= state


    def reset_for_new_round(self):
        self.one_round_generated_tokens
        
    def set_checked_tokens(self):
        self.num_checked_tokens = self.num_tokens
        
    def set_consume_tokens(self):
        self.num_consume_tokens = self.num_tokens
        