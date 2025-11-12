from collections import deque
from typing import Optional

from nanovllm.config import Config
from nanovllm.engine.request import Request, RequestStatus
from nanovllm.engine.block_manager import BlockManager


class TargetScheduler:

    def __init__(self, config: Config):
        self.max_num_reqs = config.max_num_reqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Request] = deque()
        self.running: deque[Request] = deque()
        self.suspend: deque[Request] = deque()
        self.request_map = {}
    
    def is_finished(self):
        return not self.waiting and not self.running and not self.suspend
    
    def is_round_finished(self):
        return not self.waiting and not self.running
    
    def add(self, request: Request):
        self.waiting.append(request)
        self.request_map[request.request_id] = request
        
    def add_token_ids(self, token_ids_map):
        for req in self.waiting:
            if req.request_id in token_ids_map:
                req.extend_token(token_ids_map[req.request_id])
                
        for req in self.suspend:
            if req.request_id in token_ids_map:
                req.extend_token(token_ids_map[req.request_id])

    def resume_from_suspend(self):
        while self.suspend:
            req = self.suspend.popleft()
            req.status = RequestStatus.RUNNING
            self.running.append(req)
    
    def schedule(self) -> tuple[list[Request], bool]:

        scheduled_reqs = []
        num_reqs = 0
        num_batched_tokens = 0
        while self.waiting and num_reqs < self.max_num_reqs:
            req = self.waiting[0]
            if num_batched_tokens + len(req) > self.max_num_batched_tokens or not self.block_manager.can_allocate(req):
                break
            num_reqs += 1
            self.block_manager.allocate(req)
            num_batched_tokens += len(req) - req.num_cached_tokens
            req.status = RequestStatus.RUNNING
            self.waiting.popleft()
            self.running.append(req)
            scheduled_reqs.append(req)
        if scheduled_reqs:
            return scheduled_reqs, True
    
        while self.running and num_reqs < self.max_num_reqs and num_batched_tokens < self.max_num_batched_tokens:
            req = self.running.popleft()
            while not self.block_manager.can_append_mul_tokens(req):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(req)
                    break
            else:
                num_reqs += 1
                num_batched_tokens += req.num_tokens - req.num_consume_tokens
                self.block_manager.may_append_mul_tokens(req)
                scheduled_reqs.append(req)
        assert scheduled_reqs
        self.running.extendleft(reversed(scheduled_reqs))
        return scheduled_reqs, False    
    
    def preempt(self, req: Request):
        req.status = RequestStatus.WAITING
        self.block_manager.deallocate(req)
        self.waiting.appendleft(req)
        
    def postprocess(self, reqs: list[Request]):
        for req in reqs:
            req.status = RequestStatus.SUSPEND
            self.running.remove(req)
            self.suspend.append(req)
            
            
    def verify_process(self, req_id, num_unaccept_tokens, new_token_id):
        req = self.request_map[req_id]
        if num_unaccept_tokens > 0:
            self.block_manager.reclaim_tokens(req, num_unaccept_tokens)
            req.delete_tokens(num_unaccept_tokens)
        req.set_consume_tokens()
        req.append_token(new_token_id)
        if new_token_id == self.eos or req.num_completion_tokens >= req.max_tokens:
            req.status = RequestStatus.FINISHED
            self.suspend.remove(req)
            self.block_manager.deallocate(req)
            self.request_map.pop(req_id)
        return req