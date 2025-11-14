from collections import deque

from nanovllm.config import Config
from nanovllm.engine.request import Request, RequestStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_reqs = config.max_num_reqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Request] = deque()
        self.running: deque[Request] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, req: Request):
        self.waiting.append(req)

    def schedule(self) -> tuple[list[Request], bool]:
        # prefill
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

        # decode
        while self.running and num_reqs < self.max_num_reqs:
            req = self.running.popleft()
            while not self.block_manager.can_append(req):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(req)
                    break
            else:
                num_reqs += 1
                self.block_manager.may_append(req)
                scheduled_reqs.append(req)
        assert scheduled_reqs
        self.running.extendleft(reversed(scheduled_reqs))
        return scheduled_reqs, False

    def preempt(self, req: Request):
        req.status = RequestStatus.WAITING
        self.block_manager.deallocate(req)
        self.waiting.appendleft(req)

    def postprocess(self, reqs: list[Request], token_ids: list[int]) -> list[bool]:
        for req, token_id in zip(reqs, token_ids):
            req.set_consume_tokens()
            req.append_token(token_id)
            if (not req.ignore_eos and token_id == self.eos) or req.num_completion_tokens == req.max_tokens:
                req.status = RequestStatus.FINISHED
                self.block_manager.deallocate(req)
                self.running.remove(req)
                
    def abort(self, req_id: int):
        for req in self.running:
            if req_id == req.request_id:
                req.status = RequestStatus.FINISHED
                self.block_manager.deallocate(req)
                self.running.remove(req)
                return 
        for req in self.waiting:
            if req_id == req.request_id:
                req.status = RequestStatus.FINISHED
                self.waiting.remove(req)