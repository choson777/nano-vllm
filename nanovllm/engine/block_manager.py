from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.request import Request


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, request: Request) -> bool:
        return len(self.free_block_ids) >= request.num_blocks

    def allocate(self, request: Request):
        assert not request.block_table
        h = -1
        cache_miss = False
        for i in range(request.num_blocks):
            token_ids = request.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                request.num_cached_tokens += self.block_size
                request.num_consume_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            request.block_table.append(block_id)

    def deallocate(self, request: Request):
        for block_id in reversed(request.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        request.num_cached_tokens = 0
        request.num_consume_tokens = 0
        request.block_table.clear()

    def can_append(self, request: Request) -> bool:
        return len(self.free_block_ids) >= (len(request) % self.block_size == 1)
    
    def can_append_mul_tokens(self, request: Request) -> bool:
        return len(self.free_block_ids) >= (len(request) % self.block_size > 0 and len(request) % self.block_size <= (request.num_tokens - request.num_consume_tokens))

    def may_append(self, request: Request):
        try:
            block_table = request.block_table
            last_block = self.blocks[block_table[-1]]
            if len(request) % self.block_size == 1:
                assert last_block.hash != -1
                block_id = self.free_block_ids[0]
                self._allocate_block(block_id)
                block_table.append(block_id)
            elif len(request) % self.block_size == 0:
                assert last_block.hash == -1
                token_ids = request.block(request.num_blocks-1)
                prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
                h = self.compute_hash(token_ids, prefix)
                last_block.update(h, token_ids)
                self.hash_to_block_id[h] = last_block.block_id
            else:
                assert last_block.hash == -1
        except:
            print(f"req id {request.request_id} num tokens {request.num_tokens} block table {request.block_table} last_block hash {self.blocks[request.block_table[-1]].hash}")
        
    def may_append_mul_tokens(self, request: Request):
        block_table = request.block_table
        last_block = self.blocks[block_table[-1]]
        num_remain_tokens = len(request) % self.block_size
        num_new_tokens = request.num_tokens - request.num_consume_tokens
        try:
            if num_remain_tokens == 0:
                assert last_block.hash == -1
                token_ids = request.block(request.num_blocks-1)
                prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
                h = self.compute_hash(token_ids, prefix)
                last_block.update(h, token_ids)
                self.hash_to_block_id[h] = last_block.block_id
            elif num_remain_tokens == num_new_tokens:
                assert last_block.hash != -1
                block_id = self.free_block_ids[0]
                self._allocate_block(block_id)
                block_table.append(block_id)
            elif num_remain_tokens < num_new_tokens:
                assert last_block.hash == -1
                token_ids = request.block(request.num_blocks-1)
                prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
                h = self.compute_hash(token_ids, prefix)
                last_block.update(h, token_ids)
                self.hash_to_block_id[h] = last_block.block_id
                block_id = self.free_block_ids[0]
                self._allocate_block(block_id)
                block_table.append(block_id)
            else:
                assert last_block.hash == -1
        except:    
            print(f"req id {request.request_id} num tokens {request.num_tokens} block table {request.block_table} last_block hash {self.blocks[request.block_table[-1]].hash} 还没分配的token数量{request.num_verify_tokens} 最后一个块剩余的数量{num_remain_tokens}")

    
    def reclaim_tokens(self, request: Request, num_tokens: int):
        print("重退block")
        block_table = request.block_table
        last_block = self.blocks[block_table[-1]]
        last_block_num_tokens = request.last_block_num_tokens
        print(f"req id {request.request_id} 一共{request.num_tokens}token, block table {request.block_table} 回退{num_tokens}个token, 最后一个块的block数量是 {last_block_num_tokens}")
        if num_tokens >= last_block_num_tokens:
            last_block.ref_count -= 1
            if last_block.ref_count == 0:
                self._deallocate_block(block_table[-1])
                request.block_table.pop()
            if num_tokens > last_block_num_tokens:
                self.blocks[request.block_table[-1]].hash = -1      
        else:
            self.blocks[request.block_table[-1]].hash = -1
        print(f"block table 变成{request.block_table}, 最后一个块的哈希{self.blocks[request.block_table[-1]].hash}")