import torch.distributed as dist

def safe_get_rank(is_distributed) -> int:
    return dist.get_rank() if is_distributed else 0  # 单机默认 rank 0

def safe_get_world_size(is_distributed) -> int:
    return dist.get_world_size() if is_distributed else 1