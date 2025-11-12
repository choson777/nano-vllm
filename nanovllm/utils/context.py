from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_reqlens_q: torch.Tensor | None = None
    cu_reqlens_k: torch.Tensor | None = None
    max_reqlen_q: int = 0
    max_reqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    logit_indexs: torch.Tensor | None = None
_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_reqlens_q=None, cu_reqlens_k=None, max_reqlen_q=0, max_reqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, logit_indexs=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_reqlens_q, cu_reqlens_k, max_reqlen_q, max_reqlen_k, slot_mapping, context_lens, block_tables, logit_indexs)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
