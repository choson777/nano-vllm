from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.request import Request, RequestStatus
from nanovllm.engine.draft_engine import DraftEngine
from nanovllm.engine.target_engine import TargetEngine
from nanovllm.config import Config, DraftConfig, TargetConfig
from nanovllm import SamplingParams
from dataclasses import fields, asdict
from collections import defaultdict

from transformers import AutoTokenizer
import torch
import tqdm
from time import perf_counter
from typing import Dict, Any, List
import torch.multiprocessing as mp
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
import pickle
import atexit
import os
import sys
import traceback
from dataclasses import dataclass
import random

@dataclass
class DraftRunResult:
    """
    DraftEngine.run() 的返回结果封装
    """
    outputs: Dict[int, List[int]]          # {request_id: [token_id1, token_id2, ...]}
    logits_map: Dict[int, torch.Tensor]    # {request_id: logits_tensor} logits_tensor shape: [num_speculative_tokens, vocab_size]
    run_end_time: float

# ===================== 新增：Draft Worker 进程 =====================
def _draft_worker(
    draft_model_path: str,
    target_model_path: str,
    config_dict: Dict[str, Any],
    init_ready_event: Event,
    request_queue: mp.Queue,   # 👈 新增：接收请求
    result_ready_event: Event,
    verify_process_over_event: Event,
    shm_result_name: str,
    shm_size: int,
):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        torch.cuda.set_device(0)
        
        config = DraftConfig(**config_dict)
        tokenizer = AutoTokenizer.from_pretrained(target_model_path)
        config.eos = tokenizer.eos_token_id
        
        draft_engine = DraftEngine(draft_model_path, num_turn_spec_tokens=4, **asdict(config))
        print(f"[DRAFT WORKER] Ready on GPU {torch.cuda.current_device()}", flush=True)
        shm_result = SharedMemory(name=shm_result_name, size=shm_size)
        # 通知主进程：已准备好接收请求
        init_ready_event.set()
        
        # 👇 主循环：持续处理请求
        while True:
            try:
                # 阻塞等待请求
                cmd, data = request_queue.get()
                if cmd == "add_request":
                    # data = (prompt_tokens, sampling_params, request_id)
                    prompt, sampling_params, req_id = data
                    draft_req = Request(prompt, sampling_params, req_id)
                    draft_engine.add_request(draft_req)
                elif cmd == "run":
                    try:
                        # 执行一轮推理
                        draft_outputs, draft_req_logits_map = draft_engine.run()
                        print(draft_outputs, draft_req_logits_map)
                        run_end_time = perf_counter()
                        # 准备结果：转为 CPU 张量（必须！）
                        result = DraftRunResult(
                            outputs=draft_outputs,  # Dict[int, List[int]]
                            logits_map=draft_req_logits_map,
                            run_end_time=run_end_time,
                        )
                        
                        # 序列化到共享内存
                        payload = pickle.dumps(result)
                        if len(payload) > shm_size:
                            raise RuntimeError(
                                f"Draft run result too large: {len(payload)} > {shm_size}. "
                                "Consider reducing batch size or num_speculative_tokens."
                            )
                        
                        # 写入共享内存：[4 bytes length][payload]
                        shm_result.buf[0:4] = len(payload).to_bytes(4, 'little')
                        shm_result.buf[4:4 + len(payload)] = payload
                        
                        # 通知主进程结果就绪
                        result_ready_event.set()
                    except Exception as e:
                        print(f"[DRAFT] Error in run(): {e}", file=sys.stderr, flush=True)
                        traceback.print_exc(file=sys.stderr)
                        # 即使出错也要通知主进程（避免死锁）
                        result_ready_event.set()
                elif cmd == "verify_process":
                    req_id, num_unaccept_tokens, new_token_id = data
                    draft_engine.verify_process(req_id, num_unaccept_tokens, new_token_id)
                    verify_process_over_event.set()
                elif cmd == "reset_block_manager":
                    draft_engine.reset_block_manager()
                elif cmd == "exit":
                    break
                    
            except Exception as e:
                print(f"[DRAFT WORKER] Error processing request: {e}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
                
    except Exception as e:
        print(f"[DRAFT WORKER INIT ERROR] {e}", file=sys.stderr, flush=True)
        init_ready_event.set()  # 确保主进程不卡住
        raise

class SpeculativeEngine:
    
    def __init__(
        self,
        draft_model: str,
        target_model: str,
        num_speculative_tokens: int = 4,
        random_seed: int = 42,
        **kwargs
    ):
        if random_seed:
            torch.manual_seed(random_seed)
        
        # 提取配置参数
        draft_kwargs = {}
        target_kwargs = {}
        for k, v in kwargs.items():
            if k.startswith("draft_"):
                draft_kwargs[k[6:]] = v
            elif k.startswith("target_"):
                target_kwargs[k[7:]] = v
        
        self.draft_config = DraftConfig(**draft_kwargs)
        self.target_config = TargetConfig(**target_kwargs)
        self.num_speculative_tokens = num_speculative_tokens
        
        # 共享 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(target_model, use_fast=True)
        self.eos = self.tokenizer.eos_token_id
        self.draft_config.eos = self.eos
        self.target_config.eos = self.eos
        
        self.target_engine = TargetEngine(target_model, **asdict(self.target_config))
        print("✅ TargetEngine initialized")
        
        ctx = mp.get_context("spawn")
        shm_size = 64 * 1024 * 1024
        
        self.draft_shm_result = SharedMemory(create=True, size=shm_size)
        self.draft_result_ready_event = ctx.Event()
        self.draft_request_queue = ctx.Queue()
        self.verify_process_over_event = ctx.Event()
        init_ready_event = ctx.Event()
        
        self.draft_process = ctx.Process(
            target=_draft_worker,
            args=(
                draft_model,
                target_model,  # 👈 传给 worker 用于 tokenizer
                asdict(self.draft_config),
                init_ready_event,
                self.draft_request_queue,
                self.draft_result_ready_event,
                self.verify_process_over_event,
                self.draft_shm_result.name,  # 👈 共享内存名称
                shm_size,
            ),
            daemon=True
        )
        self.draft_process.start()
        
        # 👇 精确等待：阻塞直到 worker 调用 init_ready_event.set()
        print("⏳ Waiting for DraftEngine to initialize...")
        init_ready_event.wait()  # 阻塞直到事件被设置
        print(f"✅ DraftEngine ready on GPU2 (PID={self.draft_process.pid})")
        
        atexit.register(self._cleanup_draft)
        
    def _cleanup_draft(self):
        if hasattr(self, 'draft_process') and self.draft_process.is_alive():
            self.draft_request_queue.put(("exit", None))
            self.draft_process.join(timeout=5.0)
            if self.draft_process.is_alive():
                self.draft_process.terminate()
            self.draft_shm_result.close()
            self.draft_shm_result.unlink()
            print("✅ DraftEngine cleaned up")
        
    def add_request(self, prompt, sampling_param):
        # if not isinstance(sampling_params, list):
        #     sampling_params = [sampling_params] * len(prompts)
        # for prompt, sp in zip(prompts, sampling_params):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        target_req = Request(prompt, sampling_param)
        self.draft_request_queue.put((
            "add_request",
            (prompt, sampling_param, target_req.request_id)
        ))
        self.target_engine.add_request(target_req)         
    
    def _get_draft_run_result(self) -> tuple[dict, dict]:
        """
        从共享内存中读取 DraftEngine.run() 的结果
        Returns:
            draft_outputs: Dict[int, List[int]]
            draft_req_logits_map: Dict[int, torch.Tensor] (on GPU0)
        """
        try:
            # 1. 等待结果就绪（带超时）
            self.draft_result_ready_event.wait(timeout=None)
            
            # 2. 重置事件（为下一次 run 做准备）
            self.draft_result_ready_event.clear()
            
            # 3. 读取数据长度
            n_bytes = int.from_bytes(self.draft_shm_result.buf[0:4], 'little')
            if n_bytes <= 0 or n_bytes > self.draft_shm_result.size - 4:
                raise ValueError(f"Invalid payload size: {n_bytes}")
            
            # 4. 反序列化结果
            payload = self.draft_shm_result.buf[4:4 + n_bytes]
            result: DraftRunResult = pickle.loads(payload)
            
            receive_end_time = perf_counter()
            print(f"打包传输 解包总时间: {(receive_end_time - result.run_end_time)*1000:.2f} ms")
            return result.outputs, result.logits_map
            
        except Exception as e:
            print(f"❌ Failed to read draft result from shared memory: {e}")
            # 可选：返回空结果让系统降级
            return {}, {}
    
    def verify(self, draft_req_logits, target_req_logits, draft_outputs, target_outputs):
        common_req_ids = set(draft_req_logits.keys()) & set(target_req_logits.keys())
        verbose = False
        outputs = []
        for req_id in common_req_ids:
            draft_logits = draft_req_logits[req_id]
            target_logits = target_req_logits[req_id]
            draft_token_ids = draft_outputs[req_id]
            target_token_ids = target_outputs[req_id]
            num_round_generate = len(draft_token_ids)
            num_unaccept_tokens = num_round_generate
            # print(f"req {req_id} draft logit shape {draft_logits.shape} target_logit shape {target_logits.shape} {draft_token_ids} {target_token_ids}")
            for i in range(num_round_generate):    
                r = random.random()
                draft_logit = draft_logits[i]
                target_logit = target_logits[i]
                draft_token_id = draft_token_ids[i]
                if r > (target_logit[draft_token_id].cpu().item() / draft_logit):
                    break
                
                if verbose:
                    print(f"req id{req_id}, r is {r}, accpeted token{draft_token_ids[i]}")
                
                if draft_token_id == self.eos:
                    break
                num_unaccept_tokens -= 1
            new_token_id = target_token_ids[num_round_generate - num_unaccept_tokens]
            self.draft_request_queue.put(("verify_process", (req_id, num_unaccept_tokens, new_token_id)))
            req = self.target_engine.verify_process(req_id, num_unaccept_tokens, new_token_id)
            outputs.append(req)
            print(f"req id {req_id} unaccept token {num_unaccept_tokens}")
        return outputs        
    
    def one_round(self):
        print("========================draft=======================")
        self.draft_request_queue.put(("run", None))
        draft_outputs, draft_req_logits_map = self._get_draft_run_result()
        print(draft_outputs)
        print("========================target=======================")
        self.target_engine.integrate_draft_output(draft_outputs)
        target_outputs, target_req_logits_map = self.target_engine.run()
        print(target_outputs)
        print("========================verify=======================")
        outputs = self.verify(draft_req_logits_map, target_req_logits_map, draft_outputs, target_outputs)
        self.verify_process_over_event.wait()
        return outputs
    
    def is_finished(self):
        return self.target_engine.is_finished()
    
    def reset_block_manager(self):
        self.draft_request_queue.put(("reset_block_manager", None))
        self.target_engine.reset_block_manager()
    
    
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        ):
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        while not self.is_finished():
            output = self.one_round()
            for req in output:
                outputs[req.request_id] = req.token_ids
        outputs = [outputs[req_id] for req_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs