import torch
import math
from enum import Enum, auto
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import logging
import threading
from queue import Empty, Full, PriorityQueue, Queue
import time
from torch.profiler import record_function
from tqdm import tqdm

from specmoe.backend.forward_batch_info import ForwardBatch, DecodePart, ForwardMode
from specmoe.backend.policy import Policy, SpecPolicy
from specmoe.utils.model_config import ModelConfig
from specmoe.utils.server_args import ServerArgs
from specmoe.backend.memory import TokenToKVPoolAllocatorGPU, TokenToKVPoolAllocatorCPU, ReqToTokenPool, MHATokenToKVPool
from specmoe.layers.logits_processor import LogitsProcessorOutput
from specmoe.backend.pin_memory_allocator import make_pinned_tensor, free_pinned_tensor
from specmoe.backend.htod_transfer_engine import HtoDTransferEngine

logger = logging.getLogger(__name__)

class TransferTaskType(Enum):
    HIDDEN_LOAD = auto()
    WEIGHT_LOAD = auto()
    HIDDEN_OFFLOAD = auto()
    KV_OFFLOAD = auto()

task_priority_mapping = {
    TransferTaskType.HIDDEN_LOAD: 2,
    TransferTaskType.WEIGHT_LOAD: 1,
    TransferTaskType.HIDDEN_OFFLOAD: 2,
    TransferTaskType.KV_OFFLOAD: 1,
}

class StreamPriorityEngine:
    """基于CUDA流优先级的数据传输引擎"""
    
    def __init__(self, device="cuda"):
        # 创建具有不同优先级的CUDA流
        self.compute_stream = torch.cuda.current_stream()

        # 定义不同的优先级级别
        highest_priority = -3  # 最高优先级
        high_priority = -2    # 高优先级
        low_priority = -1     # 低优先级
        
        # 为每种操作创建单独的流
        # load操作的流
        self.hidden_load_stream = torch.cuda.Stream(
            device=device, 
            priority=highest_priority  # hidden load使用最高优先级
        )
        
        self.weight_load_stream = torch.cuda.Stream(
            device=device,
            priority=low_priority  # weight load使用低优先级
        )
        
        self.kv_cache_load_stream = torch.cuda.Stream(
            device=device,
            priority=high_priority  # kv cache load使用高优先级
        )
        self.kv_cache_load_stream2 = torch.cuda.Stream( # 用于kv cache的D2D传输
            device=device,
            priority=high_priority  # kv cache load使用高优先级
        )
    
        # offload操作的流
        self.hidden_offload_stream = torch.cuda.Stream(
            device=device,
            priority=high_priority  # hidden offload使用高优先级
        )
        
        self.kv_cache_offload_stream = torch.cuda.Stream(
            device=device,
            priority=low_priority  # KV cache offload使用低优先级
        )
        
        # 记录每层任务的事件
        self.layer_events = {}
        self.num_micro_batches = 0
        self.num_nano_batches = 0

        # 创建HtoDTransferEngine
        self.htod_transfer_engine = HtoDTransferEngine()
        self.htod_transfer_engine.start()

        self.expert_chunk_num = 16  # 每个expert切成2份
        
    def set_micro_batch_num(self, num_micro_batches: int, num_nano_batches: int):
        """设置micro-batch数量，用于初始化事件列表"""
        self.num_micro_batches = num_micro_batches
        self.num_nano_batches = num_nano_batches


    def _get_or_create_layer_events(self, layer_id: int) -> Dict[str, torch.cuda.Event]:
        """获取或创建特定层的事件集合"""
        if layer_id not in self.layer_events:
            self.layer_events[layer_id] = {
                'weight_load_events': [],  # 为每个专家块创建一个事件
                'hidden_load_event': [torch.cuda.Event() for _ in range(self.num_micro_batches)],
                'attn_event': [torch.cuda.Event() for _ in range(self.num_micro_batches)],
                'compute_event': [torch.cuda.Event() for _ in range(self.num_micro_batches)],
                'hidden_offload_event': [torch.cuda.Event() for _ in range(self.num_micro_batches)],
                'kv_offload_event': [torch.cuda.Event() for _ in range(self.num_micro_batches)],
                'layer_compute_event': [torch.cuda.Event() for _ in range(self.num_micro_batches)], 
                'load_kv_cache_event': [[torch.cuda.Event() for _ in range(self.num_nano_batches)] for __ in range(self.num_micro_batches)],
                'gpu_attn_event': [[torch.cuda.Event() for _ in range(self.num_nano_batches)] for __ in range(self.num_micro_batches)]
            }
        return self.layer_events[layer_id]
    
    def load_weight_chunked(self, layer_id: int, src: torch.Tensor, dst: torch.Tensor, num_chunks: int):
        """分块加载权重参数，一个专家一个专家地传输"""
        events = self._get_or_create_layer_events(layer_id)
        
        # 确保有足够的事件记录每个块的加载
        while len(events['weight_load_events']) < num_chunks:
            events['weight_load_events'].append(torch.cuda.Event())
        
        chunk_size = src.size(0) // num_chunks
        
        with torch.cuda.stream(self.weight_load_stream):
            if layer_id >= 2:
                # 等待前面层的所有micro-batch计算完成
                for micro_batch_id in range(self.num_micro_batches):
                    self.layer_events[layer_id-2]['compute_event'][micro_batch_id].wait(self.weight_load_stream)
            for i in range(num_chunks):
                # 计算当前块的起始和结束位置
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < num_chunks - 1 else src.size(0)
                
                # 拷贝当前块的权重
                dst[start_idx:end_idx].copy_(src[start_idx:end_idx], non_blocking=True)
                
                # 记录该块传输完成的事件
                events['weight_load_events'][i].record(self.weight_load_stream)
                # logger.debug(f"Weight chunk {i+1}/{num_chunks} loaded for layer {layer_id}")
    
    def decode_load_weight(self, layer_id: int, src: torch.Tensor, dst: torch.Tensor):
        # events = self._get_or_create_layer_events(layer_id)
        
        with torch.cuda.stream(self.weight_load_stream):
            if layer_id >= 2:
                # 等待前面层的所有micro-batch计算完成
                # for micro_batch_id in range(self.num_micro_batches): # XXX: 不能这样做，比如当前GPU正在执行layer 2 post，CPU进入此函数，prefetch layer 4 expert，但是这样会导致layer 3的prefetch被这个wait阻碍。
                #     self.layer_events[layer_id-2]['layer_compute_event'][micro_batch_id].wait(self.weight_load_stream)
                for micro_batch_id in range(self.num_micro_batches): 
                    self.layer_events[layer_id-2]['layer_compute_event'][micro_batch_id].synchronize()
                    # self.layer_events[layer_id-2]['compute_event'][micro_batch_id].synchronize()
            self.htod_transfer_engine.submit_weight_loading_task(src, dst, self.weight_load_stream, f"weight_load_layer_{layer_id}", num_chunks=self.expert_chunk_num, layer_id=layer_id)
    
    def load_kv_cache(self, need_load: bool, layer_id: int, src_indices: List[Tuple[int, int]], dst_indices: List[int], cpu_kv_pool: MHATokenToKVPool, gpu_kv_pool: MHATokenToKVPool): # DEPRECATED !!!
        events = self._get_or_create_layer_events(layer_id)
        if not need_load:
            for i in range(self.num_micro_batches):
                events['load_kv_cache_event'][i].record(self.weight_load_stream)
            return
        with torch.cuda.stream(self.kv_cache_load_stream):
            if layer_id >= 2:
                # 等待前面层的所有micro-batch计算完成
                for micro_batch_id in range(self.num_micro_batches):
                    self.layer_events[layer_id-2]['compute_event'][micro_batch_id].wait(self.weight_load_stream)
            for i in range(len(src_indices)): # 枚举micro-batch
                for j in range(len(src_indices[i])): # 枚举request
                    seq_len = src_indices[i][j][1] - src_indices[i][j][0]
                    src = cpu_kv_pool.get_kv_buffer(layer_id)[:, slice(src_indices[i][j][0], src_indices[i][j][1]), ...] # 读取此request的kv cache
                    # 方法1：连续传输
                    # XXX: 这里会不会存在问题呢，因为gpu indicies可能不连续，导致不能复制成功？需要检查一下不同轮次的gpu slot分配情况
                    begin_idx = dst_indices[i][j][0]
                    gpu_kv_pool.load_kv_cache_from_cpu(layer_id, slice(begin_idx, begin_idx + seq_len), src, non_blocking=True, gpu_two_buffer=True) 
                    # 方法2：逐个传输
                    # for k in range(seq_len):
                    #     gpu_kv_pool.load_kv_cache_from_cpu(layer_id, dst_indices[i][j][k], src[:, k, ...], non_blocking=True, gpu_two_buffer=True)
                events['load_kv_cache_event'][i].record(self.weight_load_stream)
    
    def get_two_buffer_offset(self, layer_id, micro_batch_id, nano_batch_id):
        # return nano_batch_id % 2
        return 0
    
    def decode_load_kv_cache(self, need_load: bool, layer_id: int, src_indices: List[Tuple[int, int]], dst_indices: List[int], cpu_kv_pool: MHATokenToKVPool, gpu_kv_pool: MHATokenToKVPool, micro_batch_id: int, nano_batch_id: int, begin: int, end: int):
        events = self._get_or_create_layer_events(layer_id)
        if not need_load:
            events['load_kv_cache_event'][micro_batch_id][nano_batch_id].record(self.kv_cache_load_stream)
            return
        src_list = []
        dst_list = []
        for j in range(begin, end): # 枚举request
            seq_len = src_indices[micro_batch_id][j][1] - src_indices[micro_batch_id][j][0]
            src = cpu_kv_pool.get_kv_buffer(layer_id)[:, slice(src_indices[micro_batch_id][j][0], src_indices[micro_batch_id][j][1]), ...] # 读取此request的kv cache
            begin_idx = dst_indices[micro_batch_id][j][0]
            src_list.append(src)
            dst_list.append(slice(begin_idx, begin_idx + seq_len))
        self.htod_transfer_engine.submit_kv_cache_loading_task_in_batch(src_list, dst_list, gpu_kv_pool, events['load_kv_cache_event'][micro_batch_id][nano_batch_id], self.kv_cache_load_stream, self.kv_cache_load_stream2, f"load_kv_cache_layer_{layer_id}_mb_{micro_batch_id}_nb_{nano_batch_id}", layer_id, two_buffer_offset=self.get_two_buffer_offset(layer_id, micro_batch_id, nano_batch_id))
        # for j in range(begin, end): # 枚举request
        #     seq_len = src_indices[micro_batch_id][j][1] - src_indices[micro_batch_id][j][0]
        #     src = cpu_kv_pool.get_kv_buffer(layer_id)[:, slice(src_indices[micro_batch_id][j][0], src_indices[micro_batch_id][j][1]), ...] # 读取此request的kv cache
        #     # 方法1：连续传输
        #     # XXX: 这里会不会存在问题呢，因为gpu indicies可能不连续，导致不能复制成功？需要检查一下不同轮次的gpu slot分配情况
        #     begin_idx = dst_indices[micro_batch_id][j][0]
        #     self.htod_transfer_engine.submit_kv_cache_loading_task(src, slice(begin_idx, begin_idx + seq_len), gpu_kv_pool, events['load_kv_cache_event'][micro_batch_id][nano_batch_id], self.kv_cache_load_stream, f"load_kv_cache_layer_{layer_id}_mb_{micro_batch_id}_nb_{nano_batch_id}", layer_id, two_buffer_offset=nano_batch_id % 2)
        #     # 方法2：逐个传输
        #     # for k in range(seq_len):
        #     #     gpu_kv_pool.load_kv_cache_from_cpu(layer_id, dst_indices[micro_batch_id][j][k], src[:, k, ...], non_blocking=True, gpu_two_buffer=True)
        events['load_kv_cache_event'][micro_batch_id][nano_batch_id].record(self.kv_cache_load_stream)
        # events['load_kv_cache_event'][micro_batch_id][nano_batch_id].record(self.kv_cache_load_stream2) # 最后一个执行的是D2D传输

    def decode_load_kv_cache2(self, need_load: bool, layer_id: int, src_pinned: torch.Tensor, src_indices: List[Tuple[int, int]], dst_indices: List[int], gpu_kv_pool: MHATokenToKVPool, micro_batch_id: int, nano_batch_id: int, indptr: torch.Tensor, begin: int, end: int):
        events = self._get_or_create_layer_events(layer_id)
        if not need_load:
            events['load_kv_cache_event'][micro_batch_id][nano_batch_id].record(self.kv_cache_load_stream2)
            return
        src_list = []
        dst_list = []
        for j in range(begin, end): # 枚举request
            src_list.append((indptr[j], indptr[j+1]))
            dst_list.append(slice(dst_indices[micro_batch_id][j][0], dst_indices[micro_batch_id][j][1]))
        self.htod_transfer_engine.submit_kv_cache_loading_task_in_batch2(src_pinned, src_list, dst_list, gpu_kv_pool, events['load_kv_cache_event'][micro_batch_id][nano_batch_id], self.kv_cache_load_stream, layer_id, two_buffer_offset=self.get_two_buffer_offset(layer_id, micro_batch_id, nano_batch_id))

        events['load_kv_cache_event'][micro_batch_id][nano_batch_id].record(self.kv_cache_load_stream)
    
    
    def load_hidden_prefill(self, layer_id: int, micro_batch_id: int, batch: ForwardBatch):
        events = self._get_or_create_layer_events(layer_id)
        with torch.cuda.stream(self.hidden_load_stream):
            if layer_id > 0:
                self.layer_events[layer_id-1]['compute_event'][micro_batch_id].wait(self.hidden_load_stream)
            batch.hidden_states = batch.hidden_states_cpu.to('cuda', non_blocking=True)
            batch.residual = batch.residual_cpu.to('cuda', non_blocking=True)
            events['hidden_load_event'][micro_batch_id].record(self.hidden_load_stream)
    
    def load_hidden_decode(self, layer_id: int, micro_batch_id: int, src: torch.Tensor, dst: torch.Tensor):
        """加载hidden states"""
        events = self._get_or_create_layer_events(layer_id)
        self.htod_transfer_engine.submit_hidden_loading_task(src, dst, self.hidden_load_stream, events['hidden_load_event'][micro_batch_id])
        # with torch.cuda.stream(self.hidden_load_stream):
        #     # 等待前一层对应micro-batch的hidden load完成
        #     # if layer_id > 0:
        #     #     self.layer_events[layer_id-1]['hidden_load_event'][micro_batch_id].wait(self.hidden_load_stream)
        #     dst.copy_(src, non_blocking=True)
        #     # 记录事件
        #     events['hidden_load_event'][micro_batch_id].record(self.hidden_load_stream)
        #     # logger.debug(f"Hidden loaded for layer {layer_id}, batch {micro_batch_id}")
    
    def wait_for_layer_load(self, layer_id: int, stage: str = "prefill", decode_performance_recoder: dict = None):
        """等待特定层的加载完成"""
        if stage == "prefill":
            if layer_id in self.layer_events:
                events = self.layer_events[layer_id]
                
                # 等待所有权重块加载完成
                num_chunks = len(events['weight_load_events'])
                
                for i in range(num_chunks):
                    events['weight_load_events'][i].wait(self.compute_stream)
        elif stage == "decode":
            # if layer_id == 0: # 第一层不需要等待
            #     return
            # else:
            if self.htod_transfer_engine.has_task_id(f"weight_load_layer_{layer_id}"):
                ret = self.htod_transfer_engine.wait_for_completion(f"weight_load_layer_{layer_id}", timeout=None)
                # 处理WEIGHT_LOADING任务的返回值（包含执行时间）
                if isinstance(ret, tuple):
                    success, execution_time = ret
                    # print(f"layer {layer_id} weight load time: {execution_time}")
                    assert success
                    if execution_time is not None and decode_performance_recoder is not None:
                        decode_performance_recoder["htod_transfer_time"] += execution_time
                else:
                    assert ret
                if layer_id == 0:
                    self.htod_transfer_engine.clear_condition(["weight_load_layer_0"])
            else:
                assert layer_id == 0, "只有第一个decode的第一层才没有"
        else:
            assert False, f"Invalid stage: {stage}"

    
    def wait_for_hidden_load_prefill(self, layer_id: int, micro_batch_id: int = None):
        if layer_id in self.layer_events:
            events = self.layer_events[layer_id]
            # 等待hidden states加载完成
            if micro_batch_id is not None:
                events['hidden_load_event'][micro_batch_id].wait(self.compute_stream)
            else:
                # 等待所有micro-batch的hidden load完成
                for mb_id in range(self.num_micro_batches):
                    events['hidden_load_event'][mb_id].wait(self.compute_stream)
            
            # logger.debug(f"Layer {layer_id} load complete, computation can start")

    def wait_for_last_kvcache_offload_prefill(self, layer_id, micro_batch_id):
        if layer_id == 0 and micro_batch_id == 0:
            return
        if micro_batch_id == 0:
            self.layer_events[layer_id-1]['kv_offload_event'][self.num_micro_batches-1].wait(self.compute_stream)
        else:
            self.layer_events[layer_id]['kv_offload_event'][micro_batch_id-1].wait(self.compute_stream)
    
    def record_compute_done(self, layer_id: int, micro_batch_id: int):
        """记录计算完成事件"""
        events = self._get_or_create_layer_events(layer_id)
        events['compute_event'][micro_batch_id].record(self.compute_stream)
        # logger.debug(f"Compute done for layer {layer_id}, batch {micro_batch_id}")
    
    def record_attn_done(self, layer_id: int, micro_batch_id: int):
        """记录attention完成事件"""
        events = self._get_or_create_layer_events(layer_id)
        events['attn_event'][micro_batch_id].record(self.compute_stream)
        # logger.debug(f"Attention done for layer {layer_id}, batch {micro_batch_id}")
    
    def offload_hidden(self, layer_id: int, micro_batch_id: int, src: torch.Tensor, dst: torch.Tensor, src2: torch.Tensor = None, dst2: torch.Tensor = None):
        """卸载hidden states到CPU"""
        events = self._get_or_create_layer_events(layer_id)
        
        with torch.cuda.stream(self.hidden_offload_stream):
            # 等待对应micro-batch的计算完成
            events['compute_event'][micro_batch_id].wait(self.hidden_offload_stream)
            # 卸载hidden
            dst.copy_(src, non_blocking=True)
            if src2 is not None and dst2 is not None:
                dst2.copy_(src2, non_blocking=True)
            # 记录事件
            events['hidden_offload_event'][micro_batch_id].record(self.hidden_offload_stream)
            # logger.debug(f"Hidden offloaded for layer {layer_id}, batch {micro_batch_id}")
    
    def offload_kv_cache_prefill(self, layer_id: int, micro_batch_id: int, src_indices: torch.Tensor, dst_indices: torch.Tensor, gpu_token_to_kv_pool_allocator: TokenToKVPoolAllocatorGPU, cpu_token_to_kv_pool_allocator: TokenToKVPoolAllocatorCPU, seq_lens: torch.Tensor):
        """卸载KV cache到CPU"""
        events = self._get_or_create_layer_events(layer_id)
        
        with torch.cuda.stream(self.kv_cache_offload_stream):
            # 等待对应micro-batch的attention完成
            events['attn_event'][micro_batch_id].wait(self.kv_cache_offload_stream)
            # 卸载KV cache
            # dst.copy_(src, non_blocking=True)
            src = gpu_token_to_kv_pool_allocator.get_layer_kv_cache(0, src_indices)
            cpu_token_to_kv_pool_allocator.offload_kv_cache_prefill(layer_id, src, dst_indices, seq_lens)
            # 记录事件
            events['kv_offload_event'][micro_batch_id].record(self.kv_cache_offload_stream)
            # logger.debug(f"KV cache offloaded for layer {layer_id}, batch {micro_batch_id}")
    
    def offload_qkv(self, layer_id: int, micro_batch_id: int, src: torch.Tensor, dst: torch.Tensor, src2: torch.Tensor = None, dst2: torch.Tensor = None):
        """Offload QKV to qkv_pin"""
        events = self._get_or_create_layer_events(layer_id)
        with torch.cuda.stream(self.kv_cache_offload_stream):
            events['compute_event'][micro_batch_id].wait(self.kv_cache_offload_stream)
            # logger.debug(f"src shape: {src.shape}, dst shape: {dst.shape}")
            dst.copy_(src, non_blocking=True)
            if src2 is not None and dst2 is not None:
                dst2.copy_(src2, non_blocking=True)
            events['kv_offload_event'][micro_batch_id].record(self.kv_cache_offload_stream)
            # logger.debug(f"QKV offloaded for layer {layer_id}, batch {micro_batch_id}")
    
    def wait_for_offload_complete(self, layer_id: int, micro_batch_id: int = None):
        """等待特定层的offload完成"""
        if layer_id in self.layer_events:
            events = self.layer_events[layer_id]
            if micro_batch_id is not None:
                # 等待特定micro-batch的offload完成
                events['hidden_offload_event'][micro_batch_id].synchronize()
                events['kv_offload_event'][micro_batch_id].synchronize()
            else:
                # 等待所有micro-batch的offload完成
                for mb_id in range(self.num_micro_batches):
                    events['hidden_offload_event'][mb_id].synchronize()
                    events['kv_offload_event'][mb_id].synchronize()
            # logger.debug(f"Layer {layer_id} offload complete")
    
    def synchronize_all(self):
        """同步所有流"""
        self.hidden_load_stream.synchronize()
        self.hidden_offload_stream.synchronize()
        self.weight_load_stream.synchronize()
        self.kv_cache_offload_stream.synchronize()
        self.compute_stream.synchronize()
        logger.debug("All streams synchronized")
    
    def reset_events(self):
        self.layer_events = {}

class DraftStreamPriorityEngine:
    def __init__(self):
        pass

    def set_micro_batch_num(self, num_micro_batches: int):
        pass

class ExecutionEngine:

    def __init__(
        self,
        model,
        model_config: ModelConfig,
        server_args: ServerArgs,
        policy: Policy,
        cpu_req_to_token_pool: ReqToTokenPool,
        gpu_req_to_token_pool: ReqToTokenPool,
        cpu_token_to_kv_pool_allocator: TokenToKVPoolAllocatorCPU,
        gpu_token_to_kv_pool_allocator: TokenToKVPoolAllocatorGPU,
    ):
        self.model = model
        self.model_config = model_config
        self.server_args = server_args
        self.policy = policy

        self.micro_batches: List[ForwardBatch] = None

        # TODO: Lazy initialization, init it in the first real model forward
        self.context = ExecutionContext.build_context(model_config, server_args, policy, cpu_req_to_token_pool, gpu_req_to_token_pool, cpu_token_to_kv_pool_allocator, gpu_token_to_kv_pool_allocator)

        # 使用新的基于流优先级的引擎替代传输引擎
        self.stream_engine = StreamPriorityEngine()

        self.cpu_attn_futures = None
        self.copy_futures = None

        # experts cache mapping: experts_id -> idx in self.context.experts_cache
        self.experts_mapping = [torch.empty(self.model_config.num_local_experts, dtype=torch.int64, device="cuda") for _ in range(self.model_config.num_hidden_layers)]

        # TODO:
        # kv cache mapping: tokens' slot in cpu kv cache -> tokens' slot in gpu kv cache
        # self.kv_mapping = [torch.empty(self.model_config.num_local_experts, dtype=torch.int64, device="cuda") for _ in range(self.model_config.num_hidden_layers)]

        self.performance_recoder = {
            "cpu_attention_time": 0,
            "gpu_moe_time": 0, 
            "htod_transfer_time": 0,
            "post_attention_time": 0,
        }
        self.cpu_time_record = []
        self.gpu_time_record = []
        
        # post_attention timing events - 初始化为None，在update_before_forward中根据micro_batch数量分配
        self.post_attention_start_events = None
        self.post_attention_end_events = None
    
    def reset_performance_recoder(self, micro_batch_num: int):
        self.performance_recoder = {
            "cpu_attention_time": 0,
            "gpu_moe_time": 0, 
            "htod_transfer_time": 0,
            "post_attention_time": 0,
        }
        self.cpu_time_record = [0 for _ in range(micro_batch_num)]
        self.gpu_time_record = [0 for _ in range(micro_batch_num)]
        # 为每个layer和micro_batch分配post_attention timing events
        self.post_attention_start_events = [[torch.cuda.Event(enable_timing=True) for _ in range(micro_batch_num)] for _ in range(self.model_config.num_hidden_layers)]
        self.post_attention_end_events = [[torch.cuda.Event(enable_timing=True) for _ in range(micro_batch_num)] for _ in range(self.model_config.num_hidden_layers)]
        

    def switch_to_decode(self, decode_policy: Policy):
        self.policy = decode_policy
        self.stream_engine.reset_events()
        self.model.del_experts_cache()
        self.context.change_expert_cache_size(decode_policy)
        self.context.change_pin_size(decode_policy, self.server_args)
        self.context.init_kv_cache_pin(decode_policy, self.server_args)
        self.init_gpu_experts()

    def _get_prefetch_cpu_slice(self, layer_id: int, stage: str):
        num_gpu_expert = int(self.model_config.num_local_experts * self.policy.weight_cache_ratio)
        num_cpu_expert = self.model_config.num_local_experts - num_gpu_expert
        return slice(num_gpu_expert, num_gpu_expert + num_cpu_expert)

    def _get_prefetch_gpu_slice(self, layer_id: int, stage: str):
        num_gpu_expert = int(self.model_config.num_local_experts * self.policy.weight_cache_ratio)
        num_cpu_expert = self.model_config.num_local_experts - num_gpu_expert
        gpu_start_pos = num_gpu_expert * self.model_config.num_hidden_layers + (layer_id % 2) * num_cpu_expert
        return slice(gpu_start_pos, gpu_start_pos + num_cpu_expert)

    def init_gpu_experts(self):
        begin_time = time.time()
        self.context.init_gpu_experts(self.model)
        # link the experts cache to the model
        self.model.link_gpu_experts_cache(self.context.experts_cache)
        num_gpu_experts = int(self.model_config.num_local_experts * self.policy.weight_cache_ratio)
        if num_gpu_experts > 0:
            for i in range(self.model_config.num_hidden_layers):
                self.experts_mapping[i][:num_gpu_experts] = torch.arange(i * num_gpu_experts, (i + 1) * num_gpu_experts, dtype=torch.int64, device="cuda")

        num_cpu_experts = self.model_config.num_local_experts - num_gpu_experts
        for i in range(self.model_config.num_hidden_layers):
            start_pos_cache = num_gpu_experts * self.model_config.num_hidden_layers + (i % 2) * num_cpu_experts
            self.experts_mapping[i][num_gpu_experts:] = torch.arange(start_pos_cache, start_pos_cache + num_cpu_experts, dtype=torch.int64, device="cuda")

        # logger.debug(f"experts_mapping: {self.experts_mapping}")
        # assert self.model_config.num_hidden_layers % 2 == 0
        logger.info(f"Execution Engine: init gpu experts time: {time.time() - begin_time:.2f} s")
        
        
    def prefill(self, batch: ForwardBatch, decode_part: DecodePart, layer_id: int, experts_mapping: torch.Tensor, attn_event: Optional[torch.cuda.Event] = None, micro_batch_id: int = -1):
        batch.decode_part = decode_part
        batch.attn_event = attn_event
        batch.experts_mapping = experts_mapping
        return self.model.forward(input_ids=batch.input_ids,
                                  positions=batch.positions,
                                  forward_batch=batch,
                                  hidden_states=batch.hidden_states,
                                  residual=batch.residual,
                                  cur_layers = [layer_id],
                                  micro_batch_id=micro_batch_id,
                                  )

    def update_before_forward(self, micro_batches: List[ForwardBatch]):
        self.micro_batches = micro_batches
        # 设置StreamPriorityEngine的micro-batch数量
        nano_batch_num = 1
        if self.policy.gpu_attention_nano_batch_size is not None:
            # if self.policy.gpu_attention_nano_batch_size > 0:
            #     nano_batch_num = (self.policy.gpu_attention_micro_batch_size + self.policy.gpu_attention_nano_batch_size - 1) // self.policy.gpu_attention_nano_batch_size
            nano_batch_num = self.policy.gpu_attention_micro_batch_size # 最不会出错的方式
        self.stream_engine.set_micro_batch_num(len(micro_batches), nano_batch_num)
        self.stream_engine.reset_events()

        self.cpu_attn_futures = [None for _ in range(len(micro_batches))]
        self.gpu_attn_futures = [None for _ in range(len(micro_batches))]
        self.copy_futures = None # [None for _ in range(len(micro_batches))]
        self.kv_cache_copy_futures = [None for _ in range(self.model_config.num_hidden_layers)] # used for kv cache loading

        self.gpu_attn_output = [None for _ in range(len(micro_batches))]
        if self.policy.stage == 'decode':
            if self.server_args.decode_gpu_attention_ratio > 0:
                self.prefetch_kvcache_cpu_indicies = [] # [[(begine, end), (begine, end), ...], [(begine, end), (begine, end), ...], ...]
                self.prefetch_kvcache_gpu_indicies = [] # [[kv_indices0, kv_indices1, ...], [kv_indices0, kv_indices1, ...], ...]
                self.kv_cache_pin_begin = [0] # 每个micro-batch的kv cache在kv_cache_pin中的起始位置
                for mb in micro_batches:
                    rids = mb.gpu_batch_rids
                    gpu_seq_len_list = mb.gpu_seq_lens.cpu().tolist()
                    self.kv_cache_pin_begin.append(self.kv_cache_pin_begin[-1] + sum(gpu_seq_len_list))
                    cpu_start_locs = self.context.cpu_token_to_kv_pool_allocator.get_start_loc(rids)
                    cpu_end_locs = [start + gpu_seq_len_list[i] for i, start in enumerate(cpu_start_locs)]
                    tmp_cpu_indices = [(start, end) for start, end in zip(cpu_start_locs, cpu_end_locs)]
                    self.prefetch_kvcache_cpu_indicies.append(tmp_cpu_indices)
                    tmp_gpu_indices = []
                    for j in range(len(gpu_seq_len_list)):
                        # tmp_gpu_indices.append(mb.gpu_req_to_token_pool.req_to_token[mb.gpu_req_pool_indices[j], :mb.gpu_seq_lens[j]])
                        # tmp_gpu_indices.append(mb.gpu_req_to_token_pool.req_to_token[mb.gpu_req_pool_indices[j], :mb.gpu_seq_lens[j]].tolist())
                        gpu_slice_begin = mb.gpu_req_to_token_pool.req_to_token[mb.gpu_req_pool_indices[j], 0].cpu().item()
                        tmp_gpu_indices.append((gpu_slice_begin, gpu_slice_begin + gpu_seq_len_list[j]))
                    self.prefetch_kvcache_gpu_indicies.append(tmp_gpu_indices)
                    mb.gpu_seq_lens_list = gpu_seq_len_list
            else:
                self.prefetch_kvcache_cpu_indicies = None
                self.prefetch_kvcache_gpu_indicies = None

    def prefetch_experts(self, layer_id: int, stage: str = "prefill"):
        """预取专家参数，按专家分块传输"""
        # logger.debug(f"Prefetch experts for layer {layer_id}, stage {stage}")
        if stage == "prefill":
            prefetch_gpu_slice = self._get_prefetch_gpu_slice(layer_id, stage)
            prefetch_cpu_slice = self._get_prefetch_cpu_slice(layer_id, stage)

            # logger.debug(f"prefetch layer {layer_id}, prefetch_gpu_slice: {prefetch_gpu_slice}, prefetch_cpu_slice: {prefetch_cpu_slice}")

            src = self.model.get_experts_cache(layer_id)[prefetch_cpu_slice]
            # logger.debug(f"src: {src}")
            dst = self.context.experts_cache[prefetch_gpu_slice]

            # 计算专家数量
            num_cpu_experts = self.model_config.num_local_experts - int(self.model_config.num_local_experts * self.policy.weight_cache_ratio)

            # 使用分块方式加载专家权重
            # 每个专家作为一个块
            self.stream_engine.load_weight_chunked(layer_id, src, dst, num_chunks=num_cpu_experts)
            # logger.debug(f"Started loading {num_cpu_experts} expert chunks for layer {layer_id}")
        elif stage == "decode":
            # self.context.prefetch_executor.submit(self.prefetch_experts_func, layer_id, stage)
            num_cpu_experts = self.model_config.num_local_experts - int(self.model_config.num_local_experts * self.policy.weight_cache_ratio)
            if self.copy_futures is not None:
                # assert False, "Now, new HtoDTransferEngine is not support pin_experts_when_load_weight=False"
                self.copy_futures.result()

            with torch.cuda.stream(self.stream_engine.weight_load_stream):
                cpu_slice = self._get_prefetch_cpu_slice(layer_id, 'decode')
                self.context.experts_pin = self.model.get_experts_cache(layer_id)[cpu_slice]

                prefetch_gpu_slice = self._get_prefetch_gpu_slice(layer_id, stage)
                src = self.context.experts_pin
                dst = self.context.experts_cache[prefetch_gpu_slice]
                # logger.debug(f"prefetch_experts_to_pin: src shape: {src.shape}, dst shape: {dst.shape}")
                # self.stream_engine.load_weight_chunked(layer_id, src, dst, num_chunks=num_cpu_experts)
                self.stream_engine.decode_load_weight(layer_id, src, dst)
            # logger.debug(f"Started loading {num_cpu_experts} expert chunks for layer {layer_id}")
        else: 
            assert False, f"Invalid stage: {stage}"
    
    def prefetch_kv_cache(self, layer_id: int, policy: Policy):
        need_load = policy.gpu_attention_ratio > 0
        self.stream_engine.load_kv_cache(need_load, layer_id, self.prefetch_kvcache_cpu_indicies, self.prefetch_kvcache_gpu_indicies, self.context.cpu_token_to_kv_pool_allocator.get_kvcache(), self.context.gpu_token_to_kv_pool_allocator.get_kvcache())
    
    def prefetch_experts_func(self, layer_id: int, stage: str='decode'):
        num_cpu_experts = self.model_config.num_local_experts - int(self.model_config.num_local_experts * self.policy.weight_cache_ratio)
        self.copy_futures.result()
        prefetch_gpu_slice = self._get_prefetch_gpu_slice(layer_id, stage)
        intermediate_size = self.model_config.intermediate_size // self.server_args.tp_size
        src = self.context.experts_pin
        dst = self.context.experts_cache[prefetch_gpu_slice]
        self.stream_engine.load_weight_chunked(layer_id, src, dst, num_chunks=num_cpu_experts)
        # logger.debug(f"Started loading {num_cpu_experts} expert chunks for layer {layer_id}")

    def prefetch_experts_to_pin(self, layer_id: int):
        self.copy_futures = self.context.copy_executor.submit(self.prefetch_experts_to_pin_func, layer_id)

    def prefetch_experts_to_pin_func(self, layer_id: int):
        with record_function("prefetch_experts_to_pin_func"):
            if layer_id != 0:
                events = self.stream_engine._get_or_create_layer_events(layer_id - 1)
                for weight_load_event in events["weight_load_events"]:
                    weight_load_event.synchronize()
            if not self.server_args.pin_experts_when_load_weight:
                assert False, "Now, new HtoDTransferEngine is not support pin_experts_when_load_weight=False"

            else:
                cpu_slice = self._get_prefetch_cpu_slice(layer_id, 'decode')
                self.context.experts_pin = self.model.get_experts_cache(layer_id)[cpu_slice]

    def prefill_kv_cache_offload(self, layer_id: int, micro_batch_id: int, stage: str = "prefill"):
        """卸载KV cache"""
        if stage == "prefill":
            src_indices = self.micro_batches[micro_batch_id].gpu_out_cache_loc
            dst_indices = self.micro_batches[micro_batch_id].cpu_out_cache_loc
            # src = self.context.gpu_token_to_kv_pool_allocator.get_layer_kv_cache(0, src_indices)
            # dst = self.context.cpu_token_to_kv_pool_allocator.get_layer_kv_cache(layer_id, dst_indices)

            # logger.debug(f"kv cache offload: src: {src.shape}, dst: {dst.shape}, src_indices: {src_indices}, dst_indices: {dst_indices}")

            # 使用流引擎卸载KV cache
            self.stream_engine.offload_kv_cache_prefill(layer_id, micro_batch_id, src_indices=src_indices, dst_indices=dst_indices, gpu_token_to_kv_pool_allocator=self.context.gpu_token_to_kv_pool_allocator, cpu_token_to_kv_pool_allocator=self.context.cpu_token_to_kv_pool_allocator, seq_lens = self.micro_batches[micro_batch_id].seq_lens)
            # logger.debug(f"offload_kv_cache, cpu cache: {self.context.cpu_token_to_kv_pool_allocator.get_layer_kv_cache(layer_id, dst_indices)}")
        else:
            assert False, f"Invalid stage: {stage}"

    def prefill_hidden_offload(self, layer_id: int, micro_batch_id: int, logitoutput: LogitsProcessorOutput = None):
        """卸载hidden states"""
        if self.micro_batches[micro_batch_id].hidden_states is not None:
            # logger.debug("prefill hidden offload, layer_id: %d, micro_batch_id: %d", layer_id, micro_batch_id)
            if self.micro_batches[micro_batch_id].hidden_states_cpu is None:
                self.micro_batches[micro_batch_id].hidden_states_cpu = torch.empty_like(
                    self.micro_batches[micro_batch_id].hidden_states, device="cpu"
                ).pin_memory()
                self.micro_batches[micro_batch_id].residual_cpu = torch.empty_like(
                    self.micro_batches[micro_batch_id].residual, device="cpu"
                ).pin_memory()

            # 使用流引擎卸载hidden states
            if layer_id != self.model_config.num_hidden_layers - 1:
                self.stream_engine.offload_hidden(
                    layer_id,
                    micro_batch_id,
                    self.micro_batches[micro_batch_id].hidden_states,
                    self.micro_batches[micro_batch_id].hidden_states_cpu,
                    self.micro_batches[micro_batch_id].residual,
                    self.micro_batches[micro_batch_id].residual_cpu
                )
            self.micro_batches[micro_batch_id].hidden_states = None 
            self.micro_batches[micro_batch_id].residual = None
        
        if logitoutput is not None:
            if logitoutput.hidden_states is not None:
                assert layer_id == self.model_config.num_hidden_layers - 1, "logitoutput should only be used in the last layer"
                with torch.cuda.stream(self.stream_engine.hidden_offload_stream):
                    if logitoutput.hidden_states_cpu is None:
                        logitoutput.hidden_states_cpu = torch.empty_like(
                            logitoutput.hidden_states, device="cpu"
                        ).pin_memory()
                    logitoutput.hidden_states_cpu.copy_(logitoutput.hidden_states, non_blocking=True)
                    logitoutput.hidden_states = None 
                    self.stream_engine.layer_events[layer_id]["hidden_offload_event"][micro_batch_id].record(self.stream_engine.hidden_offload_stream)

    def prefill_hidden_load(self, layer_id: int, micro_batch_id: int):
        """加载hidden states"""
        if self.micro_batches[micro_batch_id].hidden_states_cpu is not None:
            # logger.debug("prefill hidden load, layer_id: %d, micro_batch_id: %d", layer_id, micro_batch_id)
            self.stream_engine.load_hidden_prefill(
                layer_id,
                micro_batch_id,
                self.micro_batches[micro_batch_id]
            )

    def pre_attention(self, layer_id: int, micro_batch_id: int):
        """QKV Calculation"""
        with record_function("pre_attention"):
            batch = self.micro_batches[micro_batch_id]
            if batch.forward_mode != ForwardMode.TARGET_VERIFY:
                batch.forward_mode = ForwardMode.DECODE
            batch.decode_part = DecodePart.PREATTN
            # logger.debug(f"IN Pre_Attention, batch input_ids: {batch.input_ids}")
            batch.qkv, batch.residual = self.model.forward(input_ids=batch.input_ids,
                                                            positions=batch.positions,
                                                            forward_batch=batch,
                                                            hidden_states=batch.hidden_states,
                                                            residual=batch.residual,
                                                            cur_layers=[layer_id])
            # 记录计算完成事件
            self.stream_engine.record_compute_done(layer_id, micro_batch_id)

    def decode_load_hidden(self, layer_id: int, micro_batch_id: int):
        """Load hidden states"""
        # 等待CPU attention计算完成
        self.cpu_attn_futures[micro_batch_id].result()

        # 等待上一个post_attention完成
        with torch.cuda.stream(self.stream_engine.hidden_load_stream):
            if layer_id != 0 or micro_batch_id != 0: # (0, 0) 不处理
                if micro_batch_id == 0:
                    events = self.stream_engine._get_or_create_layer_events(layer_id - 1)
                    # events["layer_compute_event"][len(self.micro_batches) - 1].wait(self.stream_engine.hidden_load_stream)
                    events["layer_compute_event"][len(self.micro_batches) - 1].synchronize()
                else:
                    events = self.stream_engine._get_or_create_layer_events(layer_id)
                    # events["layer_compute_event"][micro_batch_id - 1].wait(self.stream_engine.hidden_load_stream)
                    events["layer_compute_event"][micro_batch_id - 1].synchronize()
            bs = self.micro_batches[micro_batch_id].batch_size
            src = self.micro_batches[micro_batch_id].hidden_pin[:bs]
            self.micro_batches[micro_batch_id].hidden_states = torch.empty_like(src, device="cuda")
            dst = self.micro_batches[micro_batch_id].hidden_states
            # logger.debug(f"src shape: {src.shape}, dst shape: {dst.shape}, bs: {bs}")
            self.stream_engine.load_hidden_decode(layer_id, micro_batch_id, src, dst)

    def post_attention(self, layer_id: int, micro_batch_id: int):
        """Post Attention"""
        with record_function("post_attention"):
            # 记录开始时间
            
            # wait for the last prefetch event - 只在第一个micro_batch时记录htod_transfer_time
            if micro_batch_id == 0:
                self.stream_engine.wait_for_layer_load(layer_id, stage="decode", decode_performance_recoder=self.performance_recoder)
            else:
                # 其他micro_batch只等待，不记录时间
                self.stream_engine.wait_for_layer_load(layer_id, stage="decode", decode_performance_recoder=None)
            # events["gpu_attn_event"][micro_batch_id].wait(self.context.cur_stream)
            self.gpu_attn_futures[micro_batch_id].result()
            # wait for hidden states from CPU to GPU
            events = self.stream_engine._get_or_create_layer_events(layer_id)
            events["hidden_load_event"][micro_batch_id].wait(self.context.cur_stream)

            if self.policy.stage == 'decode' and self.post_attention_start_events is not None:
                self.post_attention_start_events[layer_id][micro_batch_id].record(self.context.cur_stream)
            
            # 拼接cpu_attn_output和gpu_attn_output
            if self.gpu_attn_output[micro_batch_id] is not None:
                # if layer_id == 0 and micro_batch_id == 0:
                    # print(f"post_attention, self.gpu_attn_output[micro_batch_id]: {self.gpu_attn_output[micro_batch_id]}")
                    # print(f"post_attention, self.gpu_attn_output[micro_batch_id].shape: {self.gpu_attn_output[micro_batch_id].shape}")
                # 在明确的stream context中执行cat操作，并立即同步
                with torch.cuda.stream(self.context.cur_stream):
                    self.micro_batches[micro_batch_id].hidden_states = torch.cat(
                        [
                            self.gpu_attn_output[micro_batch_id].view(
                                -1,
                                self.gpu_attn_output[micro_batch_id].shape[2],
                                self.gpu_attn_output[micro_batch_id].shape[3]
                            ),
                            self.micro_batches[micro_batch_id].hidden_states.view(
                                -1,
                                self.micro_batches[micro_batch_id].hidden_states.shape[2],
                                self.micro_batches[micro_batch_id].hidden_states.shape[3]
                            )
                        ],
                        dim=0
                    )
                # 确保cat操作在此stream上完成
                self.context.cur_stream.synchronize()
                self.gpu_attn_output[micro_batch_id] = None # 释放引用

            batch = self.micro_batches[micro_batch_id]
            if batch.forward_mode != ForwardMode.TARGET_VERIFY:
                batch.forward_mode = ForwardMode.DECODE
            batch.decode_part = DecodePart.POSTATTN
            batch.experts_mapping = self.experts_mapping[layer_id]
            # logger.debug(f"input_ids: {batch.input_ids}, positions: {batch.positions}, hidden_states: {batch.hidden_states}, residual: {batch.residual}, experts_mapping: {self.experts_mapping[layer_id]}")
            res = None
            if layer_id != self.model_config.num_hidden_layers - 1:
                batch.hidden_states, batch.residual = self.model.forward(input_ids=batch.input_ids, positions=batch.positions, forward_batch=batch, hidden_states=batch.hidden_states, residual=batch.residual, cur_layers=[layer_id]) # TODO:
            else:
                res = self.model.forward(input_ids=batch.input_ids, positions=batch.positions, forward_batch=batch, hidden_states=batch.hidden_states, residual=batch.residual, cur_layers=[layer_id]) 
            events["layer_compute_event"][micro_batch_id].record(self.context.cur_stream)
            
            # 记录结束时间
            if self.policy.stage == 'decode' and self.post_attention_end_events is not None:
                self.post_attention_end_events[layer_id][micro_batch_id].record(self.context.cur_stream)
            
            return res


    def qkv_offload(self, layer_id: int, micro_batch_id: int):
        """Offload QKV to qkv_pin"""
        with torch.cuda.stream(self.stream_engine.kv_cache_offload_stream):
            events = self.stream_engine._get_or_create_layer_events(layer_id)
            events['compute_event'][micro_batch_id].wait(self.stream_engine.kv_cache_offload_stream)
            batch = self.micro_batches[micro_batch_id]
            # TODO: check qkv shape
            if len(batch.gpu_seq_lens) > 0: # have gpu attention request
                src = batch.qkv[batch.gpu_attn_input_ids_size:, ...].view(len(batch.cpu_seq_lens), -1, batch.qkv.shape[-1]) # cpu request src
                dst = self.context.qkv_pin[micro_batch_id, :len(batch.cpu_seq_lens), :]
                src2 = batch.qkv[:batch.gpu_attn_input_ids_size, ...].view(len(batch.gpu_seq_lens), -1, batch.qkv.shape[-1]) # gpu request src
                dst2 = self.context.qkv_pin_gpu[micro_batch_id, :len(batch.gpu_seq_lens), :]
                batch.qkv_gpu = batch.qkv[:batch.gpu_attn_input_ids_size, ...]
                self.stream_engine.offload_qkv(layer_id, micro_batch_id, src, dst, src2, dst2)
            else:
                src = batch.qkv.view(batch.batch_size, -1, batch.qkv.shape[-1]) # (bs, query_num, hidden_size)
                dst = self.context.qkv_pin[micro_batch_id, :batch.batch_size, :]
                # print(src.shape, dst.shape)
                # print(batch.batch_size)
                # print(self.context.qkv_pin.shape)
                self.stream_engine.offload_qkv(layer_id, micro_batch_id, src, dst)
    
    def attention(self, layer_id: int, micro_batch_id: int):
        self.cpu_attention(layer_id, micro_batch_id)
        self.gpu_attention(layer_id, micro_batch_id)
    
    def gpu_attention(self, layer_id: int, micro_batch_id: int):
        self.gpu_attn_futures[micro_batch_id] = self.context.gpu_attn_executor.submit(self.gpu_attention_func, layer_id, micro_batch_id)
        # self.gpu_attention_func(layer_id, micro_batch_id)

    def gpu_attention_func(self, layer_id: int, micro_batch_id: int):
        """Run GPU Attention"""
        self.kv_cache_copy_futures[layer_id].result() # 等待kv cache到pin_memory中
        events = self.stream_engine._get_or_create_layer_events(layer_id)
        batch = self.micro_batches[micro_batch_id]
        if batch.policy.gpu_attention_ratio == 0 or len(batch.gpu_seq_lens_list) == 0:
            # events['gpu_attn_event'][micro_batch_id].record(self.context.cur_stream)
            return
        nano_batch_num = (len(batch.gpu_seq_lens_list) + batch.policy.gpu_attention_nano_batch_size - 1) // batch.policy.gpu_attention_nano_batch_size
        assert len(batch.gpu_attn_init_slot) == nano_batch_num, f"gpu_attn_init_slot length: {len(batch.gpu_attn_init_slot)}, nano_batch_num: {nano_batch_num}"
        attn_outputs = []
        # print("GPU attention seq length: ", batch.gpu_seq_lens)
        seq_len_cusum = torch.zeros(len(batch.gpu_seq_lens_list) + 1, dtype=torch.int32, device="cpu")
        seq_len_cusum[1:] = torch.cumsum(torch.tensor(batch.gpu_seq_lens_list, dtype=torch.int32, device="cpu"), dim=0)
        seq_len_cusum += self.kv_cache_pin_begin[micro_batch_id] # 加上每个micro-batch的kv cache在kv_cache_pin中的起始位置
        for j in range(nano_batch_num):
            begin = j * batch.policy.gpu_attention_nano_batch_size
            end = min(begin + batch.policy.gpu_attention_nano_batch_size, len(batch.gpu_seq_lens_list))
            # print(f"gpu_attention, layer_id: {layer_id}, micro_batch_id: {micro_batch_id}, j: {j}, begin: {begin}, end: {end}, len(batch.gpu_seq_lens): {len(batch.gpu_seq_lens)}")
            '''Load KV Cache'''
            with record_function("load_kv_cache"):
                if j == 0:
                    if layer_id == 0:
                        pass
                    else:
                        for k in range(len(self.stream_engine.layer_events[layer_id-1]["load_kv_cache_event"][micro_batch_id])):
                            self.stream_engine.layer_events[layer_id-1]["load_kv_cache_event"][micro_batch_id][k].wait(self.stream_engine.kv_cache_load_stream)
                else:
                    self.stream_engine.layer_events[layer_id]["load_kv_cache_event"][micro_batch_id][j-1].wait(self.stream_engine.kv_cache_load_stream)
                    # self.stream_engine.layer_events[layer_id]["load_kv_cache_event"][micro_batch_id][j-1].synchronize()
                # self.stream_engine.decode_load_kv_cache(True, layer_id, self.prefetch_kvcache_cpu_indicies, self.prefetch_kvcache_gpu_indicies, self.context.cpu_token_to_kv_pool_allocator.get_kvcache(), self.context.gpu_token_to_kv_pool_allocator.get_kvcache(), micro_batch_id, j, begin, end)
                self.stream_engine.decode_load_kv_cache2(True, layer_id, self.context.kv_cache_pin[layer_id % 2, ...], self.prefetch_kvcache_cpu_indicies, self.prefetch_kvcache_gpu_indicies, self.context.gpu_token_to_kv_pool_allocator.get_kvcache(), micro_batch_id, j, seq_len_cusum, begin, end)

            events["load_kv_cache_event"][micro_batch_id][j].synchronize() 
            events['compute_event'][micro_batch_id].synchronize() # 得等pre-attn算完
            '''GPU Attention'''
            with torch.cuda.stream(self.context.cur_stream):
                if batch.forward_mode != ForwardMode.TARGET_VERIFY:
                    batch.forward_mode = ForwardMode.DECODE
                batch.decode_part = DecodePart.CPU_ATTN

                # 同步
                events["load_kv_cache_event"][micro_batch_id][j].wait(self.context.cur_stream)
                # if j > 0:
                #     self.stream_engine.layer_events[layer_id]["load_kv_cache_event"][micro_batch_id][j-1].wait(self.context.cur_stream)

                attnoutput = self.model.forward(
                    input_ids=batch.input_ids,
                    positions=batch.positions, # core-attn并不需要position，pre-attn才需要position
                    forward_batch=batch,
                    hidden_states=batch.hidden_states,
                    residual=batch.residual,
                    cur_layers=[layer_id],
                    is_decode_gpu_attn=True,
                    micro_batch_id=micro_batch_id,
                    nano_batch_id=j,
                    gpu_attn_init_slot=batch.gpu_attn_init_slot[j],
                    gpu_attn_nano_batch_slice=(begin, end),
                )
                # attn_outputs.append(attnoutput)
                if self.gpu_attn_output[micro_batch_id] is None:
                    self.gpu_attn_output[micro_batch_id] = attnoutput
                else:
                    self.gpu_attn_output[micro_batch_id] = torch.cat([self.gpu_attn_output[micro_batch_id], attnoutput], dim=0)
                events['gpu_attn_event'][micro_batch_id][j].record(self.context.cur_stream)
            events['gpu_attn_event'][micro_batch_id][j].synchronize() # XXX: FOR DEBUG

        # for j in range(nano_batch_num):
        #     events['gpu_attn_event'][micro_batch_id][j].synchronize()
        
        # self.gpu_attn_output[micro_batch_id] = torch.cat(attn_outputs, dim=0)

    def cpu_attention(self, layer_id: int, micro_batch_id: int):
        """Run CPU Attention"""
        self.cpu_attn_futures[micro_batch_id] = self.context.cpu_executor.submit(self.cpu_attetion_func, layer_id, micro_batch_id)

    def cpu_attetion_func(self, layer_id: int, micro_batch_id: int):
        """Run CPU Attention"""
        self.cpu_time_record[micro_batch_id] = time.time()
        with record_function("cpu_attention"):
            batch = self.micro_batches[micro_batch_id]
            if batch.forward_mode != ForwardMode.TARGET_VERIFY:
                batch.forward_mode = ForwardMode.DECODE
            batch.decode_part = DecodePart.CPU_ATTN

            batch.qkv_pin = self.context.qkv_pin[micro_batch_id, :len(batch.cpu_seq_lens), :]
            if len(batch.gpu_seq_lens) > 0:
                batch.qkv_pin_gpu = self.context.qkv_pin_gpu[micro_batch_id, :len(batch.gpu_seq_lens), :]
            # TODO: 需要修改hidden_pin的大小
            batch.hidden_pin = self.context.hidden_pin[micro_batch_id, :len(batch.cpu_seq_lens), :]

            events = self.stream_engine._get_or_create_layer_events(layer_id)
            events["kv_offload_event"][micro_batch_id].synchronize()
            self.model.forward(
                input_ids=batch.input_ids,
                positions=batch.positions, # core-attn并不需要position，pre-attn才需要position
                forward_batch=batch,
                hidden_states=batch.hidden_states,
                residual=batch.residual,
                cur_layers=[layer_id],
                micro_batch_id=micro_batch_id,
            )
            # 记录计算完成事件
            # self.stream_engine.record_compute_done(layer_id, micro_batch_id)
        self.performance_recoder["cpu_attention_time"] += time.time() - self.cpu_time_record[micro_batch_id]
    
    def prefetch_kv_cache_to_pin(self, layer_id: int):
        self.kv_cache_copy_futures[layer_id] = self.context.kv_cache_copy_executor.submit(self.prefetch_kv_cache_to_pin_func, layer_id)
    
    def prefetch_kv_cache_to_pin_func(self, layer_id: int):
        if self.server_args.target_cg_nano_kv_cache_slot <= 1:
            return
        if layer_id >= 2:
            for j in range(len(self.micro_batches)):
                self.stream_engine.layer_events[layer_id - 2]["compute_event"][j].synchronize()
        # self.kv_cache_pin_begin = [] # 每个micro-batch的kv cache在kv_cache_pin中的起始位置
        pt = 0
        src_indices = self.prefetch_kvcache_cpu_indicies
        for micro_batch_id in range(len(self.micro_batches)):
            batch = self.micro_batches[micro_batch_id]
            # self.kv_cache_pin_begin.append(pt)
            for j in range(len(batch.gpu_seq_lens)):
                seq_len = src_indices[micro_batch_id][j][1] - src_indices[micro_batch_id][j][0]
                src = self.context.cpu_token_to_kv_pool_allocator.get_kvcache().get_kv_buffer(layer_id)[:, slice(src_indices[micro_batch_id][j][0], src_indices[micro_batch_id][j][1]), ...] # 读取此request的kv cache
                self.context.kv_cache_pin[layer_id % 2, pt:pt+seq_len, ...] = src.transpose(0, 1).contiguous()
                pt += seq_len
        # print("self.kv_cache_pin is pinned?: ", self.context.kv_cache_pin.is_pinned())

    def forward_layer_prefill(self, layer_id: int):
        """执行单个layer的前向计算，处理所有micro-batch"""
        logger.debug(f"开始处理第 {layer_id} 层的前向计算")

        # # 1. 首先预取并加载当前layer的权重参数（按专家分块）

        # 计算CPU端专家数量
        num_cpu_experts = self.model_config.num_local_experts - int(self.model_config.num_local_experts * self.policy.weight_cache_ratio)

        # 2. 等待权重块加载完成
        self.stream_engine.wait_for_layer_load(layer_id, stage="prefill")

        # 3. 遍历所有micro-batch
        outputs = []
        self.prefetch_experts((layer_id + 1) % self.model_config.num_hidden_layers, "prefill")
        for micro_batch_id in range(len(self.micro_batches)):
            batch = self.micro_batches[micro_batch_id]
            logger.debug(f"处理第 {layer_id} 层, micro-batch {micro_batch_id}")

            self.stream_engine.wait_for_last_kvcache_offload_prefill(layer_id, micro_batch_id)
            self.stream_engine.wait_for_hidden_load_prefill(layer_id, micro_batch_id)

            self.stream_engine.hidden_offload_stream.synchronize() # XXX: 加上这个同步，用于防止计算出错，出错原因还不清楚。
            
            # 3.1 如果有预加载的hidden states，先等待加载完成
            # if hasattr(batch, 'hidden_states_cpu') and batch.hidden_states_cpu is not None:
            # if layer_id > 0:  # 第一个layer的
            # self.prefill_hidden_load(layer_id, micro_batch_id)
            # 这里不能阻塞等待
            # self.stream_engine.wait_for_layer_load(layer_id, num_chunks=num_cpu_experts)

            # 3.2 执行layer计算
            # print(f"micro_batch_id: {micro_batch_id}, layer_id: {layer_id}")
            if layer_id != self.model_config.num_hidden_layers - 1:
                batch.hidden_states, batch.residual = self.prefill(
                    batch, DecodePart.ALL, 
                    layer_id, self.experts_mapping[layer_id],
                    self.stream_engine.layer_events[layer_id]['attn_event'][micro_batch_id], 
                    micro_batch_id,
                )
                # outputs.append((batch.hidden_states, batch.residual))
                outputs.append(None)
            else:
                logits_processor_output = self.prefill(
                    batch, DecodePart.ALL, 
                    layer_id, self.experts_mapping[layer_id],
                    self.stream_engine.layer_events[layer_id]['attn_event'][micro_batch_id],
                    micro_batch_id,
                )
                outputs.append(logits_processor_output)


            # 记录计算完成事件
            self.stream_engine.record_compute_done(layer_id, micro_batch_id)

            # 3.3 offload KV cache
            self.prefill_kv_cache_offload(layer_id, micro_batch_id, "prefill")

            # hidden offload策略1：逐层offload，显存中会保留所有micro-batch的hidden states
            # if layer_id < self.model_config.num_hidden_layers - 1 and len(self.micro_batches) > 1:
            #     self.prefill_hidden_offload(layer_id, micro_batch_id)
            #     self.prefill_hidden_load(layer_id + 1, (micro_batch_id + 1) % len(self.micro_batches))


            # hidden offload策略2：逐micro-batch offload，现存中只会保留一个micro-batch的hidden state
            if len(self.micro_batches) > 1:
                if micro_batch_id == len(self.micro_batches) - 1:
                    if layer_id < self.model_config.num_hidden_layers - 1:
                        self.prefill_hidden_load(layer_id + 1, 0)
                else:
                    self.prefill_hidden_load(layer_id, micro_batch_id + 1)
                torch.cuda.synchronize(self.stream_engine.compute_stream)
                self.prefill_hidden_offload(layer_id, micro_batch_id)
            
            if layer_id == self.model_config.num_hidden_layers - 1:
                torch.cuda.synchronize(self.stream_engine.compute_stream)
                self.prefill_hidden_offload(layer_id, micro_batch_id, outputs[-1])
                
            
        if layer_id == self.model_config.num_hidden_layers - 1:
            for mb in self.micro_batches:
                # del mb.hidden_states # we shouldn't delete hidden_state of the last layer here, since it will be used in draft model extend
                del mb.residual

        return outputs

    def prefill_forward(self, forward_micro_batches: List[ForwardBatch], policy: Policy) -> List[LogitsProcessorOutput]:
        """
        执行模型前向计算，处理所有层
        返回：List[LogitsProcessorOutput]
            LogitsProcessorOutput: 最后一层的logits输出, hidden_state_to_store
        """
        self.micro_batches = forward_micro_batches
        self.update_before_forward(forward_micro_batches)

        self.prefetch_experts(0, "prefill")

        # 创建进度条
        pbar = tqdm(
            total=self.model_config.num_hidden_layers,
            desc="Prefill Process",
            unit="layer",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

        # 逐层执行前向计算
        res = None
        for layer_id in range(self.model_config.num_hidden_layers):
            res = self.forward_layer_prefill(layer_id)
            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({"Current layer": f"{layer_id+1}/{self.model_config.num_hidden_layers}"})

        # 关闭进度条
        pbar.close()

        # 最终同步所有流
        self.stream_engine.synchronize_all()

        # 返回最后一层的输出
        return res

    def decode_forward(self, forward_micro_batches: List[ForwardBatch], policy: Policy) -> Tuple[List[LogitsProcessorOutput], Optional[dict]]:
        self.micro_batches = forward_micro_batches
        self.reset_performance_recoder(len(self.micro_batches))

        torch.cuda.synchronize()
        self.prefetch_kv_cache_to_pin(0)
        for k, ub in enumerate(self.micro_batches[:2]):
            self.pre_attention(0, k)
            self.qkv_offload(0, k)
            self.attention(0, k)

        # self.prefetch_experts_to_pin(1)
        
        res = []
        for layer_id in range(self.model_config.num_hidden_layers):
            # logger.debug(f"decode_forward layer {layer_id}")
            # self.prefetch_kv_cache(layer_id, policy)
            self.prefetch_experts((layer_id + 1) % self.model_config.num_hidden_layers, "decode")
            if layer_id < self.model_config.num_hidden_layers - 1:
                self.prefetch_kv_cache_to_pin(layer_id + 1)
            for micro_batch_id in range(len(self.micro_batches)):
                # logger.debug(f"decode_forward layer {layer_id}, micro_batch_id {micro_batch_id}")
                self.decode_load_hidden(layer_id, micro_batch_id)
                ret = self.post_attention(layer_id, micro_batch_id)
                if layer_id == self.model_config.num_hidden_layers - 1:
                    res.append(ret)

                if len(self.micro_batches) >= 2:
                    if micro_batch_id + 2 < len(self.micro_batches):
                        self.pre_attention(layer_id, micro_batch_id + 2)
                        self.qkv_offload(layer_id, micro_batch_id + 2)
                        self.attention(layer_id, micro_batch_id + 2)
                    elif micro_batch_id + 2 >= len(self.micro_batches) and layer_id < self.model_config.num_hidden_layers - 1:
                        self.pre_attention(layer_id + 1, (micro_batch_id + 2) % len(self.micro_batches))
                        self.qkv_offload(layer_id + 1, (micro_batch_id + 2) % len(self.micro_batches))
                        self.attention(layer_id + 1, (micro_batch_id + 2) % len(self.micro_batches))
                elif len(self.micro_batches) == 1:
                    if layer_id < self.model_config.num_hidden_layers - 1:
                        self.pre_attention(layer_id + 1, 0)
                        self.qkv_offload(layer_id + 1, 0)
                        self.attention(layer_id + 1, 0)
                
        
        # 统一计算所有post_attention的执行时间
        if self.post_attention_start_events is not None and self.post_attention_end_events is not None:
            total_post_attention_time = 0.0
            for layer_id in range(self.model_config.num_hidden_layers):
                for micro_batch_id in range(len(self.micro_batches)):
                    start_event = self.post_attention_start_events[layer_id][micro_batch_id]
                    end_event = self.post_attention_end_events[layer_id][micro_batch_id]
                    # 同步end event
                    end_event.synchronize()
                    # 计算时间并累加（转换为秒）
                    post_attention_time = start_event.elapsed_time(end_event) / 1000.0
                    total_post_attention_time += post_attention_time
            
            self.performance_recoder["post_attention_time"] = total_post_attention_time
        
        self.stream_engine.htod_transfer_engine.clear_condition(["weight_load_layer_" + str(layer_id) for layer_id in range(1, self.model_config.num_hidden_layers)])

                
        return res, self.performance_recoder


@dataclass
class ExecutionContext:
    model_config: ModelConfig
    policy: Policy
    gpu_token_to_kv_pool_allocator: TokenToKVPoolAllocatorGPU
    cpu_token_to_kv_pool_allocator: TokenToKVPoolAllocatorCPU
    gpu_req_to_token_pool: ReqToTokenPool
    cpu_req_to_token_pool: ReqToTokenPool

    # experts cache on gpu
    experts_cache: torch.Tensor

    # cpu pinned relay
    qkv_pin: torch.Tensor
    hidden_pin: torch.Tensor
    experts_pin: torch.Tensor # pin memory for experts is only used in the decode stage
    experts_pin_ptr: torch.Tensor # pointer to the pinned memory for experts, used when delete experts_pin

    # multi-thread executors and futures
    cpu_executor: ThreadPoolExecutor
    gpu_attn_executor: ThreadPoolExecutor
    copy_executor: ThreadPoolExecutor
    prefetch_executor: ThreadPoolExecutor
    kv_cache_copy_executor: ThreadPoolExecutor
    # prefetch_stream: torch.cuda.stream
    # offload_stream: torch.cuda.stream
    # load_stream: torch.cuda.stream
    cur_stream: torch.cuda.stream

    # kv cache pin
    kv_cache_pin: torch.Tensor = None

    # # meta data
    # avg_prompt_tokens: int
    # gen_len: int

    @classmethod
    def build_context(
        cls,
        model_config: ModelConfig,
        server_args: ServerArgs,
        policy: Policy,
        cpu_req_to_token_pool: ReqToTokenPool,
        gpu_req_to_token_pool: ReqToTokenPool,
        cpu_token_to_kv_pool_allocator: TokenToKVPoolAllocatorCPU,
        gpu_token_to_kv_pool_allocator: TokenToKVPoolAllocatorGPU,
    ):
        micro_bache_size = policy.micro_batch_size
        torch.set_default_dtype(torch.float16)

        # experts cache
        num_layers = model_config.num_hidden_layers
        num_experts_in_gpu = int(model_config.num_local_experts * policy.weight_cache_ratio)
        experts_pool_size  = num_layers * num_experts_in_gpu + 2 * (model_config.num_local_experts - num_experts_in_gpu)
        intermediate_size = model_config.intermediate_size // server_args.tp_size

        experts_cache: torch.Tensor = torch.empty(experts_pool_size, 3 * intermediate_size * model_config.hidden_size, dtype=torch.get_default_dtype(), device="cuda")

        logger.info(f"expert cache size: {experts_cache.numel() * 2 / (1 << 30):.2f} GB")

        # experts_pin: torch.Tensor = torch.empty( model_config.num_local_experts - num_experts_in_gpu,
        #                                         3 * intermediate_size * model_config.hidden_size,
        #                                         dtype=torch.get_default_dtype(), device="cpu").pin_memory()
        element_size = torch.tensor([], dtype=torch.get_default_dtype()).element_size()
        experts_pin, experts_pin_ptr = make_pinned_tensor(size_in_bytes=(model_config.num_local_experts - num_experts_in_gpu) * 3 * intermediate_size * model_config.hidden_size * element_size, dtype=torch.get_default_dtype())
        experts_pin = experts_pin.view(model_config.num_local_experts - num_experts_in_gpu, 3 * intermediate_size * model_config.hidden_size)
        logger.info(f"experts pin size: {experts_pin.numel() * 2 / (1 << 30):.2f} GB")

        num_q_heads = model_config.num_attention_heads // server_args.tp_size
        n_kv_heads = model_config.num_key_value_heads // server_args.tp_size
        head_dim = model_config.hidden_size // model_config.num_attention_heads

        qkv_pin = torch.empty(policy.micro_batch_num, micro_bache_size, (num_q_heads + 2 * n_kv_heads) * head_dim, 
                              dtype=torch.get_default_dtype(), device="cpu").pin_memory()

        # XXX: 1 may changed in the speculative decoding
        hidden_pin = torch.empty(policy.micro_batch_num, micro_bache_size, 1, num_q_heads, head_dim,
                                 dtype=torch.get_default_dtype(), device="cpu").pin_memory()

        logger.info(f"qkv pin size: {qkv_pin.numel() * 2 / (1 << 30):.2f} GB")
        logger.info(f"hidden pin size: {hidden_pin.numel() * 2 / (1 << 30):.2f} GB")

        #  gpu memory usage
        free_gpu_memory, _ = torch.cuda.mem_get_info()
        logger.info(f"Free GPU memory: {free_gpu_memory / (1 << 30)}")

        # cpu workers
        cpu_executor = ThreadPoolExecutor(max_workers=1)
        gpu_attn_executor = ThreadPoolExecutor(max_workers=1)
        copy_executor = ThreadPoolExecutor(max_workers=1)
        prefetch_executor = ThreadPoolExecutor(max_workers=1)
        kv_cache_copy_executor = ThreadPoolExecutor(max_workers=1)

        # cuda events and streams
        prefetch_stream = torch.cuda.Stream() 
        offload_stream = torch.cuda.Stream()
        load_stream = torch.cuda.Stream()
        cur_stream = torch.cuda.current_stream()

        return cls(model_config=model_config,
                   policy=policy,
                   cpu_req_to_token_pool=cpu_req_to_token_pool,
                   gpu_req_to_token_pool=gpu_req_to_token_pool,
                   cpu_token_to_kv_pool_allocator=cpu_token_to_kv_pool_allocator,
                   gpu_token_to_kv_pool_allocator=gpu_token_to_kv_pool_allocator,
                   experts_cache=experts_cache,
                   qkv_pin=qkv_pin,
                   hidden_pin=hidden_pin,
                   experts_pin=experts_pin,
                   experts_pin_ptr=experts_pin_ptr,
                   cpu_executor=cpu_executor,
                   gpu_attn_executor=gpu_attn_executor,
                   copy_executor=copy_executor,
                   prefetch_executor=prefetch_executor,
                   kv_cache_copy_executor=kv_cache_copy_executor,
                #    prefetch_stream=prefetch_stream,
                #    offload_stream=offload_stream,
                #    load_stream=load_stream,
                   cur_stream=cur_stream
                   )

    def init_gpu_experts(self, model):
        num_gpu_experts = int(self.model_config.num_local_experts * self.policy.weight_cache_ratio)
        if num_gpu_experts != 0:
            for i in range(self.model_config.num_hidden_layers):
                self.experts_cache[i * num_gpu_experts: (i + 1) * num_gpu_experts].copy_(model.get_experts_cache(i)[:num_gpu_experts])

    def get_ecache_size(self):
        return self.experts_cache.shape[0]

    def delete_gpu_context(self):
        self.token_to_kv_pool.delete_gpu_cache()

    def change_expert_cache_size(self, decode_policy: Policy):
        if self.policy.weight_cache_ratio == decode_policy.weight_cache_ratio:
            self.policy = decode_policy
            return
        begin_time = time.time()
        self.policy = decode_policy
        num_layers = self.model_config.num_hidden_layers
        num_experts_in_gpu = int(self.model_config.num_local_experts * decode_policy.weight_cache_ratio)
        experts_pool_size  = num_layers * num_experts_in_gpu + 2 * (self.model_config.num_local_experts - num_experts_in_gpu)
        intermediate_size = self.model_config.intermediate_size // 1 # (self.server_args.tp_size)

        del self.experts_cache
        self.experts_cache: torch.Tensor = torch.empty(experts_pool_size, 3 * intermediate_size * self.model_config.hidden_size, dtype=torch.get_default_dtype(), device="cuda")

        logger.info(f"new expert_cache size: {self.experts_cache.numel() * 2 / (1 << 30):.2f} GB")

        free_pinned_tensor(self.experts_pin_ptr)
        element_size = torch.tensor([], dtype=torch.get_default_dtype()).element_size()
        self.experts_pin, self.experts_pin_ptr = make_pinned_tensor(size_in_bytes=(self.model_config.num_local_experts - num_experts_in_gpu) * 3 * intermediate_size * self.model_config.hidden_size * element_size, dtype=torch.get_default_dtype())
        self.experts_pin = self.experts_pin.view(self.model_config.num_local_experts - num_experts_in_gpu, 3 * intermediate_size * self.model_config.hidden_size)

        logger.info(f"new experts_pin size: {self.experts_pin.numel() * 2 / (1 << 30):.2f} GB")
        logger.info(f"Execution Engine: change expert cache size time: {time.time() - begin_time:.2f} s")

    def change_pin_size(self, decode_policy: Policy, server_args: ServerArgs):
        begin_time = time.time()
        num_q_heads = self.model_config.num_attention_heads // 1
        n_kv_heads = self.model_config.num_key_value_heads // 1
        head_dim = self.model_config.hidden_size // self.model_config.num_attention_heads

        gpu_req_num = decode_policy.gpu_attention_micro_batch_size

        token_num_per_req = 1
        token_num_per_req_gpu = -1

        if gpu_req_num != 0:
            token_num_per_req_gpu = 1

        if decode_policy.spec_policy == SpecPolicy.SequentialGPUonly:
            # 不需要分配pin memory，
            return
        elif decode_policy.spec_policy == SpecPolicy.SequentialCPUonly:
            # verify阶段每个request会有 speculative_num_draft_tokens个token
            token_num_per_req = server_args.speculative_num_draft_tokens
        elif decode_policy.spec_policy == SpecPolicy.NONE:
            pass
        elif decode_policy.spec_policy == SpecPolicy.SequentialCGCoop:
            # TODO: 应该还需要改，cpu request和gpu request的token num不一样
            token_num_per_req = server_args.speculative_num_draft_tokens
            token_num_per_req_gpu = server_args.speculative_num_draft_tokens_gpu
        else:
            assert False, f"spec policy: {decode_policy.spec_policy} not supported yet"

        del self.qkv_pin
        del self.hidden_pin

        # 根据speculative decoding的topk来确定qkv_pin和hidden_pin要分配多少个token的，之前默认是1
        if token_num_per_req_gpu == -1 or token_num_per_req_gpu is None:
            self.qkv_pin = torch.empty(decode_policy.micro_batch_num, decode_policy.micro_batch_size, token_num_per_req, (num_q_heads + 2 * n_kv_heads) * head_dim, dtype=torch.get_default_dtype(), device="cpu").pin_memory()
            self.hidden_pin = torch.empty(
                decode_policy.micro_batch_num,
                decode_policy.micro_batch_size,
                token_num_per_req,
                num_q_heads,
                head_dim,
                dtype=torch.get_default_dtype(),
                device="cpu",
            ).pin_memory()
        else: # gpu request may have different draft token num
            self.qkv_pin = torch.empty(decode_policy.micro_batch_num, decode_policy.micro_batch_size - gpu_req_num, token_num_per_req, (num_q_heads + 2 * n_kv_heads) * head_dim, dtype=torch.get_default_dtype(), device="cpu").pin_memory()
            self.hidden_pin = torch.empty(
                decode_policy.micro_batch_num,
                decode_policy.micro_batch_size - gpu_req_num,
                token_num_per_req,
                num_q_heads,
                head_dim,
                dtype=torch.get_default_dtype(),
                device="cpu",
            ).pin_memory()
            self.qkv_pin_gpu = torch.empty(decode_policy.micro_batch_num, gpu_req_num, token_num_per_req_gpu, (num_q_heads + 2 * n_kv_heads) * head_dim, dtype=torch.get_default_dtype(), device="cpu").pin_memory()

        if self.experts_pin.shape[0] != self.model_config.num_local_experts - int(self.model_config.num_local_experts * decode_policy.weight_cache_ratio):
            del self.experts_pin
            self.experts_pin = torch.empty(
                self.model_config.num_local_experts - int(self.model_config.num_local_experts * decode_policy.weight_cache_ratio),
                3 * self.model_config.intermediate_size * self.model_config.hidden_size,
                dtype=torch.get_default_dtype(),
                device="cpu",
            ).pin_memory()
        logger.info(f"Execution Engine: change pin size time: {time.time() - begin_time:.2f} s")

    def init_kv_cache_pin(self, decode_policy: Policy, server_args: ServerArgs):
        '''
            kv_cache_pin，存储一个micro-batch的kv cache，将其连续排布起来，然后一起传输给GPU。
            要有2层，一层用于当前micro-batch，一层用于prefetch下一个micro-batch。
            XXX: 以micro-batch为单位还是以layer为单位呢？
        '''
        head_num = self.model_config.num_key_value_heads
        head_dim = self.model_config.head_dim
        # XXX: 为保证kv_cache_pin的连续性，需要保证KV维度在后面。因为在传输时需要kv slots维度进行切分，有可能导致tensor不连续
        self.kv_cache_pin = torch.empty(
            2, # 2 micro-batch (or layer)
            decode_policy.gpu_attention_micro_batch_size * decode_policy.micro_batch_num * server_args.max_seq_length, # maximum kv slots in a micro-batch (or a layer)
            2, # K and V
            head_num,
            head_dim,
            dtype=torch.get_default_dtype(),
            device="cpu",
        ).pin_memory()