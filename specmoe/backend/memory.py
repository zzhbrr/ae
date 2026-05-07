"""Memory pool."""
import logging
import torch
from typing import List, Optional, Union, Tuple
import abc
import numpy as np

logger = logging.getLogger(__name__)

from sglang.srt.utils import debug_timing, get_compiler_backend

from specmoe.layers.attention import RadixAttention

GB = 1024 * 1024 * 1024

class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
    ):
        self.size = size
        self.max_context_len = max_context_len
        self.device = device
        self.req_to_token = torch.zeros(
            (size, max_context_len), dtype=torch.int32, device=device
        )
        self.free_slots = list(range(size))

    def write(self, indices, values):
        # logger.debug(f"indices: {indices}, values: {values.shape}")
        # logger.debug(f"req_to_token: {self.req_to_token.shape}")
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int) -> List[int]:
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    def free(self, free_index: Union[int, List[int]]):
        if isinstance(free_index, (int,)):
            self.free_slots.append(free_index)
        else:
            self.free_slots.extend(free_index)

    def clear(self):
        self.free_slots = list(range(self.size))

class KVCache(abc.ABC):

    def register_layer_transfer_counter(self, layer_transfer_counter):
        self.layer_transfer_counter = layer_transfer_counter
    
    @abc.abstractmethod
    def adjust_capacity(self, new_layer_num: int, new_size: int):
        raise NotImplementedError()


class TokenToKVPoolAllocatorGPU:
    """An allocator managing the indices to kv cache data."""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
    ):
        self.size = size
        self.dtype = dtype
        self.device = device
        self.page_size = 1

        self.free_slots = None
        self.is_not_in_free_group = True
        self.free_group = []
        self.clear()

        self._kvcache = kvcache

    def available_size(self):
        return len(self.free_slots)

    def adjust_capacity(self, new_layer_num: int, new_size: int):
        self.size = new_size
        self.clear()
        self._kvcache.adjust_capacity(new_layer_num, new_size)

    def get_kvcache(self):
        return self._kvcache

    def alloc(self, need_size: int):
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            # self.free_slots = torch.cat((self.free_slots, free_index))
            self.free_slots = torch.cat((free_index, self.free_slots))
        else:
            self.free_group.append(free_index)

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.cat(self.free_group))

    def backup_state(self):
        return self.free_slots.clone()

    def restore_state(self, free_slots):
        self.free_slots = free_slots

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_slots = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )
        self.is_not_in_free_group = True
        self.free_group = []
    
    def set_free_slots_force(self, free_slots: torch.Tensor):
        self.free_slots = free_slots

    def get_layer_kv_cache(self, layer_id: int, indices: torch.Tensor):
        return self._kvcache.get_layer_kv_cache(layer_id, indices)
    
class TokenToKVPoolAllocatorCPU:
    """An allocator managing the indices to kv cache data."""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        max_output_length: int,
        max_speculative_draft_tokens: int,
    ):
        self.size = size
        self.dtype = dtype
        self.device = device
        self.page_size = 1
        self.max_output_length = max_output_length
        self.max_speculative_draft_tokens = max_speculative_draft_tokens

        self.rid_to_kvcache_begin: dict[int, int] = {}
        self.rid_to_kvcache_end: dict[int, int] = {}
        self.rid_to_kvcache_now: dict[int, int] = {}
        
        self.allocated_now: int = 0

        self._kvcache = kvcache

    def available_size(self):
        return self.size - self.allocated_now

    def get_kvcache(self):
        return self._kvcache
    
    def alloc_for_a_new_request(self, rid: int, request_length:int): # only used before the prefill stage
        self.rid_to_kvcache_begin[rid] = self.allocated_now
        self.rid_to_kvcache_end[rid] = self.allocated_now + request_length + self.max_output_length + self.max_speculative_draft_tokens
        self.rid_to_kvcache_now[rid] = self.allocated_now + request_length
        self.allocated_now += request_length + self.max_output_length + self.max_speculative_draft_tokens
        # logger.debug(f"alloc {request_length} slots for request {rid}, begin: {self.rid_to_kvcache_begin[rid]}, end: {self.rid_to_kvcache_end[rid]}, now: {self.rid_to_kvcache_now[rid]}.")
        assert self.allocated_now <= self.size, "allocated_now is greater than size"
        # logger.debug(f"alloc {request_length} slots for request {rid}, begin: {self.rid_to_kvcache_begin[rid]}, end: {self.rid_to_kvcache_end[rid]}, now: {self.rid_to_kvcache_now[rid]}.")
        return [i for i in range(self.rid_to_kvcache_begin[rid], self.rid_to_kvcache_begin[rid] + request_length)]

    def alloc(self, rid: int, need_size: int):
        if need_size > self.available_size():
            return None
        if need_size > self.rid_to_kvcache_end[rid] - self.rid_to_kvcache_now[rid]:
            return None
        ret = [i for i in range(self.rid_to_kvcache_now[rid], self.rid_to_kvcache_now[rid] + need_size)]
        self.rid_to_kvcache_now[rid] += need_size
        # logger.debug(f"alloc {need_size} slots for request {rid}, return ret: {ret}.")
        return ret

    def backup_state(self):
        return (self.rid_to_kvcache_begin.copy(), self.rid_to_kvcache_now.copy(), self.rid_to_kvcache_end.copy(), self.allocated_now)

    def restore_state(self, states: Tuple[dict[int, int], dict[int, int], dict[int, int], int]):
        self.rid_to_kvcache_begin, self.rid_to_kvcache_now, self.rid_to_kvcache_end, self.allocated_now = states

    def clear(self):
        self.rid_to_kvcache_begin = {}
        self.rid_to_kvcache_now = {}
        self.rid_to_kvcache_end = {}
        self.allocated_now = 0
    
    def get_layer_kv_cache(self, layer_id: int, indices: torch.Tensor):
        return self._kvcache.get_layer_kv_cache(layer_id, indices)
    
    def get_start_loc(self, rids: List[int]):
        return [self.rid_to_kvcache_begin[rid] for rid in rids]
    
    def offload_kv_cache_prefill(self, layer_id: int, src: torch.Tensor, dst_indices: torch.Tensor, seq_lens: torch.Tensor, kv_cache = None):
        '''
        当用于draft model时，需要指定kv_cache，而不是用之前绑定的
        '''
        bs = seq_lens.shape[0]
        now = 0
        # logger.debug(f"src shape: {src.shape}, bs: {bs}, seq_lens: {seq_lens}")
        for i in range(bs):
            dst_start = dst_indices[now]
            dst_end = dst_start + seq_lens[i]
            if kv_cache is None:
                self._kvcache.kv_buffer[layer_id][:, dst_start:dst_end].copy_(src[:, now:now+seq_lens[i], :], non_blocking=True)
            else:
                kv_cache[layer_id][:, dst_start:dst_end].copy_(src[:, now:now+seq_lens[i], :], non_blocking=True)
            now += seq_lens[i]
    
    def get_tail_offload_indices(self, rids: List[int], offload_lens: List[int]):
        res = []
        bs = len(offload_lens)
        for i in range(bs):
            res.extend(list(range(self.rid_to_kvcache_now[rids[i]]-offload_lens[i], self.rid_to_kvcache_now[rids[i]])))
        return res
    
    def offload_kv_cache_new_verified(self, layer_id: int, src: torch.Tensor, dst_indices: torch.Tensor, offload_lens: List[int], kv_cache = None):
        '''
        用于将新验证的token的kv cache offload到GPU上
        '''
        bs = len(offload_lens)
        now = 0
        for i in range(bs):
            dst_start = dst_indices[now]
            dst_end = dst_start + offload_lens[i]
            if kv_cache is None:
                self._kvcache.kv_buffer[layer_id][:, dst_start:dst_end].copy_(src[:, now:now+offload_lens[i], :], non_blocking=True)
            else:
                kv_cache[layer_id][:, dst_start:dst_end].copy_(src[:, now:now+offload_lens[i], :], non_blocking=True)
            now += offload_lens[i]
    
    def compact_after_verify(self, rids:List[int], draft_token_num:int, accepted_cache_loc:torch.Tensor, draft_token_num_gpu:int = 0, gpu_req_num:int = 0) -> List[int]:
        '''
        用于将成功验证的token移到序列后面，并释放掉未成功验证的token slot，需要调整底层kv cache pool的排布.
        返回：new_cpu_out_cache_loc
        XXX: 这里假设所有request的draft token num都相同
        '''
        pt = 0
        new_cpu_out_cache_loc = []

        for i, rid in enumerate(rids):
            if i < gpu_req_num:
                first_loc_after_sequence = self.rid_to_kvcache_now[rid] - draft_token_num_gpu # 当前位置减去draft_token_num就是初始位置
            else:
                first_loc_after_sequence = self.rid_to_kvcache_now[rid] - draft_token_num # 当前位置减去draft_token_num就是初始位置
            while accepted_cache_loc[pt] < self.rid_to_kvcache_end[rid] and accepted_cache_loc[pt] >= self.rid_to_kvcache_begin[rid]: # pt指向的cache loc属于当前的req
                new_cpu_out_cache_loc.append(first_loc_after_sequence)
                for layer_id in range(self._kvcache.layer_num):
                    if first_loc_after_sequence != accepted_cache_loc[pt]:
                        self._kvcache.kv_buffer[layer_id][:, first_loc_after_sequence].copy_(self._kvcache.kv_buffer[layer_id][:, accepted_cache_loc[pt]], non_blocking=False) 
                        # XXX: 这里non-blocking=True会不会出问题，比如slot3正在向slot2拷贝数据，同时slot4正在向slot3拷贝数据，会不会出现错误的情况？
                first_loc_after_sequence += 1
                pt += 1
                if pt >= accepted_cache_loc.shape[0]:
                    break
            self.rid_to_kvcache_now[rid] = first_loc_after_sequence
            if pt >= accepted_cache_loc.shape[0]:
                break

        return new_cpu_out_cache_loc

    def compact_after_verify_optimized(self, rids:List[int], draft_token_num:int, accepted_cache_loc:torch.Tensor, draft_token_num_gpu:int = 0, gpu_req_num:int = 0) -> List[int]:
        '''
        优化版本的compact_after_verify，使用批量索引操作提升性能
        '''
        if len(accepted_cache_loc) == 0:
            return []
        
        # 预先计算所有的源索引和目标索引映射
        src_indices = []
        dst_indices = []
        new_cpu_out_cache_loc = []
        
        pt = 0
        for i, rid in enumerate(rids):
            if i < gpu_req_num:
                first_loc_after_sequence = self.rid_to_kvcache_now[rid] - draft_token_num_gpu
            else:
                first_loc_after_sequence = self.rid_to_kvcache_now[rid] - draft_token_num
            
            # 为当前request收集accepted tokens
            current_src_indices = []
            current_dst_indices = []
            
            while (pt < accepted_cache_loc.shape[0] and 
                   accepted_cache_loc[pt] < self.rid_to_kvcache_end[rid] and 
                   accepted_cache_loc[pt] >= self.rid_to_kvcache_begin[rid]):
                
                src_idx = accepted_cache_loc[pt].item()
                dst_idx = first_loc_after_sequence
                
                # 只有当源索引和目标索引不同时才需要拷贝
                if src_idx != dst_idx:
                    current_src_indices.append(src_idx)
                    current_dst_indices.append(dst_idx)
                
                new_cpu_out_cache_loc.append(dst_idx)
                first_loc_after_sequence += 1
                pt += 1
            
            # 将当前request的索引添加到全局列表
            src_indices.extend(current_src_indices)
            dst_indices.extend(current_dst_indices)
            
            # 更新request的当前位置
            self.rid_to_kvcache_now[rid] = first_loc_after_sequence
        
        # 批量执行KV cache的拷贝操作
        if src_indices:
            src_tensor = torch.tensor(src_indices, dtype=torch.long)
            dst_tensor = torch.tensor(dst_indices, dtype=torch.long)
            
            # 对所有layers执行批量索引操作
            for layer_id in range(self._kvcache.layer_num):
                # 使用torch的高级索引一次性完成所有拷贝
                self._kvcache.kv_buffer[layer_id][:, dst_tensor] = self._kvcache.kv_buffer[layer_id][:, src_tensor]
        
        return new_cpu_out_cache_loc

class MHATokenToKVPool(KVCache):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool = False,
        is_draft_model: bool = False,
        use_pinned_memory: bool = False,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype

        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self._create_buffers(use_pinned_memory=use_pinned_memory)

        self.layer_transfer_counter = None
        self.capture_mode = False
        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = self.device_module.Stream()

        kv_size = self.get_kv_size_bytes()
        logger.info(
            f"KV Cache is allocated. #tokens: {size}, KV size: {kv_size / GB:.2f} GB"
        )
    
    def adjust_capacity(self, new_layer_num: int, new_size: int, use_pinned_memory: bool = False):
        self.size = new_size
        self.layer_num = new_layer_num
        self._create_buffers(use_pinned_memory=use_pinned_memory)

    def _create_buffers(self, use_pinned_memory: bool = False):
        # [size, head_num, head_dim] for each layer
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        if hasattr(self, "kv_buffer") and self.kv_buffer is not None:
            del self.kv_buffer
        self.kv_buffer = [
            torch.zeros(
                (2, self.size + self.page_size, self.head_num, self.head_dim),
                dtype=self.store_dtype,
                device=self.device,
                pin_memory=(self.device == 'cpu' and use_pinned_memory)
            )
            for _ in range(self.layer_num)
        ]
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n is draft model: ", is_draft_model)
        # print("device: ", self.device)

    def _clear_buffers(self):
        del self.kv_buffer
    
    def get_layer_kv_cache(self, layer_id: int, indices: torch.Tensor):
        return self.kv_buffer[layer_id][:, indices]

    def get_kv_size_bytes(self):
        assert hasattr(self, "kv_buffer")
        k_size_bytes = 0
        for kv_cache in self.kv_buffer:
            k_size_bytes += np.prod(kv_cache.shape) * kv_cache.dtype.itemsize
        return k_size_bytes

    def get_key_buffer(self, layer_id: int, gpu_two_buffer: bool = False, two_buffer_offset: int = -1):
        if self.device == 'cuda':
            if gpu_two_buffer:
                layer_id = layer_id % 2
                if two_buffer_offset != -1:
                    layer_id = two_buffer_offset
            else:
                layer_id = 0
        return self.kv_buffer[layer_id][0]

    def get_value_buffer(self, layer_id: int, gpu_two_buffer: bool = False, two_buffer_offset: int = -1):
        if self.device == 'cuda':
            if gpu_two_buffer:
                layer_id = layer_id % 2
                if two_buffer_offset != -1:
                    layer_id = two_buffer_offset
            else:
                layer_id = 0
        return self.kv_buffer[layer_id][1]

    def get_kv_buffer(self, layer_id: int, gpu_two_buffer: bool = False):
        if self.device == 'cuda':
            if gpu_two_buffer:
                layer_id = layer_id % 2
            else:
                layer_id = 0
        return self.kv_buffer[layer_id]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        gpu_two_buffer: bool = False,
        two_buffer_offset: int = -1
    ):
        # 如果layer是int类型，则直接使用layer作为layer_id
        if isinstance(layer, int):
            layer_id = layer
        else:
            layer_id = layer.layer_id
        if self.device == 'cuda':
            if gpu_two_buffer:
                if two_buffer_offset == -1:
                    layer_id = layer_id % 2
                else:
                    layer_id = two_buffer_offset
            else:
                layer_id = 0
        
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        self.kv_buffer[layer_id][0][loc] = cache_k
        self.kv_buffer[layer_id][1][loc] = cache_v
    
    def load_kv_cache_from_cpu(self, layer_id: int, indices: torch.Tensor, src: torch.Tensor, non_blocking: bool = False, gpu_two_buffer: bool = False, two_buffer_offset: int = -1):
        '''
        用于加载draft model kv cache
        '''
        assert self.device == 'cuda'
        # logger.debug(f"load_kv_cache_from_cpu, layer_id: {layer_id}, indices: {indices}, src: {src.shape}, src.device: {src.device}")
        src = src.to(device='cuda', non_blocking=non_blocking) # XXX: 阻塞式传输
        self.set_kv_buffer(layer_id, indices, src[0], src[1], gpu_two_buffer=gpu_two_buffer, two_buffer_offset=two_buffer_offset)
        # logger.debug(f"kv_buffer: {self.kv_buffer[layer_id][:, indices]}")

    def load_kv_cache_from_cpu_in_batch(self, layer_id: int, indices_list: List[torch.Tensor], src_list: List[torch.Tensor], non_blocking: bool = False, gpu_two_buffer: bool = False, two_buffer_offset: int = -1, stream: torch.cuda.Stream = None, stream2: torch.cuda.Stream = None):
        '''
        用于加载target model kv cache
        '''
        assert self.device == 'cuda'
        assert two_buffer_offset != -1
        with torch.cuda.stream(stream):
            for i in range(len(indices_list)):
                self.kv_buffer[two_buffer_offset][0][indices_list[i]] = src_list[i][0].to(device='cuda', non_blocking=non_blocking)
                self.kv_buffer[two_buffer_offset][1][indices_list[i]] = src_list[i][1].to(device='cuda', non_blocking=non_blocking)

class MLATokenToKVPool(KVCache):
    pass