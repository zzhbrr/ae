"""
GPU Attention算子实现 - 仅包含core attention部分
"""

import torch
import time
import torch.nn.functional as F
import math
from typing import Dict, Any, Tuple
import sys
sys.path.append("/home/zzh/codes/triton_cpu_test/test")
from specmoe.optimizer.performance_simulator.operators.base import BaseOperator
from sglang.srt.utils import get_bool_env_var, get_device_core_count
from sglang.srt.utils import get_available_gpu_memory
# from sglang.srt.layers.attention.triton_ops.decode_attention import (
#     decode_attention_fwd,
# )
# from sglang.srt.layers.attention.triton_ops.extend_attention import ( 
#     extend_attention_fwd,
# )
from specmoe.layers.attention_backend.triton_ops.decode_attention import (
    decode_attention_fwd,
)
from specmoe.layers.attention_backend.triton_ops.extend_attention import ( 
    extend_attention_fwd,
)
import triton
import triton.language as tl

class GPUTritonAttentionOperator(BaseOperator):
    """GPU Triton Attention算子实现"""
    
    def __init__(self, num_heads: int = 32, kv_heads: int = 8, head_dim: int = 128):
        super().__init__("GPU_Triton_Attention")
        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.triton_attention_num_kv_splits = 8
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, kcache: torch.Tensor, vcache: torch.Tensor, qo_indptr: torch.Tensor, kv_indptr: torch.Tensor, kv_indices: torch.Tensor, attn_logits, attn_lse, num_kv_splits, causal, custom_mask, mask_indptr, max_extend_len, **kwargs) -> torch.Tensor:
        # print("q shape:", q.shape)
        # # print("k shape:", k.shape)
        # # print("v shape:", v.shape)
        # print("o shape:", o.shape)
        # print("kcache shape:", kcache.shape)
        # print("vcache shape:", vcache.shape)
        # print("qo_indptr:", qo_indptr)
        # print("kv_indptr:", kv_indptr)  
        # print("kv_indices:", kv_indices)
        # print("attn_logits shape:", attn_logits.shape)
        # print("attn_lse shape:", attn_lse.shape)
        # print("num_kv_splits:", num_kv_splits)
        # print("triton_attention_num_kv_splits:", self.triton_attention_num_kv_splits)
        # print("scale:", self.scale)
        # print("custom_mask:", custom_mask)
        # print("causal:", causal)
        # print("mask_indptr:", mask_indptr)
        # print("max_extend_len:", max_extend_len)

        if q.shape[1] == 1: # decoding
            decode_attention_fwd(
                q.view(-1, self.num_heads, self.head_dim),
                kcache,
                vcache,
                o.view(-1, self.num_heads, self.head_dim),
                kv_indptr,
                kv_indices,
                attn_logits,
                attn_lse,
                num_kv_splits,
                self.triton_attention_num_kv_splits,
                self.scale,
            )
        else: # extend
            extend_attention_fwd(
                q.view(-1, self.num_heads, self.head_dim), 
                k.view(-1, self.kv_heads, self.head_dim).contiguous(),
                v.view(-1, self.kv_heads, self.head_dim).contiguous(),
                o.view(-1, self.num_heads, self.head_dim),
                kcache,
                vcache,
                qo_indptr,
                kv_indptr,
                kv_indices,
                custom_mask,
                causal,
                mask_indptr,
                max_extend_len,
                self.scale,
                0.0
            )
        
        return o
    
    def get_input_shapes(self, batch_size: int = 8, seq_length: int = 2048, 
                        query_length: int = None, **kwargs) -> Dict[str, Tuple]:
        """获取输入张量形状"""
        if query_length is None:
            query_length = seq_length
            
        return {
            'query': (batch_size, self.num_heads, query_length, self.head_dim),
            'key': (batch_size, self.num_heads, seq_length, self.head_dim),
            'value': (batch_size, self.num_heads, seq_length, self.head_dim)
        }
    
    def is_input_shape_valid(self, batch_size: int = 8, seq_length: int = 2048, 
                        query_length: int = None, **kwargs) -> bool:
        # get gpu memory
        gpu_memory = get_available_gpu_memory('cuda', 0)
        # print("IN GPU TRITON OPERATOR, gpu_memory:", gpu_memory)
        # print("kv cache size: ", 2 * batch_size * seq_length * self.head_dim * self.kv_heads * 2 / 1024**3)
        if 2 * batch_size * seq_length * self.head_dim * self.kv_heads * 2 / 1024**3 > gpu_memory - 1:
            # print("kv cache size is too large, return False")
            return False
        if batch_size * seq_length >= 1200*2000:
            return False
        return True
    
    def get_flops(self, batch_size: int = 8, seq_length: int = 2048, 
                  query_length: int = None, **kwargs) -> int:
        """计算理论FLOPS"""
        if query_length is None:
            query_length = seq_length
            
        # QK^T计算: batch_size * num_heads * query_length * seq_length * head_dim * 2
        qk_flops = batch_size * self.num_heads * query_length * seq_length * self.head_dim * 2
        
        # Softmax计算 (简化估算): batch_size * num_heads * query_length * seq_length * 5
        # (exp, sum, div等操作)
        softmax_flops = batch_size * self.num_heads * query_length * seq_length * 5
        
        # Attention*V计算: batch_size * num_heads * query_length * seq_length * head_dim * 2  
        av_flops = batch_size * self.num_heads * query_length * seq_length * self.head_dim * 2
        
        return qk_flops + softmax_flops + av_flops
    
    def get_memory_usage(self, batch_size: int = 8, seq_length: int = 2048,
                        query_length: int = None, **kwargs) -> Dict[str, int]:
        """计算内存使用量"""
        if query_length is None:
            query_length = seq_length
            
        dtype_size = 4  # float32
        
        # 输入张量内存
        query_memory = batch_size * self.num_heads * query_length * self.head_dim * dtype_size
        key_memory = batch_size * self.num_heads * seq_length * self.head_dim * dtype_size
        value_memory = batch_size * self.num_heads * seq_length * self.head_dim * dtype_size
        
        # 中间计算结果内存
        scores_memory = batch_size * self.num_heads * query_length * seq_length * dtype_size
        
        # 输出内存
        output_memory = batch_size * self.num_heads * query_length * self.head_dim * dtype_size
        
        return {
            'query': query_memory,
            'key': key_memory,
            'value': value_memory,
            'scores': scores_memory,
            'output': output_memory,
            'total': query_memory + key_memory + value_memory + scores_memory + output_memory
        }
    
    def create_dummy_input(self, batch_size: int = 8, seq_length: int = 2048, 
                          query_length: int = None):
        """创建用于测试的虚拟输入"""
        if query_length is None:
            query_length = seq_length
            
        query = torch.randn(batch_size, query_length, self.num_heads, self.head_dim, dtype=torch.float16, device='cuda')
        key = torch.randn(batch_size, query_length, self.kv_heads, self.head_dim, dtype=torch.float16, device='cuda')
        value = torch.randn(batch_size, query_length, self.kv_heads, self.head_dim, dtype=torch.float16, device='cuda')
        seq_len_list = torch.tensor([seq_length] * batch_size, device='cuda').to(torch.int32)
        k_cache = torch.empty(sum(seq_len_list), self.kv_heads, self.head_dim, dtype=torch.float16, device="cuda")
        k_cache.uniform_(-1e-3, 1e-3)
        v_cache = torch.empty(sum(seq_len_list), self.kv_heads, self.head_dim, dtype=torch.float16, device="cuda")
        v_cache.uniform_(-1e-3, 1e-3)
        kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
        kv_indptr[1:] = torch.cumsum(seq_len_list, 0)
        kv_indices = torch.arange(sum(seq_len_list), dtype=torch.int32, device="cuda")
        attn_logits = torch.empty(batch_size, self.num_heads, self.triton_attention_num_kv_splits, self.head_dim, dtype=torch.float32, device="cuda")
        attn_lse = torch.empty(batch_size, self.num_heads, self.triton_attention_num_kv_splits, dtype=torch.float32, device="cuda")
        
        if query_length == 1:
            qo_indptr = None
            causal = None
            custom_mask = None
            mask_indptr = None
            max_extend_len = None
            num_kv_splits = torch.empty((batch_size,), dtype=torch.int32, device="cuda")
            # num_kv_splits.fill_(self.triton_attention_num_kv_splits)
            self.get_num_kv_splits(num_kv_splits, seq_len_list)
        else:
            num_kv_splits = None
            query_len_list = torch.tensor([query_length] * batch_size, device='cuda').to(torch.int32)
            qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
            qo_indptr[1:] = torch.cumsum(query_len_list, 0)
            causal = True
            custom_mask = None
            mask_indptr = None
            max_extend_len = query_length
        
        output = torch.empty(batch_size, query_length, self.num_heads, self.head_dim, device='cuda')

        return query, key, value, output, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, attn_logits, attn_lse, num_kv_splits, causal, custom_mask, mask_indptr, max_extend_len
    
    def get_num_kv_splits(
        self,
        num_kv_splits: torch.Tensor,
        seq_lens: torch.Tensor,
    ):
        num_token, num_seq = num_kv_splits.shape[0], seq_lens.shape[0]
        num_group = num_token // num_seq

        assert (
            num_group * num_seq == num_token
        ), f"num_seq({num_seq}), num_token({num_token}), something goes wrong!"

        if num_seq < 256:
            SCHEDULE_SEQ = 256
        else:
            SCHEDULE_SEQ = triton.next_power_of_2(num_seq)

        get_num_kv_splits_triton[(1,)](
            num_kv_splits,
            seq_lens,
            num_seq,
            num_group,
            self.num_heads,
            self.kv_heads,
            self.triton_attention_num_kv_splits,
            get_device_core_count(0),
            MAX_NUM_SEQ=SCHEDULE_SEQ,
        )

@triton.jit
def get_num_kv_splits_triton(
    num_kv_splits_ptr,
    seq_lens_ptr,
    num_seq,
    num_group,
    num_head,
    num_kv_head,
    max_kv_splits,
    device_core_count,
    MAX_NUM_SEQ: tl.constexpr,
):
    # TODO: this method is tunable, we need more online serving data to tune it
    offs_seq = tl.arange(0, MAX_NUM_SEQ)
    mask_seq = offs_seq < num_seq

    seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=0)
    max_seq_len = tl.max(seq_lens)
    seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=max_seq_len)
    min_seq_len = tl.min(seq_lens)
    if max_seq_len * 8 < min_seq_len * 10:
        min_seq_len = max_seq_len
    max_kv_splits_1 = tl.minimum(tl.cdiv(max_seq_len, min_seq_len), max_kv_splits)
    kv_chunk_size_1 = tl.cdiv(max_seq_len, max_kv_splits_1)

    # NOTE: this is a hack to let num_kv_split grows up with seqlen gradually
    ext_seq_len = tl.cast(max_seq_len, tl.float32) / 64.0
    ext_device_core_count = tl.cast(
        device_core_count * tl.maximum(tl.log2(ext_seq_len), 1.0), tl.int32
    )
    block_h, num_kv_group = 16, num_head // num_kv_head
    if num_kv_group == 1:
        token_grid = num_seq * num_group * num_head
    else:
        # from triton_ops/decode_attention.py:_decode_grouped_att_m_fwd
        block_h = tl.minimum(block_h, num_kv_group)
        token_grid = num_seq * num_group * tl.cdiv(num_head, block_h)
    max_kv_splits_2 = tl.minimum(
        tl.cdiv(ext_device_core_count, token_grid), max_kv_splits
    )
    kv_chunk_size_2 = tl.cdiv(max_seq_len, max_kv_splits_2)

    num_kv_splits = tl.maximum(
        tl.cdiv(seq_lens, kv_chunk_size_1), tl.cdiv(seq_lens, kv_chunk_size_2)
    )

    offs_token = offs_seq * num_group
    mask_token = offs_token < num_seq * num_group
    for i in range(0, num_group):
        tl.store(num_kv_splits_ptr + i + offs_token, num_kv_splits, mask=mask_token)

if __name__ == "__main__":
    operator = GPUTritonAttentionOperator(num_heads=32, kv_heads=32, head_dim=128)
    input_data = operator.create_dummy_input(batch_size=5000, seq_length=128, query_length=1)
    operator.forward(*input_data)

# python specmoe/optimizer/performance_simulator/operators/gpu_triton_attention.py