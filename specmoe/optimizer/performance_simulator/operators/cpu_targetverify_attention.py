"""
CPU Attention算子实现 - 仅包含core attention部分
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
import math
import time
from typing import Dict, Any, Tuple
from specmoe.optimizer.performance_simulator.operators.base import BaseOperator
from specmoe._cpu_kernel import token_attention_cpu_verified3
import matplotlib.pyplot as plt

class CPUTargetVerifyAttentionOperator(BaseOperator):
    """CPU Attention算子实现 - 只计算softmax(QK)V部分"""
    
    def __init__(self, num_heads: int = 32, head_dim: int = 128, kv_heads: int = 8, fast_preparation: bool = True, max_seq_len: int = 2048, max_batchsize: int = 3000, max_q_len: int = 5):
        super().__init__("CPU_TargetVerify_Attention") 
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.kv_heads = kv_heads

        self.fast_preparation = fast_preparation
        # for fast data preparation, 6 minutes allocate 80GB memory
        if fast_preparation:
            MAX_SEQ_LEN = max_seq_len
            MAX_BATCHSIZE = max_batchsize
            MAX_Q_LEN = max_q_len
            self.kcache = torch.empty(MAX_SEQ_LEN*MAX_BATCHSIZE, self.kv_heads, self.head_dim, dtype=torch.float16, device="cpu")
            self.kcache.uniform_(-1e-3, 1e-3)
            self.vcache = torch.empty(MAX_SEQ_LEN*MAX_BATCHSIZE, self.kv_heads, self.head_dim, dtype=torch.float16, device="cpu")
            self.vcache.uniform_(-1e-3, 1e-3)
            self.start_loc_prepared = torch.zeros(MAX_BATCHSIZE+1, dtype=torch.int64, device="cpu")
            self.start_loc_prepared[1:] = torch.cumsum(torch.tensor([MAX_SEQ_LEN]*MAX_BATCHSIZE), 0)
            self.custom_mask_prepared = torch.zeros(MAX_SEQ_LEN*MAX_BATCHSIZE*MAX_Q_LEN, dtype=torch.float32, device="cpu")
        
    def forward(self, query: torch.Tensor, output: torch.Tensor, kcache: torch.Tensor, vcache: torch.Tensor, seq_lens:torch.Tensor, start_loc: torch.Tensor, custom_mask:torch.Tensor, mask_indptr:torch.Tensor, **kwargs) -> torch.Tensor:
        """
        CPU Core Attention计算: softmax(QK^T/sqrt(d))V
        Args:
            query: [batch_size, query_length, num_heads, head_dim]
        """
        token_attention_cpu_verified3(
            output, query, kcache, vcache, seq_lens, start_loc, custom_mask, mask_indptr, self.scale
        )
        
        return output
    
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
    
    def get_flops(self, batch_size: int = 8, seq_length: int = 2048,
                  query_length: int = None, **kwargs) -> int:
        """计算理论FLOPS"""
        if query_length is None:
            query_length = seq_length
            
        # QK^T计算
        qk_flops = batch_size * self.num_heads * query_length * seq_length * self.head_dim * 2
        
        # Softmax计算 
        softmax_flops = batch_size * self.num_heads * query_length * seq_length * 5
        
        # Attention*V计算
        av_flops = batch_size * self.num_heads * query_length * seq_length * self.head_dim * 2
        
        return qk_flops + softmax_flops + av_flops
    
    def get_memory_usage(self, batch_size: int = 8, seq_length: int = 2048,
                        query_length: int = None, **kwargs) -> Dict[str, int]:
        """计算内存使用量 - 考虑CPU缓存层次"""
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
        
        total_memory = query_memory + key_memory + value_memory + scores_memory + output_memory
        
        return {
            'query': query_memory,
            'key': key_memory,
            'value': value_memory,
            'scores': scores_memory,
            'output': output_memory,
            'total': total_memory,
        }
    
    def create_dummy_input(self, batch_size: int = 8, seq_length: int = 2048,
                          query_length: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """创建用于测试的虚拟输入"""
        if self.fast_preparation:
            query = torch.randn(batch_size, query_length, self.num_heads, self.head_dim, device='cpu', dtype=torch.float16)
            output = torch.randn(batch_size, query_length, self.num_heads, self.head_dim, device='cpu', dtype=torch.float16)
            seq_len = torch.tensor([seq_length] * batch_size, device='cpu').to(torch.int64)
            kcache = self.kcache
            vcache = self.vcache
            start_loc = self.start_loc_prepared[:batch_size + 1]
            seq_mask_len = query_length * seq_len
            custom_mask = self.custom_mask_prepared[:seq_mask_len.sum()]
            mask_indptr = torch.zeros(batch_size + 1, dtype=torch.int64, device="cpu")
            mask_indptr[1 : batch_size + 1] = torch.cumsum(seq_mask_len, dim=0)
            return query, output, kcache, vcache, seq_len, start_loc, custom_mask, mask_indptr
        else:
            query = torch.randn(batch_size, query_length, self.num_heads, self.head_dim, device='cpu', dtype=torch.float16)
            output = torch.randn(batch_size, query_length, self.num_heads, self.head_dim, device='cpu', dtype=torch.float16)
            seq_len = torch.tensor([seq_length] * batch_size, device='cpu').to(torch.int64)
            kcache = torch.empty(sum(seq_len), self.kv_heads, self.head_dim, dtype=torch.float16, device="cpu")
            kcache.uniform_(-1e-3, 1e-3)
            vcache = torch.empty(sum(seq_len), self.kv_heads, self.head_dim, dtype=torch.float16, device="cpu")
            vcache.uniform_(-1e-3, 1e-3)
            start_loc = torch.zeros(batch_size+1, dtype=torch.int64, device="cpu")
            start_loc[1:] = torch.cumsum(seq_len, 0)
            seq_mask_len = query_length * seq_len
            custom_mask = torch.zeros(seq_mask_len.sum(), dtype=torch.float32, device="cpu")
            mask_indptr = torch.zeros(batch_size + 1, dtype=torch.int64, device="cpu")
            mask_indptr[1 : batch_size + 1] = torch.cumsum(seq_mask_len, dim=0)
            
            return query, output, kcache, vcache, seq_len, start_loc, custom_mask, mask_indptr

    

if __name__ == "__main__":
    operator = CPUTargetVerifyAttentionOperator(num_heads=32, head_dim=128, kv_heads=8, fast_preparation=True)
    ql_list = []
    time_list = []
    for ql in range(1, 40):
        inputs = operator.create_dummy_input(batch_size=1000, seq_length=1024, query_length=ql)
        output = operator.forward(*inputs)
        time_start = time.time()
        for i in range(3):
            output = operator.forward(*inputs)
        time_end = time.time()
        print(f"query_length: {ql}, output shape: {output.shape}")
        print(f"time: {(time_end - time_start) / 3}")
        ql_list.append(ql)
        time_list.append((time_end - time_start) / 3)
    import numpy as np
    plt.clf()
    plt.figure(figsize=(8, 5))
    plt.plot(ql_list, time_list, marker='o')
    plt.xlabel("query_length")
    plt.ylabel("time (s)")
    plt.title("CPU TargetVerify Attention time")
    plt.xticks(np.arange(0, max(ql_list)+1, 1))
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("cpu_targetverify_attention_time.png")
    plt.show()

