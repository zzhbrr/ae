"""
GPU Attention算子实现 - 仅包含core attention部分
"""

import torch
import torch.nn.functional as F
import math
from typing import Dict, Any, Tuple
from specmoe.optimizer.performance_simulator.operators.base import BaseOperator


class GPUAttentionOperator(BaseOperator):
    """GPU Attention算子实现 - 只计算softmax(QK)V部分"""
    
    def __init__(self, num_heads: int = 32, head_dim: int = 128):
        super().__init__("GPU_Attention")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Core Attention计算: softmax(QK^T/sqrt(d))V
        
        Args:
            query: [batch_size, num_heads, query_length, head_dim]
            key: [batch_size, num_heads, seq_length, head_dim]  
            value: [batch_size, num_heads, seq_length, head_dim]
            
        Returns:
            attention输出: [batch_size, num_heads, query_length, head_dim]
        """
        if torch.cuda.is_available():
            query = query.cuda()
            key = key.cuda()
            value = value.cuda()
        
        # 计算attention scores: QK^T
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        # [batch_size, num_heads, query_length, seq_length]
        
        # Softmax归一化
        attention_weights = F.softmax(scores, dim=-1)
        
        # 加权平均: Attention * V
        output = torch.matmul(attention_weights, value)
        # [batch_size, num_heads, query_length, head_dim]
        
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
                          query_length: int = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """创建用于测试的虚拟输入"""
        if query_length is None:
            query_length = seq_length
            
        query = torch.randn(batch_size, self.num_heads, query_length, self.head_dim)
        key = torch.randn(batch_size, self.num_heads, seq_length, self.head_dim)
        value = torch.randn(batch_size, self.num_heads, seq_length, self.head_dim)
        
        return query, key, value