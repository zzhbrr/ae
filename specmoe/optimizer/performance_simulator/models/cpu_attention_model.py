"""
CPU Attention性能建模
"""

import numpy as np
import math
from typing import Dict, Any, List
from specmoe.optimizer.performance_simulator.models.base import BasePerformanceModel


class CPUAttentionModel(BasePerformanceModel):
    """
    CPU Attention性能模型 (仅Core Attention部分)
    """
    
    def __init__(self, num_heads: int = 32, head_dim: int = 128, num_threads: int = None):
        super().__init__("CPU_Attention")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_threads = num_threads or 8  
        
        self.l1_cache_size = 32 * 1024     
        self.l2_cache_size = 256 * 1024    
        self.l3_cache_size = 8 * 1024 * 1024
        self.cache_line_size = 64          
        
    def predict(self, batch_size: int, seq_length: int,
               query_length: int = None, **kwargs) -> float:
        """
        预测CPU Attention执行时间
        
        Args:
            batch_size: 批大小
            seq_length: 序列长度
            query_length: 查询长度 (默认等于seq_length)
            
        Returns:
            预测执行时间 (秒)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未拟合参数")
            
        if query_length is None:
            query_length = seq_length
            
        B = batch_size
        H = self.num_heads
        Q = query_length  
        S = seq_length
        D = self.head_dim
        
        # 计算总运算量
        qk_ops = B * H * Q * S * D * 2      
        softmax_ops = B * H * Q * S * 5   
        av_ops = B * H * Q * S * D * 2   
        total_ops = qk_ops + softmax_ops + av_ops
        
        working_set_size = self._estimate_working_set(B, H, Q, S, D)
        cache_misses = self._estimate_cache_misses(working_set_size)
        
        sync_factor = math.log2(self.num_threads) if self.num_threads > 1 else 0
        
        compute_term = self.parameters['alpha1'] * total_ops
        
        cache_term = self.parameters['alpha2'] * cache_misses
        
        sync_term = self.parameters['beta'] * sync_factor * B * H
        
        base_latency = self.parameters['gamma']
        
        predicted_time = compute_term + cache_term + sync_term + base_latency
        
        return max(0, predicted_time)
    
    def _estimate_working_set(self, B: int, H: int, Q: int, S: int, D: int) -> int:
        input_size = B * H * (Q + 2 * S) * D * 4
        scores_size = B * H * Q * S * 4  
        output_size = B * H * Q * D * 4
        
        return input_size + scores_size + output_size
    
    
    def get_parameter_names(self) -> List[str]:
        """获取需要拟合的参数名称"""
        return ['alpha1', 'alpha2', 'beta', 'gamma']
    
    def get_feature_vector(self, batch_size: int, seq_length: int,
                          query_length: int = None, **kwargs) -> np.ndarray:
        """
        构造特征向量用于线性拟合
        
        Args:
            batch_size: 批大小
            seq_length: 序列长度
            query_length: 查询长度
            
        Returns:
            特征向量 [total_ops, cache_misses, sync_factor×B×H, 1]
        """
        if query_length is None:
            query_length = seq_length
            
        B = batch_size
        H = self.num_heads
        Q = query_length
        S = seq_length
        D = self.head_dim
        
        # 计算各项特征
        total_ops = B * H * Q * S * D * 4 + B * H * Q * S * 5 
        working_set_size = self._estimate_working_set(B, H, Q, S, D)
        cache_misses = self._estimate_cache_misses(working_set_size)
        sync_factor = math.log2(self.num_threads) if self.num_threads > 1 else 0
        
        features = np.array([
            total_ops,                     
            cache_misses,                  
            sync_factor * B * H,           
            1                              
        ])
        
        return features
    
    
    def validate_input(self, batch_size: int, seq_length: int,
                      query_length: int = None, **kwargs) -> bool:
        """验证输入参数"""
        if query_length is None:
            query_length = seq_length
        return (batch_size > 0 and seq_length > 0 and query_length > 0)