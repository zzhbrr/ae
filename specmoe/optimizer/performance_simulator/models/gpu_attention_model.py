"""
GPU Attention性能建模
"""

import numpy as np
import math
from typing import Dict, Any, List
from specmoe.optimizer.performance_simulator.models.base import BasePerformanceModel


class GPUAttentionModel(BasePerformanceModel):
    """
    GPU Attention性能模型 (仅Core Attention部分)
    """
    
    def __init__(self, num_heads: int = 32, head_dim: int = 128):
        super().__init__("GPU_Attention")
        self.num_heads = num_heads
        self.head_dim = head_dim
        
    def predict(self, batch_size: int, seq_length: int, 
               query_length: int = None, **kwargs) -> float:
        """
        预测GPU Attention执行时间
        
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
        
        compute_term = self.parameters['alpha1'] * B * H * Q * S * D
        
        memory_term = self.parameters['alpha2'] * B * H * Q * S
        
        softmax_term = self.parameters['beta'] * B * H * Q * S
        
        base_latency = self.parameters['gamma']
        
        predicted_time = compute_term + memory_term + softmax_term + base_latency
        
        return max(0, predicted_time)
    
    def get_parameter_names(self) -> List[str]:
        """获取需要拟合的参数名称"""
        return ['alpha1', 'alpha2', 'beta', 'gamma']
    
    def get_feature_vector(self, batch_size: int, seq_length: int,
                          query_length: int = None, **kwargs) -> np.ndarray:
        """
        构造特征向量用于线性拟合
       """
        if query_length is None:
            query_length = seq_length
            
        B = batch_size
        H = self.num_heads
        Q = query_length
        S = seq_length
        D = self.head_dim
        
        features = np.array([
            B * H * Q * S * D,  
            B * H * Q * S,      
            B * H * Q * S,      
            1                   
        ])
        
        return features
    
    
    def validate_input(self, batch_size: int, seq_length: int, 
                      query_length: int = None, **kwargs) -> bool:
        """验证输入参数"""
        if query_length is None:
            query_length = seq_length
        return (batch_size > 0 and seq_length > 0 and query_length > 0)