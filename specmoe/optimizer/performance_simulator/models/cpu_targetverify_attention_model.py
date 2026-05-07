"""
CPU Attention性能建模
"""

import numpy as np
from typing import Dict, Any, List
from specmoe.optimizer.performance_simulator.models.base import BasePerformanceModel


class CPUTargetVerifyAttentionModel(BasePerformanceModel):
    """
    CPU Attention性能模型 (Core Attention部分)
    """
    
    def __init__(self, num_heads: int = 32, head_dim: int = 128, kv_heads: int = 8):
        super().__init__("CPU_TargetVerify_Attention")
        self.num_heads = num_heads  # N_q
        self.head_dim = head_dim    # d
        self.kv_heads = kv_heads    # N_kv
        self.gqa_groups = num_heads // kv_heads  # G = N_q / N_kv
        
    def predict(self, batch_size: int, seq_length: int,
               query_length: int = None, **kwargs) -> float:
        """
        预测CPU Attention执行时间
        
        Args:
            batch_size: 批大小 (B)
            seq_length: Key/Value序列长度 (S_kv)
            query_length: 查询长度 (S_q, 默认等于seq_length)
            
        Returns:
            预测执行时间 (秒)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未拟合参数")
            
        if query_length is None:
            query_length = seq_length
            
        # 参数映射
        B = batch_size      # B
        S_q = query_length  # S_q: query length
        S_kv = seq_length   # S_kv: key/value length
        N_q = self.num_heads    # N_q: number of query heads
        N_kv = self.kv_heads    # N_kv: number of key/value heads
        d = self.head_dim       # d: head dimension
        
        # 计算三个核心特征项
        # X₁: GEMM计算特征 (Q·K^T 和 P·V 矩阵乘法)
        gemm_feature = B * S_q * N_q * S_kv * d
        
        # X₂: Softmax元素操作特征 (attention score矩阵上的逐元素操作)
        softmax_feature = B * S_q * N_q * S_kv
        
        # X₃: 数据重排与访存特征 (Q-reorder和Output-write)
        memory_feature = B * S_q * N_q * d
        
        # 应用线性模型: T = C₁×X₁ + C₂×X₂ + C₃×X₃ + C₀
        predicted_time = (
            self.parameters['C1'] * gemm_feature +
            self.parameters['C2'] * softmax_feature +
            self.parameters['C3'] * memory_feature +
            self.parameters['C0']
        )
        
        return max(0, predicted_time)
    
    def get_parameter_names(self) -> List[str]:
        """获取需要拟合的参数名称"""
        return ['C1', 'C2', 'C3', 'C0']
    
    def get_feature_vector(self, batch_size: int, seq_length: int,
                          query_length: int = None, **kwargs) -> np.ndarray:
        """
        构造特征向量用于线性拟合
        """
        if query_length is None:
            query_length = seq_length
            
        # 参数映射
        B = batch_size      # B
        S_q = query_length  # S_q: query length
        S_kv = seq_length   # S_kv: key/value length
        N_q = self.num_heads    # N_q: number of query heads
        N_kv = self.kv_heads    # N_kv: number of key/value heads
        d = self.head_dim       # d: head dimension
        
        # 计算三个核心特征
        # X₁: GEMM计算特征 - 代表Q·K^T和P·V矩阵乘法的计算量
        gemm_feature = B * S_q * N_q * S_kv * d
        
        # X₂: Softmax元素操作特征 - 代表attention score矩阵上的逐元素操作
        softmax_feature = B * S_q * N_q * S_kv
        
        # X₃: 数据重排与访存特征 - 代表Q-reorder和Output-write的内存操作
        memory_feature = B * S_q * N_q * d
        
        features = np.array([
            gemm_feature,       # C₁项系数
            softmax_feature,    # C₂项系数  
            memory_feature,     # C₃项系数
            1                   # C₀项系数 (bias)
        ])
        
        return features
    