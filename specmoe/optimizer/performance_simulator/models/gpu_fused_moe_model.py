"""
GPU MoE性能建模
"""

import numpy as np
import math
from typing import Dict, Any, List
from specmoe.optimizer.performance_simulator.models.base import BasePerformanceModel


class GPUFusedMoEModel(BasePerformanceModel):
    """
    GPU MoE性能模型 - 简化的分段线性建模
    """
    
    def __init__(self, num_experts: int = 8, hidden_size: int = 4096, 
                 expert_size: int = 16384, top_k: int = 2):
        super().__init__("GPU_Fused_MoE_Piecewise")
        self.num_experts = num_experts
        self.hidden_size = hidden_size  
        self.expert_size = expert_size
        self.top_k = top_k
        
    def predict(self, num_tokens: int, **kwargs) -> float:
        """
        预测GPU MoE执行时间 - 使用简化的分段线性模型
        
        Args:
            num_tokens: 输入token数量
            
        Returns:
            预测执行时间 (秒)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未拟合参数")
            
        N = num_tokens
        breakpoint = self.parameters.get('breakpoint', 512)
        
        if N <= breakpoint:
            # 左段: T = a1 × N + b1
            predicted_time = self.parameters['a1'] * N + self.parameters['b1']
        else:
            # 右段: T = a2 × N + b2  
            predicted_time = self.parameters['a2'] * N + self.parameters['b2']
        
        return max(0, predicted_time)
    
    def get_parameter_names(self) -> List[str]:
        """获取需要拟合的参数名称"""
        return ['a1', 'b1', 'a2', 'b2', 'breakpoint']
    
    def get_feature_vector(self, num_tokens: int, **kwargs) -> np.ndarray:
        """
        构造特征向量用于分段线性拟合
        
        Args:
            num_tokens: token数量
            
        Returns:
            特征向量 [num_tokens, 1] (简化版本)
        """
        features = np.array([num_tokens, 1])
        return features
    
    def get_model_description(self) -> str:
        """获取模型描述"""
        breakpoint = self.parameters.get('breakpoint', 'TBD') if self.is_fitted else 'TBD'
        return f"""
GPU MoE 简化分段线性模型:

分段建模公式:
if N <= {breakpoint}:
    T = a1 × N + b1  [小token数段]
else:
    T = a2 × N + b2  [大token数段]

参数说明:
- N: token数量
- 断点: {breakpoint} tokens

拟合参数:
- a1, b1: 左段线性参数
- a2, b2: 右段线性参数

物理解释:
- 左段通常为memory bound（权重加载主导）
- 右段通常为compute bound（计算主导）
        """
    
    def get_complexity_analysis(self, num_tokens: int, **kwargs) -> Dict[str, Any]:
        """获取复杂度分析"""
        N = num_tokens
        
        # 基本复杂度信息
        breakpoint = self.parameters.get('breakpoint', 512) if self.is_fitted else 512
        
        if N <= breakpoint:
            regime = "Left Segment"
            dominant_factor = "Memory/权重加载主导"
        else:
            regime = "Right Segment" 
            dominant_factor = "Compute/计算主导"
        
        return {
            'regime': regime,
            'dominant_factor': dominant_factor,
            'breakpoint': breakpoint,
            'token_count': N,
            'segment': f"{'Left' if N <= breakpoint else 'Right'} segment"
        }
    
    def validate_input(self, num_tokens: int, **kwargs) -> bool:
        """验证输入参数"""
        return num_tokens > 0