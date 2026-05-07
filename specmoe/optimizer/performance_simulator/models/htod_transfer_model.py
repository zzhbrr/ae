"""
HtoD Transfer性能建模
"""

import numpy as np
from typing import Dict, Any, List
from .base import BasePerformanceModel


class HtoDTransferModel(BasePerformanceModel):
    """
    HtoD Transfer性能模型
    
    建模公式: T = transfer_gb / bandwidth + latency
    """
    
    def __init__(self):
        super().__init__("GPU_HtoD_Transfer")
        
    def predict(self, transfer_gb: float, **kwargs) -> float:
        """
        预测HtoD Transfer执行时间
        
        Args:
            transfer_gb: 传输数据量 (GB)
            
        Returns:
            预测执行时间 (秒)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未拟合参数")
            
        predicted_time = (
            self.parameters['alpha'] * transfer_gb + 
            self.parameters['beta']
        )
        
        return max(0, predicted_time)
    
    def get_parameter_names(self) -> List[str]:
        """获取需要拟合的参数名称"""
        return ['alpha', 'beta']
    
    def get_feature_vector(self, transfer_gb: float, **kwargs) -> np.ndarray:
        features = np.array([
            transfer_gb,  # 传输量特征
            1             # bias项
        ])
        
        return features
    
    
    def validate_input(self, transfer_gb: float, **kwargs) -> bool:
        """
        验证输入参数的有效性
        
        Args:
            transfer_gb: 传输数据量
            
        Returns:
            是否有效
        """
        return transfer_gb > 0
    
    def get_bandwidth_estimate(self) -> float:
        """
        获取估计的带宽
        
        Returns:
            估计带宽 (GB/s)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未拟合参数")
        
        if self.parameters['alpha'] <= 0:
            raise ValueError("无效的alpha参数，无法计算带宽")
            
        return 1.0 / self.parameters['alpha']
    
    def get_latency_estimate(self) -> float:
        """
        获取估计的基础延迟
        
        Returns:
            基础延迟 (秒)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未拟合参数")
            
        return self.parameters['beta']
