"""
性能建模基类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import numpy as np


class BasePerformanceModel(ABC):
    """性能建模基类，定义可解释的性能模型接口"""
    
    def __init__(self, name: str):
        self.name = name
        self.parameters = {}  # 存储拟合后的参数
        self.is_fitted = False
        
    @abstractmethod
    def predict(self, **input_params) -> float:
        """
        根据输入参数预测性能
        
        Args:
            **input_params: 输入参数 (如batch_size, seq_length等)
            
        Returns:
            预测的执行时间 (秒)
        """
        pass
    
    @abstractmethod
    def get_parameter_names(self) -> List[str]:
        """
        获取模型中需要拟合的参数名称列表
        
        Returns:
            参数名称列表
        """
        pass
    
    @abstractmethod
    def get_feature_vector(self, **input_params) -> np.ndarray:
        """
        将输入参数转换为特征向量用于拟合
        
        Args:
            **input_params: 输入参数
            
        Returns:
            特征向量
        """
        pass
    
    def set_parameters(self, parameters: Dict[str, float]) -> None:
        """
        设置模型参数
        
        Args:
            parameters: 参数字典
        """
        self.parameters = parameters.copy()
        self.is_fitted = True
    
    def get_parameters(self) -> Dict[str, float]:
        """获取模型参数"""
        return self.parameters.copy()
    
    def get_model_description(self) -> str:
        """
        获取模型的数学描述
        
        Returns:
            模型公式的字符串描述
        """
        return f"{self.name} 性能模型"
    
    def validate_input(self, **input_params) -> bool:
        """
        验证输入参数的有效性
        
        Args:
            **input_params: 输入参数
            
        Returns:
            是否有效
        """
        return True
    
    def get_complexity_analysis(self, **input_params) -> Dict[str, Any]:
        """
        获取复杂度分析信息
        
        Args:
            **input_params: 输入参数
            
        Returns:
            复杂度分析结果
        """
        return {
            'computational_complexity': 'O(?)',
            'memory_complexity': 'O(?)',
            'bottleneck_analysis': '未实现'
        }