"""
算子基类 - 定义所有算子的通用接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import time


class BaseOperator(ABC):
    """算子基类，定义统一的接口"""
    
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    def forward(self, **kwargs) -> Any:
        """
        算子前向计算
        
        Args:
            **kwargs: 算子特定的输入参数
            
        Returns:
            算子输出结果
        """
        pass
    
    @abstractmethod
    def get_input_shapes(self, **kwargs) -> Dict[str, Tuple]:
        """
        获取输入张量的形状信息
        
        Args:
            **kwargs: 算子输入参数
            
        Returns:
            输入张量形状字典
        """
        pass
    
    @abstractmethod
    def get_flops(self, **kwargs) -> int:
        """
        计算算子的理论浮点运算数
        
        Args:
            **kwargs: 算子输入参数
            
        Returns:
            理论FLOPS数量
        """
        pass
    
    @abstractmethod
    def get_memory_usage(self, **kwargs) -> Dict[str, int]:
        """
        计算算子的内存使用量
        
        Args:
            **kwargs: 算子输入参数
            
        Returns:
            内存使用量字典 (bytes)
        """
        pass
    
    def profile_latency(self, num_warmup: int = 5, num_runs: int = 10, **kwargs) -> float:
        """
        测量算子延迟
        
        Args:
            num_warmup: 预热次数
            num_runs: 测试次数
            **kwargs: 算子输入参数
            
        Returns:
            平均延迟时间 (秒)
        """
        # 预热
        for _ in range(num_warmup):
            self.forward(**kwargs)
            
        # 正式测试
        start_time = time.time()
        for _ in range(num_runs):
            self.forward(**kwargs)
        end_time = time.time()
        
        return (end_time - start_time) / num_runs
    
    def get_performance_metrics(self, **kwargs) -> Dict[str, Any]:
        """
        获取性能相关指标
        
        Args:
            **kwargs: 算子输入参数
            
        Returns:
            性能指标字典
        """
        return {
            'name': self.name,
            'input_shapes': self.get_input_shapes(**kwargs),
            'flops': self.get_flops(**kwargs),
            'memory_usage': self.get_memory_usage(**kwargs),
            'latency': self.profile_latency(**kwargs)
        }
    
    def is_input_shape_valid(self, **kwargs) -> bool:
        """
        检查输入是否会爆显存等
        """
        return True