"""
HtoDTransfer算子实现
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
from specmoe.optimizer.performance_simulator.operators.base import BaseOperator

class HtoDTransferOperator(BaseOperator):
    """HtoDTransfer算子实现"""
    
    def __init__(self):
        super().__init__("HtoDTransfer")
    
    def forward(self, transfer_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        assert transfer_tensor.is_pinned()
        output = transfer_tensor.to('cuda', non_blocking=False)
        return output
    
    def get_input_shapes(self, size_in_gb: float = 1.0, **kwargs) -> Dict[str, Tuple]:
        """获取输入张量形状"""
        bytes_total = int(size_in_gb * 1024 * 1024 * 1024)
        # 假设使用float16，每个元素2字节
        num_elements = bytes_total // 2
        return {
            'transfer_tensor': (num_elements,)
        }
    
    def get_flops(self, size_in_gb: float = 1.0, **kwargs) -> int:
        """计算理论FLOPS"""
        return 0  # 数据传输不涉及计算
    
    def get_memory_usage(self, size_in_gb: float = 1.0, **kwargs) -> Dict[str, int]:
        """获取内存使用情况"""
        bytes_total = int(size_in_gb * 1024 * 1024 * 1024)
        return {
            'cpu_memory': bytes_total,  # CPU侧内存
            'gpu_memory': bytes_total   # GPU侧内存
        }
    
    def create_dummy_input(self, size_in_gb: float = 1) -> torch.Tensor:
        """创建用于测试的虚拟输入"""
        bytes = size_in_gb * 1024 * 1024 * 1024
        tensor = torch.randn(int(bytes / 2), device='cpu', dtype=torch.float16, pin_memory=True)
        return tensor