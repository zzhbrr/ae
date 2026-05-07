"""
性能测试模块 - 用于profile各种算子在不同输入下的性能
"""

from specmoe.optimizer.performance_simulator.profiler.base import BaseProfiler
from specmoe.optimizer.performance_simulator.profiler.operator_profiler import OperatorProfiler

__all__ = [
    'BaseProfiler',
    'OperatorProfiler'
]