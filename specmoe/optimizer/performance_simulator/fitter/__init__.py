"""
参数拟合模块 - 用于根据profile结果拟合性能模型参数
"""

from specmoe.optimizer.performance_simulator.fitter.base import BaseFitter
from specmoe.optimizer.performance_simulator.fitter.linear_fitter import LinearFitter
from specmoe.optimizer.performance_simulator.fitter.piecewise_fitter import PiecewiseLinearFitter

__all__ = [
    'BaseFitter',
    'LinearFitter',
    'PiecewiseLinearFitter'
]