"""
性能测试基类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import json
import os
from datetime import datetime


class BaseProfiler(ABC):
    """性能测试基类"""
    
    def __init__(self, name: str, root_path: str, output_dir: str = "profile_results"):
        self.name = name
        self.output_dir = os.path.join(root_path, output_dir)
        self.results = []
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
    
    @abstractmethod
    def run_profile(self, **config) -> List[Dict[str, Any]]:
        """
        运行性能测试
        
        Args:
            **config: 测试配置参数
            
        Returns:
            测试结果列表
        """
        pass
    
    def save_results(self, filename: str = None, filepath: str = None) -> str:
        """
        保存测试结果到文件
        
        Args:
            filename: 输出文件名 (可选)
            
        Returns:
            保存的文件路径
        """
        if filepath is None:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.name}_profile_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
        
        # 添加元数据
        output_data = {
            'profiler_name': self.name,
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(self.results),
            'results': self.results
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        print(f"性能测试结果已保存到: {filepath}")
        return filepath
    
    def load_results(self, filepath: str, results_filter = None) -> List[Dict[str, Any]]:
        """
        从文件加载测试结果
        
        Args:
            filepath: 结果文件路径
            
        Returns:
            测试结果列表
        """
        filepath = os.path.join(self.output_dir, filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if results_filter is not None:
            self.results = [r for r in data.get('results', []) if results_filter(r)]
        else: 
            self.results = data.get('results', [])
        print(f"已加载 {len(self.results)} 条测试结果")
        return self.results
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取测试结果摘要
        
        Returns:
            结果摘要
        """
        if not self.results:
            return {'message': '暂无测试结果'}
        
        latencies = [r['latency'] for r in self.results if 'latency' in r]
        
        if not latencies:
            return {'message': '结果中没有延迟数据'}
        
        return {
            'total_samples': len(self.results),
            'latency_stats': {
                'min': min(latencies),
                'max': max(latencies),
                'mean': sum(latencies) / len(latencies),
                'median': sorted(latencies)[len(latencies)//2]
            },
            'parameter_ranges': self._get_parameter_ranges()
        }
    
    def _get_parameter_ranges(self) -> Dict[str, Tuple[Any, Any]]:
        """获取参数的取值范围"""
        if not self.results:
            return {}
        
        # 收集所有参数名
        all_params = set()
        for result in self.results:
            if 'input_params' in result:
                all_params.update(result['input_params'].keys())
        
        # 计算每个参数的范围
        param_ranges = {}
        for param in all_params:
            values = []
            for result in self.results:
                if 'input_params' in result and param in result['input_params']:
                    values.append(result['input_params'][param])
            
            if values:
                param_ranges[param] = (min(values), max(values))
        
        return param_ranges
    
    def clear_results(self):
        """清空测试结果"""
        self.results = []
        print("测试结果已清空")