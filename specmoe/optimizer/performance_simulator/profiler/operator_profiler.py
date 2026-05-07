"""
算子性能测试器
"""

import numpy as np
import torch
import time
from typing import Dict, Any, List, Union
from specmoe.optimizer.performance_simulator.operators.base import BaseOperator
from specmoe.optimizer.performance_simulator.profiler.base import BaseProfiler
import logging

logger = logging.getLogger(__name__)


class OperatorProfiler(BaseProfiler):
    """算子性能测试器"""
    
    def __init__(self, operator: BaseOperator, root_path: str, output_dir: str = "profile_results"):
        super().__init__(f"{operator.name}_profiler", root_path, output_dir)
        self.operator = operator
        
    def run_profile(self, parameter_ranges: Dict[str, List], 
                   num_warmup: int = 5, num_runs: int = 10,
                   device_info: bool = True) -> List[Dict[str, Any]]:
        """
        运行算子性能测试
        
        Args:
            parameter_ranges: 参数取值范围字典，如 {'batch_size': [1,2,4,8], 'seq_length': [512,1024,2048]}
            num_warmup: 预热次数
            num_runs: 测试次数
            device_info: 是否收集设备信息
            
        Returns:
            测试结果列表
        """
        logger.info(f"开始对 {self.operator.name} 进行性能测试...")
        
        # 生成所有参数组合
        param_combinations = self._generate_param_combinations(parameter_ranges)
        print(f"总共需要测试 {len(param_combinations)} 种参数组合")
        
        # 收集设备信息
        device_info_dict = self._collect_device_info() if device_info else {}
        
        results = []
        for i, params in enumerate(param_combinations):
            try:
                print(f"测试进度: {i+1}/{len(param_combinations)} - {params}")
                
                # 运行单次测试
                result = self._profile_single_config(params, num_warmup, num_runs)
                # result['device_info'] = device_info_dict
                results.append(result)
                
            except Exception as e:
                logger.error(f"测试失败: {params}, 错误: {str(e)}")
                # 记录失败的测试
                results.append({
                    'input_params': params,
                    'latency': None,
                    'error': str(e),
                    'device_info': device_info_dict
                })
        
        self.results.extend(results)
        print(f"性能测试完成，共收集 {len(results)} 个数据点")
        return results
    
    def _generate_param_combinations(self, parameter_ranges: Dict[str, List]) -> List[Dict[str, Any]]:
        """生成所有参数组合"""
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        combinations = []
        
        def generate_recursive(index, current_combo):
            if index == len(param_names):
                combinations.append(current_combo.copy())
                return
            
            param_name = param_names[index]
            for value in param_values[index]:
                current_combo[param_name] = value
                generate_recursive(index + 1, current_combo)
        
        generate_recursive(0, {})
        return combinations
    
    def _profile_single_config(self, params: Dict[str, Any], 
                              num_warmup: int, num_runs: int) -> Dict[str, Any]:
        """对单个参数配置进行性能测试"""
        # 创建输入数据
        if not self.operator.is_input_shape_valid(**params):
            logger.error(f"输入形状无效: {params}")
            raise ValueError(f"输入形状无效: {params}")
        
        input_data = self._create_input_data(params)
        # 预热
        for _ in range(num_warmup):
            try:
                if input_data is not None:
                    if isinstance(input_data, tuple):
                        self.operator.forward(*input_data)
                    else:
                        self.operator.forward(input_data, **params)
                else:
                    self.operator.forward(**params)
            except Exception as e:
                logger.error(f"预热失败: {params}, 错误: {str(e)}")
                # 清空缓存
                torch.cuda.empty_cache()
                continue
        
        # 正式测试
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        for _ in range(num_runs):
            if input_data is not None:
                if isinstance(input_data, tuple):
                    self.operator.forward(*input_data)
                else:
                    self.operator.forward(input_data, **params)
            else:
                self.operator.forward(**params)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()
        
        avg_latency = (end_time - start_time) / num_runs
        
        # 收集性能指标
        try:
            performance_metrics = self.operator.get_performance_metrics(**params)
        except:
            performance_metrics = {}
        
        return {
            'input_params': params,
            'latency': avg_latency,
            # 'throughput': 1.0 / avg_latency if avg_latency > 0 else None,
            'performance_metrics': performance_metrics,
            'num_runs': num_runs,
            'timestamp': time.time()
        }
    
    def _create_input_data(self, params: Dict[str, Any]) -> Union[torch.Tensor, tuple, None]:
        """根据算子类型创建输入数据"""
        operator_name = self.operator.name
        
        if operator_name == "GPU_MoE":
            # GPU MoE需要input_tokens
            num_tokens = params.get('num_tokens', 1024)
            return self.operator.create_dummy_input(num_tokens)
        
        elif operator_name == "GPU_Fused_MoE":
            # GPU Fused MoE需要input_tokens, score
            num_tokens = params.get('num_tokens', 1024)
            return self.operator.create_dummy_input(num_tokens)
        
        elif operator_name == "GPU_Triton_Attention":
            # Triton Attention需要query, key, value, kcache, vcache, qo_indptr, kv_indptr, kv_indices, attn_logits, attn_lse, causal, custom_mask, max_extend_len
            batch_size = params.get('batch_size', 8)
            seq_length = params.get('seq_length', 2048)
            query_length = params.get('query_length', 1)
            return self.operator.create_dummy_input(batch_size, seq_length, query_length)
        
        elif operator_name == "CPU_TargetVerify_Attention":
            # CPU TargetVerify Attention需要query, output, kcache, vcache, seq_lens, start_loc, custom_mask, mask_indptr
            batch_size = params.get('batch_size', 8)
            seq_length = params.get('seq_length', 2048)
            query_length = params.get('query_length', 1)
            return self.operator.create_dummy_input(batch_size, seq_length, query_length)
        
        elif operator_name in ["GPU_Attention", "CPU_Attention"]:
            # Attention需要query, key, value
            batch_size = params.get('batch_size', 8)
            seq_length = params.get('seq_length', 2048)
            query_length = params.get('query_length', seq_length)
            return self.operator.create_dummy_input(batch_size, seq_length, query_length)
        
        elif operator_name == "HtoDTransfer":
            # HtoD Transfer需要transfer数据
            size_in_gb = params.get('size_in_gb', 1.0)
            return self.operator.create_dummy_input(size_in_gb)
        
        else:
            # 其他算子直接使用参数
            return None
    
    def _collect_device_info(self) -> Dict[str, Any]:
        """收集设备信息"""
        device_info = {
            'cpu_info': {
                'torch_threads': torch.get_num_threads(),
                'cpu_count': torch.get_num_interop_threads()
            }
        }
        
        # GPU信息
        if torch.cuda.is_available():
            device_info['gpu_info'] = {
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'device_name': torch.cuda.get_device_name(),
                'memory_allocated': torch.cuda.memory_allocated(),
                'memory_reserved': torch.cuda.memory_reserved(),
                'max_memory_allocated': torch.cuda.max_memory_allocated()
            }
        else:
            device_info['gpu_info'] = {'available': False}
        
        return device_info
    
    def generate_test_configs(self) -> Dict[str, List]:
        """为不同算子生成建议的测试配置"""
        operator_name = self.operator.name
        
        if operator_name == "GPU_MoE":
            return {
                'num_tokens': [64, 128, 256, 512, 1024, 2048, 4096, 8192]
            }
        
        elif operator_name in ["GPU_Attention", "CPU_Attention"]:
            return {
                'batch_size': [1, 2, 4, 8, 16],
                'seq_length': [256, 512, 1024, 2048, 4096],
                'query_length': [256, 512, 1024]  # 可以与seq_length不同
            }
        
        else:
            return {'default_param': [1, 2, 4, 8]}
    
    def run_comprehensive_profile(self, custom_ranges: Dict[str, List] = None,
                                 num_warmup: int = 5, num_runs: int = 10) -> List[Dict[str, Any]]:
        """
        运行全面的性能测试
        
        Args:
            custom_ranges: 自定义参数范围 (可选)
            num_warmup: 预热次数
            num_runs: 测试次数
            
        Returns:
            测试结果列表
        """
        if custom_ranges is None:
            parameter_ranges = self.generate_test_configs()
        else:
            parameter_ranges = custom_ranges
        
        logger.debug(f"使用测试配置: {parameter_ranges}")
        
        return self.run_profile(
            parameter_ranges=parameter_ranges,
            num_warmup=num_warmup,
            num_runs=num_runs
        )