"""
GPU Fused MoE算子实现
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
from specmoe.optimizer.performance_simulator.operators.base import BaseOperator
from sglang.srt.utils import set_weight_attrs
from specmoe.layers.FusedMoE import FusedMoE

class GPUFusedMoEOperator(BaseOperator):
    """GPU MoE算子实现"""
    
    def __init__(self, num_experts: int = 8, expert_capacity: int = None, 
                 hidden_size: int = 4096, expert_size: int = 16384, top_k: int = 2):
        super().__init__("GPU_Fused_MoE")
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.hidden_size = hidden_size
        self.expert_size = expert_size
        self.top_k = top_k
        w1 = torch.ones((2*self.num_experts, 3 * self.hidden_size * self.expert_size), device='cuda', dtype=torch.float16)
        self.indices = torch.tensor([0,2,3,6,8,10,12,14], dtype=torch.int64, device='cuda')
        self.fused_moe = FusedMoE(num_experts=self.num_experts, top_k=self.top_k, hidden_size=self.hidden_size, intermediate_size=self.expert_size, params_dtype=torch.float16, tp_size=1)
        set_weight_attrs(self.fused_moe.ws, {"gpu_cache": w1})
    
    def forward(self, input_tokens: torch.Tensor, score: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        MoE前向计算
        
        Args:
            input_tokens: 输入tokens张量 [num_tokens, hidden_size]
            
        Returns:
            输出张量 [num_tokens, hidden_size]
        """
        # print(f"input_tokens: {input_tokens.shape}, score: {score.shape}, indices: {self.indices.shape}")
        output = self.fused_moe(input_tokens, score, self.indices)
        return output
    
    def get_input_shapes(self, num_tokens: int = 1024, **kwargs) -> Dict[str, Tuple]:
        """获取输入张量形状"""
        return {
            'input_tokens': (num_tokens, self.hidden_size)
        }
    
    def get_flops(self, num_tokens: int = 1024, **kwargs) -> int:
        """计算理论FLOPS"""
        
        expert_flops = num_tokens * self.top_k * (
            self.hidden_size * self.expert_size * 3 +  # 两个线性层
            self.expert_size * self.hidden_size * 2  # ReLU激活
        )
        
        return expert_flops
    
    def get_memory_usage(self, num_tokens: int = 1024, **kwargs) -> Dict[str, int]:
        """计算内存使用量"""
        dtype_size = 2  # float16
        
        input_memory = num_tokens * self.hidden_size * dtype_size
        expert_memory = self.num_experts * (
            self.hidden_size * self.expert_size * 2 + self.expert_size * self.hidden_size
        ) * dtype_size
        output_memory = num_tokens * self.hidden_size * dtype_size
        
        return {
            'input': input_memory,
            'experts': expert_memory,
            'output': output_memory,
            'total': input_memory + expert_memory + output_memory
        }
    
    def create_dummy_input(self, num_tokens: int = 1024) -> torch.Tensor:
        """创建用于测试的虚拟输入"""
        input_tokens = torch.randn(num_tokens, self.hidden_size, device='cuda', dtype=torch.float16)
        scores = torch.randn(num_tokens, self.num_experts, device='cuda', dtype=torch.float16)
        return input_tokens, scores

if __name__ == "__main__":
    operator = GPUFusedMoEOperator(num_experts=8, expert_capacity=None, hidden_size=4096, expert_size=14336, top_k=2)
    # bs_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 3000, 4000, 5000]
    bs_list = [100, 300, 500, 824, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 7000, 8000, 9000]
    time_list = []
    iter_num = 20
    cpu_offload_tensor = torch.empty_like(operator.fused_moe.ws, device='cpu').pin_memory()
    cpu_offload_tensor.copy_(operator.fused_moe.ws, non_blocking=False)
    for bs in bs_list:
        inputs = operator.create_dummy_input(num_tokens=bs)
        output = operator.forward(*inputs)
        time_start = time.time()
        for i in range(iter_num):
            operator.fused_moe.ws.copy_(cpu_offload_tensor, non_blocking=False)
            output = operator.forward(*inputs)
            torch.cuda.synchronize()
        time_end = time.time()
        print(f"batch_size: {bs}, output shape: {output.shape}")
        print(f"time: {(time_end - time_start) / iter_num}")
        time_list.append((time_end - time_start) / iter_num)
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # 计算吞吐量
    throughput_list = [bs_list[i] / time_list[i] for i in range(len(bs_list))]
    
    # 保存结果到CSV文件
    results_df = pd.DataFrame({
        'batch_size': bs_list,
        'time_per_iteration': time_list,
        'throughput_tokens_per_sec': throughput_list
    })
    results_df.to_csv('gpu_fused_moe_performance_results.csv', index=False)
    print(f"实验结果已保存到 gpu_fused_moe_performance_results.csv")
    
    plt.clf()
    plt.figure(figsize=(8, 5))
    # plt.plot(bs_list, time_list, label='GPU Fused MoE')
    # plt.scatter(bs_list, time_list, label='GPU Fused MoE')
    
    # 绘制所有点
    plt.scatter(bs_list, throughput_list, label='GPU Fused MoE')
    
    # 特别标识bs=824的点
    if 824 in bs_list:
        idx_824 = bs_list.index(824)
        plt.scatter(824, throughput_list[idx_824], marker='*', s=200, color='red', label='bs=824')
        plt.annotate('bs=824', xy=(824, throughput_list[idx_824]), 
                    xytext=(824+200, throughput_list[idx_824]+500),
                    fontsize=15, color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    plt.xlabel('Batch Size', fontsize=15)
    plt.ylabel('Throughput (tokens/s)', fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=15)
    # plt.title('GPU Fused MoE Time')
    # plt.legend()
    
    # 调整布局，向右上角移动内容
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, bottom=0.12, right=0.95, top=0.95)
    
    # plt.savefig('gpu_fused_moe_throughput_performance_with_batch_size.png')
    plt.savefig('gpu_fused_moe_throughput_performance_with_batch_size.pdf')