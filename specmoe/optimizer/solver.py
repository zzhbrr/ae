import logging
import math
import json
import os
import time
from typing import Dict, Any, Optional, Tuple, List
import time
import itertools
from specmoe.utils.model_config import ModelConfig
from specmoe.speculative.spec_info import SpeculativeAlgorithm
from specmoe.optimizer.performance_simulator.models import CPUTargetVerifyAttentionModel, GPUFusedMoEModel, HtoDTransferModel, CPUTargetVerifyAttentionModel2, GPUTritonAttentionModel
from specmoe.optimizer.performance_simulator.operators import CPUTargetVerifyAttentionOperator, GPUFusedMoEOperator, HtoDTransferOperator, GPUTritonAttentionOperator
from specmoe.optimizer.performance_simulator.profiler import OperatorProfiler
from specmoe.optimizer.performance_simulator.fitter import PiecewiseLinearFitter, LinearFitter
from specmoe.optimizer.performance_simulator.visualizer import PerformanceVisualizer

logger = logging.getLogger(__name__)


class ExecutionPlan:
    """执行计划的数据类"""
    def __init__(self):
        # Prefill阶段参数
        self.global_batch_size: int = 1
        self.prefill_micro_batch_size: int = 1
        self.prefill_micro_batch_num: int = 1
        self.prefill_weight_cache_ratio: float = 0.0
        
        # Decode阶段参数
        self.decode_micro_batch_num: int = 1
        self.decode_micro_batch_size: int = 1
        self.decode_weight_cache_ratio: float = 0.0
        self.decode_gpu_attention_ratio: float = 0.0
        self.decode_gpu_attention_micro_batch_size: int = 0
        self.decode_gpu_attention_nano_batch_size: int = 0
        self.draft_gpu_execution_ratio: float = 0.0
        
        # Speculative相关参数
        self.speculative_num_steps: int = 1
        self.speculative_eagle_topk: int = 1
        self.speculative_num_draft_tokens: int = 1
        self.speculative_num_steps_gpu: int = 1
        self.speculative_eagle_topk_gpu: int = 1
        self.speculative_num_draft_tokens_gpu: int = 1
        self.decode_spec_policy: str = "SequentialCPUonly"
        self.draft_model_placement: str = "CPU"

        # Settingup参数
        self.cpu_dram_for_kv_cache: int = 0
        self.draft_kv_cache_slot: int = 0
        self.target_cg_nano_kv_cache_slot: int = 0
        
        # 结果
        self.throughput: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典形式，方便传递给SpecMoeEngine"""
        return {
            'global_batch_size': self.global_batch_size,
            'prefill_micro_batch_size': self.prefill_micro_batch_size,
            'prefill_micro_batch_num': self.prefill_micro_batch_num,
            'prefill_weight_cache_ratio': self.prefill_weight_cache_ratio,
            'decode_micro_batch_num': self.decode_micro_batch_num,
            'decode_micro_batch_size': self.decode_micro_batch_size,
            'decode_weight_cache_ratio': self.decode_weight_cache_ratio,
            'decode_gpu_attention_ratio': self.decode_gpu_attention_ratio,
            'decode_gpu_attention_micro_batch_size': self.decode_gpu_attention_micro_batch_size,
            'decode_gpu_attention_nano_batch_size': self.decode_gpu_attention_nano_batch_size,
            'draft_gpu_execution_ratio': self.draft_gpu_execution_ratio,
            'speculative_num_steps': self.speculative_num_steps,
            'speculative_eagle_topk': self.speculative_eagle_topk,
            'speculative_num_draft_tokens': self.speculative_num_draft_tokens,
            'speculative_num_steps_gpu': self.speculative_num_steps_gpu,
            'speculative_eagle_topk_gpu': self.speculative_eagle_topk_gpu,
            'speculative_num_draft_tokens_gpu': self.speculative_num_draft_tokens_gpu,
            'decode_spec_policy': self.decode_spec_policy,
            'draft_model_placement': self.draft_model_placement,
            'cpu_dram_for_kv_cache': self.cpu_dram_for_kv_cache,
            'draft_kv_cache_slot': self.draft_kv_cache_slot,
            'target_cg_nano_kv_cache_slot': self.target_cg_nano_kv_cache_slot,
            'throughput': self.throughput,
        }

class Simulator:
    def __init__(self, plan: ExecutionPlan, average_seq_len: int, max_seq_len: int, target_model_config: ModelConfig, draft_model_config: ModelConfig = None, gpu_moe_fit_file_path:str = None, target_cpu_attn_fit_file_path:str = None, target_gpu_attn_fit_file_path:str = None, draft_gpu_attn_fit_file_path:str = None, draft_cpu_attn_fit_file_path:str = None, htod_transfer_fit_file_path:str = None, profile_root_path:str = None):
        self.plan = plan
        self.target_model_config = target_model_config
        self.draft_model_config = draft_model_config
        self.target_model_layer_num = target_model_config.num_hidden_layers
        self.draft_model_layer_num = draft_model_config.num_hidden_layers if draft_model_config is not None else 0
        self.target_kv_size_per_token_per_layer_gb = target_model_config.get_per_token_per_layer_kv_size()
        self.draft_kv_size_per_token_per_layer_gb = draft_model_config.get_per_token_per_layer_kv_size() if draft_model_config is not None else 0
        self.target_expert_size_gb = target_model_config.get_expert_total_size()
        self.average_seq_len = average_seq_len
        self.max_seq_len = max_seq_len
        self.profile_root_path = profile_root_path

        self.htod_model: HtoDTransferModel = self.get_performance_model("htod", htod_transfer_fit_file_path)
        self.gpu_moe_model: GPUFusedMoEModel = self.get_performance_model("gpu_moe", gpu_moe_fit_file_path)
        self.target_cpu_attn_model: CPUTargetVerifyAttentionModel2 = self.get_performance_model("target_cpu_attn", target_cpu_attn_fit_file_path)
        self.target_gpu_attn_model: GPUTritonAttentionModel = self.get_performance_model("target_gpu_attn", target_gpu_attn_fit_file_path)
        if draft_model_config is not None:
            self.draft_gpu_attn_model: GPUTritonAttentionModel = self.get_performance_model("draft_gpu_attn", draft_gpu_attn_fit_file_path)
            self.draft_cpu_attn_model: CPUTargetVerifyAttentionModel2 = self.get_performance_model("draft_cpu_attn", draft_cpu_attn_fit_file_path)
        else:
            self.draft_cpu_attn_model = None

    def get_performance_model(self, model_type:str, model_fit_path:str = None):
        if model_fit_path is None:
            logger.error("model_fit_path is None")
            return None
        if model_type == "gpu_moe":
            model = GPUFusedMoEModel(
                num_experts=self.target_model_config.num_local_experts,
                hidden_size=self.target_model_config.hidden_size,
                expert_size=self.target_model_config.intermediate_size,
                top_k=self.target_model_config.topk
            )
        elif model_type == "target_cpu_attn":
            model = CPUTargetVerifyAttentionModel2(
                num_heads=self.target_model_config.num_attention_heads,
                head_dim=self.target_model_config.head_dim,
                kv_heads=self.target_model_config.num_key_value_heads
            )
        elif model_type == "target_gpu_attn":
            model = GPUTritonAttentionModel(
                num_heads=self.target_model_config.num_attention_heads,
                head_dim=self.target_model_config.head_dim,
                kv_heads=self.target_model_config.num_key_value_heads
            )
        elif model_type == "draft_gpu_attn":
            model = GPUTritonAttentionModel(
                num_heads=self.draft_model_config.num_attention_heads,
                head_dim=self.draft_model_config.head_dim,
                kv_heads=self.draft_model_config.num_key_value_heads
            )
        elif model_type == "draft_cpu_attn":
            model = CPUTargetVerifyAttentionModel2(
                num_heads=self.draft_model_config.num_attention_heads,
                head_dim=self.draft_model_config.head_dim,
                kv_heads=self.draft_model_config.num_key_value_heads
            )
        elif model_type == "htod":
            model = HtoDTransferModel()
        else:
            logger.error(f"Invalid model type: {model_type}")
            return None

        if not os.path.exists(model_fit_path):
            # 创建文件
            os.makedirs(os.path.dirname(model_fit_path), exist_ok=True)
            # 拟合
            if model_type == "gpu_moe":
                self.fit_for_gpu_moe(model, model_type, model_fit_path)
            elif model_type == "target_cpu_attn" or model_type == "draft_cpu_attn":
                self.fit_for_cpu_attn(model, model_type, model_fit_path)
            elif model_type == "target_gpu_attn" or model_type == "draft_gpu_attn":
                self.fit_for_gpu_attn(model, model_type, model_fit_path)
            elif model_type == "htod":
                self.fit_for_htod(model, model_fit_path)
            else:
                assert False, "Invalid model type"
        else:
            # 读取
            with open(model_fit_path, 'r', encoding='utf-8') as f:
                fit_data = json.load(f)

            model.set_parameters(fit_data['parameters'])
        return model

    def fit_for_gpu_moe(self, model: GPUFusedMoEModel, model_type: str, model_fit_path:str):
        logger.info(f"Fitting GPU MoE model for {model_fit_path}")
        moe_operator = GPUFusedMoEOperator(
            # num_experts=self.target_model_config.num_local_experts,
            # hidden_size=self.target_model_config.hidden_size,
            # expert_size=self.target_model_config.intermediate_size,
            # top_k=self.target_model_config.topk
            num_experts=model.num_experts,
            hidden_size=model.hidden_size,
            expert_size=model.expert_size,
            top_k=model.top_k
        )
        profiler = OperatorProfiler(moe_operator, root_path=self.profile_root_path)
        test_ranges = {
            'num_tokens': [400, 512, 600, 800, 1024, 1500, 2000, 2048, 3000, 3700, 4096, 5000, 6000, 8192]
        }
        logger.info(f"Running profile for GPU MoE model")
        profile_results = profiler.run_profile(
            parameter_ranges=test_ranges,
            num_warmup=1,
            num_runs=7
        )
        profile_file = profiler.save_results(filename=model_type + "_profile.json")

        fitter = PiecewiseLinearFitter(
            min_samples_per_segment=3,
            breakpoint_search_method="exhaustive",  # 可选: "exhaustive", "percentile", "adaptive"
            relative_error_weight=0.5,
            root_path=self.profile_root_path,
        )
        fit_result = fitter.fit(model, profile_results)
        evaluation = fitter.evaluate_fit(model, profile_results)
        logger.debug(f"\n分段线性拟合效果:")
        logger.debug(f"  总体 R² = {evaluation['r2']:.4f}")
        logger.debug(f"  总体 RMSE = {evaluation['rmse']:.6f}")
        if 'breakpoint' in evaluation:
            logger.debug(f"  最优断点 = {evaluation['breakpoint']} tokens")
        if 'max_relative_error' in evaluation:
            logger.debug(f"  最大相对误差 = {evaluation['max_relative_error']:.2%}")
        if 'avg_relative_error' in evaluation:
            logger.debug(f"  平均相对误差 = {evaluation['avg_relative_error']:.2%}")
        if 'left_r2' in evaluation:
            logger.debug(f"  左段 R² = {evaluation['left_r2']:.4f}")
        if 'right_r2' in evaluation:
            logger.debug(f"  右段 R² = {evaluation['right_r2']:.4f}")
        fit_file = fitter.save_fit_results(model, fit_result, evaluation, filepath=model_fit_path)

        visualizer = PerformanceVisualizer(root_path=self.profile_root_path)

        vis_file = visualizer.plot_gpu_moe_accuracy(
            model=model,
            profile_data=profile_results,
            title="GPU MoE Piecewise Linear Performance Prediction",
            save_name=f"{model_type}.png",
            log_scale=True
        )

    def fit_for_gpu_attn(self, model: GPUTritonAttentionModel, model_type: str, model_fit_path:str):
        logger.info(f"Fitting GPU Triton Attention model for {model_fit_path}")
        attn_operator = GPUTritonAttentionOperator(
            num_heads=model.num_heads,
            head_dim=model.head_dim,
            kv_heads=model.kv_heads
        )
        profiler = OperatorProfiler(attn_operator, root_path=self.profile_root_path)
        if model.kv_heads == 8:
            test_ranges = {
                'batch_size': [800, 1024, 1200],
                'seq_length': [200, 400, 600, 700, 800, 1000, 2000],
                'query_length': [1]
            }
        else:
            test_ranges = {
                'batch_size': [200, 300, 350, 400, 500],
                'seq_length': [200, 400, 600, 700, 800, 1000],
                'query_length': [1]
            }
        profile_results = profiler.run_profile(
            parameter_ranges=test_ranges,
            num_warmup=2,
            num_runs=5
        )
        profile_file = profiler.save_results(filename=model_type + "_profile.json")
        fitter = LinearFitter(root_path=self.profile_root_path)
        fit_result = fitter.fit(model, profile_results)
        evaluation = fitter.evaluate_fit(model, profile_results)
        logger.debug(f"拟合效果: R² = {evaluation['r2']:.4f}, RMSE = {evaluation['rmse']:.6f}")
        logger.debug(f"平均相对误差率: {evaluation['mean_relative_error']:.1%}")
        logger.debug(f"平均绝对误差: MAE = {evaluation['mae']:.6f}")

        fit_file = fitter.save_fit_results(model, fit_result, evaluation, filepath=model_fit_path)
        visualizer = PerformanceVisualizer(root_path=self.profile_root_path)
        vis_file = visualizer.plot_attention_accuracy(
            model=model,
            profile_data=profile_results,
            title="GPU Attention Performance Prediction Accuracy",
            save_name=f"{model_type}.png", 
        )

    def fit_for_cpu_attn(self, model: CPUTargetVerifyAttentionModel2, model_type: str, model_fit_path:str):
        logger.info(f"Fitting CPU Attention model for {model_fit_path}")
        attn_operator = CPUTargetVerifyAttentionOperator(
            num_heads=model.num_heads,
            head_dim=model.head_dim,
            kv_heads=model.kv_heads
        )
        profiler = OperatorProfiler(attn_operator, root_path=self.profile_root_path)
        test_ranges = {
            'batch_size': [500, 700, 1000, 1200, 2000, 3000],
            'seq_length': [128, 256, 512, 1024, 2048],
            'query_length': [1, 2, 3, 4, 5]
        }
        logger.info(f"Running profile for CPU Attention model")
        profile_results = profiler.run_profile(
            parameter_ranges=test_ranges,
            num_warmup=1,
            num_runs=7
        )
        profile_file = profiler.save_results(filename=model_type + "_profile.json")

        fitter = LinearFitter(root_path=self.profile_root_path)
        fit_result = fitter.fit(model, profile_results)

        evaluation = fitter.evaluate_fit(model, profile_results)
        logger.debug(f"拟合效果: R² = {evaluation['r2']:.4f}, RMSE = {evaluation['rmse']:.6f}")
        logger.debug(f"平均相对误差率: {evaluation['mean_relative_error']:.1%}")
        logger.debug(f"平均绝对误差: MAE = {evaluation['mae']:.6f}")

        fit_file = fitter.save_fit_results(model, fit_result, evaluation, filepath=model_fit_path)

        visualizer = PerformanceVisualizer(root_path=self.profile_root_path)
        vis_file = visualizer.plot_attention_accuracy(
            model=model,
            profile_data=profile_results,
            title="CPU TargetVerify Attention Performance Prediction Accuracy",
            save_name=f"{model_type}.png", 
            log_scale=False
        )
        error_vis_file = visualizer.plot_relative_error_analysis(
            model=model,
            profile_data=profile_results,
            title="CPU TargetVerify Attention Relative Error Analysis",
            save_name=f"{model_type}_error_analysis.png"
        )

    def fit_for_htod(self, model, model_fit_path:str):
        operator = HtoDTransferOperator()
        profiler = OperatorProfiler(operator, root_path=self.profile_root_path)
        parameter_ranges = {
            'size_in_gb': [0.5, 1.0, 1.5] 
        }
        profile_results = profiler.run_profile(
            parameter_ranges=parameter_ranges,
            num_warmup=1,
            num_runs=3,
            device_info=True
        )
        profile_file = profiler.save_results(filename="htod_transfer_profile.json")
        fitter = LinearFitter(root_path=self.profile_root_path)
        valid_results = [result for result in profile_results if result.get('latency') is not None]
        fit_data = []
        for result in valid_results:
            fit_data.append({
                'latency': result['latency'],
                'input_params': {'transfer_gb': result['input_params']['size_in_gb']}
            })
        fit_result = fitter.fit(model, fit_data)
        evaluation = fitter.evaluate_fit(model, fit_data)
        logger.debug(f"平均相对误差率: {evaluation['mean_relative_error']:.1%}")
        fit_file = fitter.save_fit_results(model, fit_result, evaluation, filepath=model_fit_path)

    def get_draft_accept_length(self, draft_l): # 包括每次verify保底的1个token
        if draft_l == 1:
            return 1
        bias = -0.1
        draft_length = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 75, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 400, 500, 600, 700]
        accept_length = [0.58, 0.78, 0.91, 1.15, 1.24, 1.32, 1.45, 1.48, 1.5, 1.59, 1.52, 1.6, 1.64, 1.73, 1.87, 1.85, 1.85, 1.89, 1.84, 1.95, 2.11, 2.00, 2.04, 2.25, 2.21, 2.26, 2.24, 2.27, 2.33, 2.18, 2.19, 2.36, 2.37, 2.44, 2.37, 2.4, 2.49, 2.36, 2.43, 2.35, 2.33, 2.42, 2.49, 2.5, 2.56, 2.55, 2.5, 2.49]
        if draft_l in draft_length:
            return accept_length[draft_length.index(draft_l)] + 1 + bias
        if draft_l >= 120: return 2.60 + 1 + bias
        a = 0.351262282884418
        b = 0.593413804028745
        return a * math.log(draft_l) + b + 1 + bias

    def get_sd_policy_from_draftl(self, draft_l):
        # draft_l -> (nstep, topk)
        draftl_to_sd_policy = {
            1: (1, 1), 
            2: (2, 2), 
            3: (2, 3), 
            4: (2, 4), 
            5: (3, 4), 
            6: (4, 4), 
            7: (4, 4), 
            8: (4, 5), 
            9: (4, 5), 
            10: (4, 5),
            11: (4, 5),
            12: (4, 5),
            13: (4, 5),
            14: (4, 5),
            15: (4, 5),
            16: (4, 5),
            17: (4, 5),
            18: (4, 5),
            19: (4, 5),
            20: (4, 5),
        }
        if draft_l in draftl_to_sd_policy:
            return draftl_to_sd_policy[draft_l]
        else:
            if draft_l < 30:
                return (6, 6)
            else:
                assert False

    def get_transfer_time(self, decode_weight_cache_ratio, decode_gpu_attention_ratio, now_generation_length) -> float:
        """计算transfer time"""
        weight_size_in_gb = self.target_expert_size_gb / self.target_model_layer_num * (1-decode_weight_cache_ratio)
        kv_cache_size_in_gb = self.plan.global_batch_size * (self.average_seq_len + now_generation_length) * decode_gpu_attention_ratio * self.target_kv_size_per_token_per_layer_gb
        return self.htod_model.predict(weight_size_in_gb + kv_cache_size_in_gb)
    
    def get_cpu_attn_factor(self, draft_l): # expert传输会争抢带宽
        factor = {
            1: 1.4, 
            2: 1.4, 
            3: 1.4, 
            4: 1.4,
            5: 1.3,
            6: 1.2,
            7: 1.2
        }
        if draft_l in factor:
            return factor[draft_l]
        else:
            return 1.3

    def get_throughput(self, plan: Tuple, now_generation_length: int, log: bool = False) -> float:
        """计算throughput"""
        (
            decode_micro_batch_num,
            decode_weight_cache_ratio,
            decode_gpu_attention_ratio,
            draft_gpu_execution_ratio,
            speculative_num_draft_tokens_cpu,
            speculative_num_draft_tokens_gpu,
        ) = plan
        if log:
            print("-----------Speculative Algorithm throughput-----------")
            print("Plan:")
            print(f"Global batch size: {self.plan.global_batch_size}")
            print(f"Decode micro batch num: {decode_micro_batch_num}")
            print(f"Decode weight cache ratio: {decode_weight_cache_ratio}")
            print(f"Decode GPU attention ratio: {decode_gpu_attention_ratio}")
            print(f"Draft GPU execution ratio: {draft_gpu_execution_ratio}")
            print(f"Speculative num draft tokens CPU: {speculative_num_draft_tokens_cpu}")
            print(f"Speculative num draft tokens GPU: {speculative_num_draft_tokens_gpu}")

        if decode_gpu_attention_ratio == 0:
            assert speculative_num_draft_tokens_gpu == 1
        micro_batch_size = self.plan.global_batch_size // decode_micro_batch_num
        gpu_attn_size_one_mb = int(micro_batch_size * decode_gpu_attention_ratio)
        cpu_attn_size_one_mb = micro_batch_size - gpu_attn_size_one_mb
        gpu_attn_size = int(self.plan.global_batch_size * decode_gpu_attention_ratio)
        cpu_attn_size = self.plan.global_batch_size - gpu_attn_size
        if self.draft_model_config is not None:
            draft_gpu_size = int(self.plan.global_batch_size * draft_gpu_execution_ratio)
            draft_cpu_size = self.plan.global_batch_size - draft_gpu_size
        else:
            draft_gpu_size = 0
            draft_cpu_size = 0

        # A Layer's time
        transfer_time = self.get_transfer_time(decode_weight_cache_ratio, decode_gpu_attention_ratio, now_generation_length)

        moe_input_token_num = cpu_attn_size_one_mb * speculative_num_draft_tokens_cpu + gpu_attn_size_one_mb * speculative_num_draft_tokens_gpu
        one_micro_batch_gpu_time = self.gpu_moe_model.predict(moe_input_token_num)
        # gpu_time = one_micro_batch_gpu_time * decode_micro_batch_num # GPU Attention时间可以忽略
        gpu_time = self.gpu_moe_model.predict(cpu_attn_size * speculative_num_draft_tokens_cpu + gpu_attn_size * speculative_num_draft_tokens_gpu)  # GPU Attention时间可以忽略

        cpu_time = self.get_cpu_attn_factor(speculative_num_draft_tokens_cpu) * self.target_cpu_attn_model.predict(cpu_attn_size_one_mb, self.average_seq_len + now_generation_length, query_length=speculative_num_draft_tokens_cpu) * decode_micro_batch_num

        target_model_execution_time = max(transfer_time, max(gpu_time, cpu_time)) * self.target_model_layer_num
        if log:
            print("Detail:")
            print("one layer transfer_time: ", transfer_time)
            print("one layer gpu_time: ", gpu_time)
            print("one micro-batch gpu_time: ", one_micro_batch_gpu_time)
            print("one layer cpu_time: ", cpu_time)
            print("one layer execution time: ", max(transfer_time, max(gpu_time, cpu_time)))
            print(f"Target model execution time: {target_model_execution_time}")

        if speculative_num_draft_tokens_cpu == 1:
            draft_time = 0
            output_tokens = self.plan.global_batch_size
        else:
            assert self.draft_model_config is not None
            gpu_attn_req_accept_length = self.get_draft_accept_length(speculative_num_draft_tokens_gpu) 
            cpu_attn_req_accept_length = self.get_draft_accept_length(speculative_num_draft_tokens_cpu)
            gpu_draft_nstep, gpu_draft_topk = self.get_sd_policy_from_draftl(speculative_num_draft_tokens_gpu)
            cpu_draft_nstep, cpu_draft_topk = self.get_sd_policy_from_draftl(speculative_num_draft_tokens_cpu)

            draft_gpu_workload_factor = 5
            draft_cpu_workload_factor = 1.8
            # GPU
            ## gpu draft extend
            draft_gpu_extend_time = self.draft_gpu_attn_model.predict(gpu_attn_size, (self.average_seq_len + now_generation_length), query_length=1) + self.draft_gpu_attn_model.predict(draft_gpu_size - gpu_attn_size, (self.average_seq_len + now_generation_length), query_length=1) # cause I fail to run gpu extend attention in profiling, so I use querylength=1 to replace it (<15 query length nearly don't affect gpu execution time)
            ## gpu draft generate
            draft_gpu_generate_time = self.draft_gpu_attn_model.predict(gpu_attn_size, (self.average_seq_len + now_generation_length), query_length=1) * (gpu_draft_nstep - 1) + self.draft_gpu_attn_model.predict(draft_gpu_size - gpu_attn_size, (self.average_seq_len + now_generation_length), query_length=1) * (cpu_draft_nstep - 1)
            draft_gpu_offload_kvcache_time = self.htod_model.predict(draft_gpu_size * cpu_attn_req_accept_length * self.draft_kv_size_per_token_per_layer_gb) * 90 
            draft_gpu_overall_time = draft_gpu_extend_time * draft_gpu_workload_factor + draft_gpu_generate_time * draft_gpu_workload_factor + draft_gpu_offload_kvcache_time

            # CPU
            ## cpu draft extend
            draft_cpu_extend_time = self.draft_cpu_attn_model.predict(draft_cpu_size, (self.average_seq_len + now_generation_length), query_length=cpu_attn_req_accept_length) * self.get_cpu_attn_factor(cpu_attn_req_accept_length)
            ## cpu draft generate
            draft_cpu_generate_time = self.draft_cpu_attn_model.predict(draft_cpu_size, (self.average_seq_len + now_generation_length), query_length=cpu_draft_topk) * (cpu_draft_nstep - 1) * self.get_cpu_attn_factor(cpu_draft_topk)
            draft_cpu_overall_time = draft_cpu_extend_time * draft_cpu_workload_factor + draft_cpu_generate_time * draft_cpu_workload_factor

            draft_time = max(draft_gpu_overall_time, draft_cpu_overall_time) # 乘上系数来表示一些其他负载
            output_tokens = gpu_attn_req_accept_length * gpu_attn_size + cpu_attn_req_accept_length * cpu_attn_size

            if log:
                print("draft_gpu_overall_time: ", draft_gpu_overall_time)
                print("    draft_gpu_offload_kvcache_time: ", draft_gpu_offload_kvcache_time)
                print("    draft_gpu_extend_time: ", draft_gpu_extend_time * draft_gpu_workload_factor)
                print("    draft_gpu_generate_time: ", draft_gpu_generate_time * draft_gpu_workload_factor)
                print("draft_cpu_overall_time: ", draft_cpu_overall_time)
                print("    draft_cpu_extend_time: ", draft_cpu_extend_time * draft_cpu_workload_factor)
                print("    draft_cpu_generate_time: ", draft_cpu_generate_time * draft_cpu_workload_factor)

        if log:
            print("draft_time: ", draft_time)
            print("output_tokens: ", output_tokens)
            print("throughput: ", output_tokens / (target_model_execution_time + draft_time))

        return output_tokens / (target_model_execution_time + draft_time)

class Solver:
    def __init__(
        self,
        target_model_path: str,
        draft_model_path: str = None,
        average_seq_len: int = 1024,
        max_seq_len: int = 1024,
        max_speculative_len: int = 0,
        gpu_memory_gb: int = 23,
        cpu_memory_gb: int = 200,
        max_output_length: int = 32,
        speculative_algorithm: str = "EAGLE",
        tp_size: int = 1,
        gpu_moe_fit_file_path:str = None, 
        target_cpu_attn_fit_file_path:str = None, 
        target_gpu_attn_fit_file_path:str = None, 
        draft_gpu_attn_fit_file_path:str = None,
        draft_cpu_attn_fit_file_path:str = None, 
        htod_transfer_fit_file_path:str = None,
        profile_root_path:str = None,
        log_level:str = "INFO",
        fix_big_draft_token_num: bool = False
    ):
        self.target_model_path = target_model_path
        self.draft_model_path = draft_model_path
        self.average_seq_len = average_seq_len
        self.max_seq_len = max_seq_len
        self.max_speculative_len = max_speculative_len
        self.gpu_memory_gb = gpu_memory_gb
        self.cpu_memory_gb = cpu_memory_gb
        self.max_output_length = max_output_length
        self.speculative_algorithm = speculative_algorithm
        self.tp_size = tp_size
        self.log_level = log_level
        logging.getLogger().setLevel(getattr(logging, log_level.upper()))
        self.fix_big_draft_token_num = fix_big_draft_token_num

        self.frozen_policy: bool = False

        # 初始化模型配置
        spec_algo = SpeculativeAlgorithm.from_string(speculative_algorithm)
        self.target_model_config = ModelConfig(
            target_model_path, 
            is_draft_model=False,
            spec_algorithm=spec_algo
        )
        self.target_model_layer_num = self.target_model_config.num_hidden_layers
        if draft_model_path is not None:
            self.draft_model_config = ModelConfig(
                draft_model_path, 
                is_draft_model=True,
                spec_algorithm=spec_algo
            )
            self.draft_model_layer_num = self.draft_model_config.num_hidden_layers
            self.draft_kv_size_per_token_per_layer_gb = self.draft_model_config.get_per_token_per_layer_kv_size()
        else:
            self.draft_model_config = None
            self.draft_kv_size_per_token_per_layer_gb = 0
            self.draft_model_layer_num = 0

        # 计算模型内存需求
        self.target_model_overall_size_gb = self.target_model_config.get_all_size()
        self.target_expert_size_gb = self.target_model_config.get_expert_total_size()
        self.target_kv_size_per_token_per_layer_gb = self.target_model_config.get_per_token_per_layer_kv_size()

        print(f"Target model overall size: {self.target_model_overall_size_gb:.2f} GB")
        print(f"Target model expert size: {self.target_expert_size_gb:.2f} GB")
        print(f"Target KV size per token: {self.target_kv_size_per_token_per_layer_gb * 1024} MB")
        print(f"Draft KV size per token: {self.draft_kv_size_per_token_per_layer_gb * 1024} MB")

        self.plan = ExecutionPlan()
        self.simulator = Simulator(self.plan, self.average_seq_len, self.max_seq_len, self.target_model_config, self.draft_model_config, gpu_moe_fit_file_path, target_cpu_attn_fit_file_path, target_gpu_attn_fit_file_path, draft_gpu_attn_fit_file_path, draft_cpu_attn_fit_file_path, htod_transfer_fit_file_path, profile_root_path)

    def solve_prefill(self) -> Dict[str, Any]:
        """计算prefill阶段的执行计划"""
        remained_for_others = 15
        if self.cpu_memory_gb > 200:
            remained_for_others = 15
        elif self.cpu_memory_gb > 150:
            remained_for_others = 10
        elif self.cpu_memory_gb > 100:
            remained_for_others = 7
        cpu_memory_for_kv_cache = self.cpu_memory_gb - self.target_expert_size_gb - remained_for_others # 预留10GB给其他(qkv_pin, kv cache for draft tokens)
        global_batchsize = cpu_memory_for_kv_cache // ((self.average_seq_len + self.max_output_length + self.max_speculative_len) * (self.target_kv_size_per_token_per_layer_gb * self.target_model_layer_num + self.draft_kv_size_per_token_per_layer_gb * self.draft_model_layer_num))

        gpu_memory_remained = self.gpu_memory_gb
        gpu_memory_remained -= self.target_expert_size_gb / self.target_model_layer_num * 2 # 减去两层expert
        gpu_memory_remained -= self.target_model_overall_size_gb - self.target_expert_size_gb # 减去non-moe部分参数
        gpu_memory_remained -= 1.63 * self.draft_model_layer_num # 减去draft model
        assert gpu_memory_remained > 0, "GPU memory is not enough"
        gpu_memory_remained -= 4
        gpu_memory_remained = max(gpu_memory_remained, 2)
        micro_batchsize = gpu_memory_remained // (self.max_seq_len * (self.target_kv_size_per_token_per_layer_gb + self.draft_kv_size_per_token_per_layer_gb) + 2*self.max_seq_len*self.target_model_config.topk*(2*self.target_model_config.intermediate_size+self.target_model_config.intermediate_size+self.target_model_config.hidden_size)/1024/1024/1024)
        micro_batchsize = min(micro_batchsize, global_batchsize)
        # 激活值，第一个2表示fp16
        print(f"Prefill global batch size: {global_batchsize}")
        print(f"Prefill micro batch size: {micro_batchsize}")
        print(
            f"CPU Utilize: Expert weight size: {self.target_expert_size_gb} GB, Target KV Cache size: {(self.average_seq_len + self.max_output_length + self.max_speculative_len) * self.target_kv_size_per_token_per_layer_gb * global_batchsize * self.target_model_layer_num} GB, Draft KV Cache size: {(self.average_seq_len + self.max_output_length + self.max_speculative_len) * self.draft_kv_size_per_token_per_layer_gb * global_batchsize * self.draft_model_layer_num} GB"
        )
        print(
            f"GPU Utilize: Non-MoE weight size: {self.target_model_overall_size_gb - self.target_expert_size_gb} GB, Target KV Cache size: {self.max_seq_len * self.target_kv_size_per_token_per_layer_gb * micro_batchsize} GB, Draft KV Cache size: {self.max_seq_len * self.draft_kv_size_per_token_per_layer_gb * micro_batchsize} GB, Expert weight size: {self.target_expert_size_gb / self.target_model_layer_num * 2} GB, activation size: {2*self.max_seq_len * micro_batchsize * (2*self.target_model_config.intermediate_size+self.target_model_config.intermediate_size+self.target_model_config.hidden_size) * self.target_model_config.topk / 1024/1024/1024} GB, Draft Model: {1.63*self.draft_model_layer_num} GB"
        )
        return {
            'global_batch_size': int(global_batchsize),
            'prefill_micro_batch_size': int(micro_batchsize),
            'prefill_micro_batch_num': int((global_batchsize + micro_batchsize - 1) // micro_batchsize),
            'prefill_weight_cache_ratio': 0.0,
        }

    def satisfy_gpu_memory(self, plan, log: bool = False, record_cpu_kvcache_size: bool = False) -> bool:
        """判断是否满足GPU内存要求"""
        decode_micro_batch_num, decode_weight_cache_ratio, decode_gpu_attention_ratio, draft_gpu_execution_ratio, speculative_num_draft_tokens_cpu, speculative_num_draft_tokens_gpu = plan
        mb_size = self.plan.global_batch_size // decode_micro_batch_num
        gpu_attn_size = mb_size * decode_gpu_attention_ratio
        gpu_memory_used = self.target_expert_size_gb / self.target_model_layer_num * 2 # expert cache two layers
        gpu_memory_used += self.target_model_overall_size_gb - self.target_expert_size_gb # non-moe weight size
        gpu_memory_used += self.target_expert_size_gb * decode_weight_cache_ratio # cache for every layers expert
        if gpu_attn_size > 0:
            gpu_memory_used += 1 
        if draft_gpu_execution_ratio > 0:
            gpu_memory_used += self.draft_kv_size_per_token_per_layer_gb * (self.average_seq_len + self.max_output_length + self.max_speculative_len) * self.plan.global_batch_size * draft_gpu_execution_ratio # draft KV cache
        gpu_memory_used += self.target_model_config.intermediate_size * 2 / 1024/1024/1024 * (gpu_attn_size * speculative_num_draft_tokens_gpu + (mb_size - gpu_attn_size) * speculative_num_draft_tokens_cpu) # activation
        if log:
            print("="*100)
            print("Memory Constraint Check, decode_micro_batch_num: ", decode_micro_batch_num, "decode_weight_cache_ratio: ", decode_weight_cache_ratio, "decode_gpu_attention_ratio: ", decode_gpu_attention_ratio, "draft_gpu_execution_ratio: ", draft_gpu_execution_ratio)
            print("expert cache two layers: ", self.target_expert_size_gb / self.target_model_layer_num * 2, "GB")
            print("expert cache for every layers: ", self.target_expert_size_gb * decode_weight_cache_ratio, "GB")
            print("target KV cache: ", 1 if gpu_attn_size > 0 else 0, "GB")
            print("draft KV cache: ", self.draft_kv_size_per_token_per_layer_gb * (self.average_seq_len + self.max_output_length + self.max_speculative_len) * self.plan.global_batch_size * draft_gpu_execution_ratio, "GB")
            print("activation: ", self.target_model_config.intermediate_size * 2 / 1024/1024/1024 * (gpu_attn_size * speculative_num_draft_tokens_gpu + (mb_size - gpu_attn_size) * speculative_num_draft_tokens_cpu), "GB")
            print("draft model: ", 1.63*self.draft_model_layer_num, "GB")
            print("total: ", gpu_memory_used + 2, "GB")
            print("="*100)
        if record_cpu_kvcache_size:
            # only count target model kv cache
            self.cpu_dram_for_kv_cache = self.target_kv_size_per_token_per_layer_gb * (self.average_seq_len + self.max_output_length + self.max_speculative_len) * self.plan.global_batch_size * self.target_model_layer_num 
        if gpu_memory_used + 2 > self.gpu_memory_gb: # 预留2GB给其他
            return False
        else:
            return True

    def new_solve_decode(self) -> Dict[str, Any]:
        stage_1_plan = self.new_solve_decode_stage1(log=True)
        self.stage_1_plan = stage_1_plan
        plan = self.new_solve_decode_stage2(stage_1_plan=stage_1_plan, now_generation_length=0)
        self.max_speculative_len = max(plan['speculative_eagle_topk'] * plan['speculative_num_steps'], plan['speculative_eagle_topk_gpu'] * plan['speculative_num_steps_gpu'])
        return {
            'decode_micro_batch_num': plan['decode_micro_batch_num'],
            'decode_micro_batch_size': plan['decode_micro_batch_size'],
            'decode_weight_cache_ratio': stage_1_plan['decode_weight_cache_ratio'],
            'decode_gpu_attention_ratio': plan['decode_gpu_attention_ratio'],
            'decode_gpu_attention_micro_batch_size': plan['decode_gpu_attention_micro_batch_size'],
            'decode_gpu_attention_nano_batch_size': plan['decode_gpu_attention_nano_batch_size'],
            'draft_gpu_execution_ratio': stage_1_plan['draft_gpu_execution_ratio'],
            'speculative_num_steps': plan['speculative_num_steps'],
            'speculative_eagle_topk': plan['speculative_eagle_topk'],
            'speculative_num_draft_tokens': plan['speculative_num_draft_tokens'],
            'speculative_num_steps_gpu': plan['speculative_num_steps_gpu'],
            'speculative_eagle_topk_gpu': plan['speculative_eagle_topk_gpu'],
            'speculative_num_draft_tokens_gpu': plan['speculative_num_draft_tokens_gpu'],
            'decode_spec_policy': "SequentialCGCoop",
            'draft_model_placement': "CPU",
            'cpu_dram_for_kv_cache': self.cpu_dram_for_kv_cache,
            'draft_kv_cache_slot': stage_1_plan['draft_kv_cache_slot'],
            'target_cg_nano_kv_cache_slot': stage_1_plan['target_cg_nano_kv_cache_slot'],
            'throughput': plan['throughput'],
        }

    def new_solve_decode_stage1(self, log: bool = False) -> Dict[str, Any]:
        '''
        需要得到:
            1. draft kv cache slot cnt, 根据此可以计算draft gpu execution ratio
            2. decode weight cache ratio, decode阶段缓存多少weight
            3. target kv cache slot cnt, target model执行中需要多少kv cache slot
        '''
        # 1. 刨去expert_cache, non-moe weight, draft mode size, 剩余的gpu memory优先留给draft kv cache；还有activation，不过<1GB
        expert_cache_size = self.target_expert_size_gb / self.target_model_layer_num * 2
        non_moe_weight_size = self.target_model_overall_size_gb - self.target_expert_size_gb
        draft_model_size = 1.63 # XXX: hardcode draft model size
        gpu_memory_remained = self.gpu_memory_gb - expert_cache_size - non_moe_weight_size - draft_model_size - 5 # remain 3 GB for other tensor like activation
        if self.fix_big_draft_token_num:
            gpu_memory_remained -= 2
        gpu_memory_remained = max(gpu_memory_remained, 0)

        # print("gpu_memory_remained: ", gpu_memory_remained)
        # print("global_batch_size: ", self.plan.global_batch_size)
        if self.draft_model_path is not None:
            draft_kv_cache_in_gpu_size = min(self.draft_kv_size_per_token_per_layer_gb * self.plan.global_batch_size * (self.average_seq_len + self.max_output_length + self.max_speculative_len) * self.draft_model_layer_num, gpu_memory_remained)
            # print("draft_kv_cache_in_gpu_size: ", draft_kv_cache_in_gpu_size)
            draft_kv_cache_slot = draft_kv_cache_in_gpu_size // (self.draft_kv_size_per_token_per_layer_gb * self.draft_model_layer_num)
            draft_gpu_execution_ratio = draft_kv_cache_slot / (self.plan.global_batch_size * (self.average_seq_len + self.max_output_length + self.max_speculative_len))
            gpu_memory_remained -= draft_kv_cache_in_gpu_size # 减去draft kv cache
        else:
            draft_kv_cache_in_gpu_size = 0
            draft_kv_cache_slot = 0
            draft_gpu_execution_ratio = 0

        # 2. 剩余的gpu memory优先给 weight
        decode_weight_cache_ratio = gpu_memory_remained / self.target_expert_size_gb
        if decode_weight_cache_ratio > 1:
            decode_weight_cache_ratio = 1
            assert False, "GPU memory > Expert weight size, no need to use offloading"

        if decode_weight_cache_ratio > 0.3: # 当gpu memory大于一定程度时，启用target cg cooperate，这个“程度“是一个超参数，需要调整
            # 启用target cg cooperate
            # 为target nano batch分配1GB空间？
            assert gpu_memory_remained > 1
            decode_weight_cache_ratio = (gpu_memory_remained - 1) / self.target_expert_size_gb
            target_cg_nano_kv_cache_slot = 1 // self.target_kv_size_per_token_per_layer_gb
        else:
            target_cg_nano_kv_cache_slot = 0

        can_choose_decode_weight_cache_ratio = [i / self.target_model_config.num_local_experts for i in range(0, self.target_model_config.num_local_experts + 1)]
        decode_weight_cache_ratio = max([x for x in can_choose_decode_weight_cache_ratio if x <= decode_weight_cache_ratio])

        if log:
            print("="*100)
            print("Stage 1 Plan")
            print("decode_weight_cache_ratio: ", decode_weight_cache_ratio)
            print("target_cg_nano_kv_cache_slot: ", target_cg_nano_kv_cache_slot)
            print("draft_kv_cache_slot: ", draft_kv_cache_slot)
            print("draft_gpu_execution_ratio: ", draft_gpu_execution_ratio)
            print("draft_kv_cache_GPU_memory: ", draft_kv_cache_in_gpu_size, "GB")
            print("="*100)

        return {
            'decode_weight_cache_ratio': decode_weight_cache_ratio,
            'target_cg_nano_kv_cache_slot': int(target_cg_nano_kv_cache_slot),
            'draft_kv_cache_slot': int(draft_kv_cache_slot),
            'draft_gpu_execution_ratio': draft_gpu_execution_ratio,
        }

    def new_solve_decode_stage2(self, stage_1_plan: Dict[str, Any], now_generation_length: int, during_running: bool = False, running_micro_batch_num: int = 0, log: bool = True) -> Dict[str, Any]:
        '''
        running_micro_batch_num: 当初始的micro-batch num确定后，之后也不改了
        '''
        begin_time = time.time()
        decode_micro_batch_num_range = [i for i in range(2, 20)]
        if during_running and running_micro_batch_num != 0:
            decode_micro_batch_num_range = [running_micro_batch_num]
        decode_weight_cache_ratio_range = [stage_1_plan['decode_weight_cache_ratio']]
        if stage_1_plan['target_cg_nano_kv_cache_slot'] > 0:
            decode_gpu_attention_ratio_range = [0.0 + 0.05 * i for i in range(20)] + [1]
        else:
            decode_gpu_attention_ratio_range = [0]
        draft_gpu_execution_ratio_range = [stage_1_plan['draft_gpu_execution_ratio']]

        if self.draft_model_path is not None:
            speculative_num_draft_tokens_cpu_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            speculative_num_draft_tokens_gpu_range = [i for i in range(1, 20)]
            if during_running:
                speculative_num_draft_tokens_cpu_range = [i for i in range(1, min(self.max_speculative_len, 11))]
                speculative_num_draft_tokens_gpu_range = [i for i in range(1, min(self.max_speculative_len, 20))]
            # speculative_num_draft_tokens_cpu_range = [2]
            # speculative_num_draft_tokens_gpu_range = [1]
        else:
            speculative_num_draft_tokens_cpu_range = [1]
            speculative_num_draft_tokens_gpu_range = [1]

        max_throughput = 0
        best_plan = None

        policies = itertools.product(
            decode_micro_batch_num_range,
            decode_weight_cache_ratio_range,
            decode_gpu_attention_ratio_range,
            draft_gpu_execution_ratio_range,
            speculative_num_draft_tokens_cpu_range,
            speculative_num_draft_tokens_gpu_range,
        )
        for (
            decode_micro_batch_num,
            decode_weight_cache_ratio,
            decode_gpu_attention_ratio,
            draft_gpu_execution_ratio,
            speculative_num_draft_tokens_cpu,
            speculative_num_draft_tokens_gpu,
        ) in policies:
            if draft_gpu_execution_ratio < decode_gpu_attention_ratio: 
                continue
            if self.draft_model_path is None and speculative_num_draft_tokens_cpu != 1:
                continue
            if self.draft_model_path is None and speculative_num_draft_tokens_gpu != 1:
                continue
            if decode_gpu_attention_ratio == 0 and speculative_num_draft_tokens_gpu != 1: 
                continue
            if decode_gpu_attention_ratio != 0 and speculative_num_draft_tokens_gpu < speculative_num_draft_tokens_cpu:
                continue
            if draft_gpu_execution_ratio != 0 and speculative_num_draft_tokens_cpu == 1 and speculative_num_draft_tokens_gpu == 1:
                continue
            if not self.satisfy_gpu_memory([decode_micro_batch_num, decode_weight_cache_ratio, decode_gpu_attention_ratio, draft_gpu_execution_ratio, speculative_num_draft_tokens_cpu, speculative_num_draft_tokens_gpu], log=False):
                continue
            # print(f"iterate on [decode_micro_batch_num, decode_weight_cache_ratio, decode_gpu_attention_ratio, draft_gpu_execution_ratio, speculative_num_draft_tokens_cpu, speculative_num_draft_tokens_gpu]: {decode_micro_batch_num, decode_weight_cache_ratio, decode_gpu_attention_ratio, draft_gpu_execution_ratio, speculative_num_draft_tokens_cpu, speculative_num_draft_tokens_gpu}")
            throughput = self.simulator.get_throughput((decode_micro_batch_num, decode_weight_cache_ratio, decode_gpu_attention_ratio, draft_gpu_execution_ratio, speculative_num_draft_tokens_cpu, speculative_num_draft_tokens_gpu), now_generation_length=now_generation_length, log=False)
            if throughput > max_throughput:
                max_throughput = throughput
                best_plan = (decode_micro_batch_num, decode_weight_cache_ratio, decode_gpu_attention_ratio, draft_gpu_execution_ratio, speculative_num_draft_tokens_cpu, speculative_num_draft_tokens_gpu)

        # print(best_plan)
        # print simulated basic information
        self.satisfy_gpu_memory(best_plan, log=log, record_cpu_kvcache_size=True)
        max_throughput = self.simulator.get_throughput(best_plan, now_generation_length=now_generation_length, log=log)

        decode_micro_batch_size = (self.plan.global_batch_size + best_plan[0] - 1) // best_plan[0]
        decode_gpu_attention_micro_batch_size = int(decode_micro_batch_size * best_plan[2])
        decode_gpu_attention_nano_batch_size = stage_1_plan['target_cg_nano_kv_cache_slot'] / (self.average_seq_len + now_generation_length + self.max_speculative_len)

        end_time = time.time()
        if log:
            print("stage 2 solve time: ", end_time - begin_time)

        return {
            'decode_micro_batch_num': best_plan[0],
            'decode_micro_batch_size': decode_micro_batch_size,
            'decode_weight_cache_ratio': best_plan[1],
            'decode_gpu_attention_ratio': best_plan[2],
            'decode_gpu_attention_micro_batch_size': decode_gpu_attention_micro_batch_size,
            'decode_gpu_attention_nano_batch_size': decode_gpu_attention_nano_batch_size,
            'draft_gpu_execution_ratio': best_plan[3],
            'speculative_num_steps': self.simulator.get_sd_policy_from_draftl(best_plan[4])[0],
            'speculative_eagle_topk': self.simulator.get_sd_policy_from_draftl(best_plan[4])[1],
            'speculative_num_draft_tokens': best_plan[4],
            'speculative_num_steps_gpu': self.simulator.get_sd_policy_from_draftl(best_plan[5])[0],
            'speculative_eagle_topk_gpu': self.simulator.get_sd_policy_from_draftl(best_plan[5])[1],
            'speculative_num_draft_tokens_gpu': best_plan[5],
            'decode_spec_policy': "SequentialCGCoop",
            'draft_model_placement': "CPU",
            'cpu_dram_for_kv_cache': self.cpu_dram_for_kv_cache,
            'throughput': max_throughput,
        }

    def set_frozen_policy(self):
        self.frozen_policy = True
    
    def forcely_set_policy(self, plan: Dict[str, Any]):
        self.plan.speculative_num_steps = plan['speculative_num_steps']
        self.plan.speculative_eagle_topk = plan['speculative_eagle_topk']
        self.plan.speculative_num_draft_tokens = plan['speculative_num_draft_tokens']
        self.plan.speculative_num_steps_gpu = plan['speculative_num_steps_gpu']
        self.plan.speculative_eagle_topk_gpu = plan['speculative_eagle_topk_gpu']
        self.plan.speculative_num_draft_tokens_gpu = plan['speculative_num_draft_tokens_gpu']

    def change_plan(self, new_global_batchsize: int, now_generation_length: int, running_micro_batch_num: int = 0, log: bool = True) -> Tuple[bool, Dict[str, Any]]:
        if self.frozen_policy:
            return False, {
                "speculative_num_steps": self.plan.speculative_num_steps,
                "speculative_eagle_topk": self.plan.speculative_eagle_topk,
                "speculative_num_draft_tokens": self.plan.speculative_num_draft_tokens,
                "speculative_num_steps_gpu": self.plan.speculative_num_steps_gpu,
                "speculative_eagle_topk_gpu": self.plan.speculative_eagle_topk_gpu,
                "speculative_num_draft_tokens_gpu": self.plan.speculative_num_draft_tokens_gpu,
            }
        self.plan.global_batch_size = new_global_batchsize
        plan = self.new_solve_decode_stage2(
            stage_1_plan=self.stage_1_plan,
            now_generation_length=now_generation_length,
            during_running=True,
            running_micro_batch_num=running_micro_batch_num,
            log=log,
        )
        # 只会更改speculative相关参数
        # TODO: 其实如果使用CGCoop，也有可能更改gpu attention ratio，留到以后做
        # XXX: 有可能在尾段，max_speculative_token比原先大，有可能导致CPU KV cache分配出问题，所以必须保证新生成的策略的max_speculative_token数量不超过初始设定
        change = False
        if (
            plan["speculative_num_steps"] != self.plan.speculative_num_steps
            or plan["speculative_eagle_topk"] != self.plan.speculative_eagle_topk
            or plan["speculative_num_draft_tokens"] != self.plan.speculative_num_draft_tokens
            or plan["speculative_num_steps_gpu"] != self.plan.speculative_num_steps_gpu
            or plan["speculative_eagle_topk_gpu"] != self.plan.speculative_eagle_topk_gpu
            or plan["speculative_num_draft_tokens_gpu"] != self.plan.speculative_num_draft_tokens_gpu
        ):
            change = True
            assert plan['speculative_num_draft_tokens'] <= self.max_speculative_len, "speculative_num_draft_tokens > max_speculative_len"
            assert plan['speculative_num_draft_tokens_gpu'] <= self.max_speculative_len, "speculative_num_draft_tokens_gpu > max_speculative_len"
            if plan['speculative_eagle_topk'] * plan['speculative_num_steps'] > self.max_speculative_len:
                plan['speculative_eagle_topk'] = int(plan['speculative_num_draft_tokens'] ** 0.5)
                plan['speculative_num_steps'] = int(self.max_speculative_len / plan['speculative_eagle_topk'])
            if plan['speculative_eagle_topk_gpu'] * plan['speculative_num_steps_gpu'] > self.max_speculative_len:
                plan['speculative_eagle_topk_gpu'] = int(plan['speculative_num_draft_tokens_gpu'] ** 0.5)
                plan['speculative_num_steps_gpu'] = int(self.max_speculative_len / plan['speculative_eagle_topk_gpu'])
            self.plan.speculative_num_steps = plan['speculative_num_steps']
            self.plan.speculative_eagle_topk = plan['speculative_eagle_topk']
            self.plan.speculative_num_draft_tokens = plan['speculative_num_draft_tokens']
            self.plan.speculative_num_steps_gpu = plan['speculative_num_steps_gpu']
            self.plan.speculative_eagle_topk_gpu = plan['speculative_eagle_topk_gpu']
            self.plan.speculative_num_draft_tokens_gpu = plan['speculative_num_draft_tokens_gpu']
        return change, {
            "speculative_num_steps": self.plan.speculative_num_steps,
            "speculative_eagle_topk": self.plan.speculative_eagle_topk,
            "speculative_num_draft_tokens": self.plan.speculative_num_draft_tokens,
            "speculative_num_steps_gpu": self.plan.speculative_num_steps_gpu,
            "speculative_eagle_topk_gpu": self.plan.speculative_eagle_topk_gpu,
            "speculative_num_draft_tokens_gpu": self.plan.speculative_num_draft_tokens_gpu,
        }

    def solve_decode(self, now_generation_length: int) -> Dict[str, Any]:
        """计算decode阶段的执行计划"""
        decode_micro_batch_num_range = [2, 3, 4, 5]
        # decode_weight_cache_ratio_range = [i/8 for i in range(9)]
        decode_weight_cache_ratio_range = [0]
        # decode_gpu_attention_ratio_range = [0.0 + 0.05 * i for i in range(10)]
        decode_gpu_attention_ratio_range = [0]
        if self.draft_model_path is not None:
            draft_gpu_execution_ratio_range = [0.0 + 0.05 * i for i in range(20)] + [1]
        else:
            draft_gpu_execution_ratio_range = [0.0]
        if self.draft_model_path is not None:
            speculative_num_draft_tokens_cpu_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            speculative_num_draft_tokens_gpu_range = [i for i in range(1, 20)]
        else:
            speculative_num_draft_tokens_cpu_range = [1]
            speculative_num_draft_tokens_gpu_range = [1]

        max_throughput = 0
        best_plan = None

        policies = itertools.product(
            decode_micro_batch_num_range,
            decode_weight_cache_ratio_range,
            decode_gpu_attention_ratio_range,
            draft_gpu_execution_ratio_range,
            speculative_num_draft_tokens_cpu_range,
            speculative_num_draft_tokens_gpu_range,
        )
        for (
            decode_micro_batch_num,
            decode_weight_cache_ratio,
            decode_gpu_attention_ratio,
            draft_gpu_execution_ratio,
            speculative_num_draft_tokens_cpu,
            speculative_num_draft_tokens_gpu,
        ) in policies:
            if draft_gpu_execution_ratio < decode_gpu_attention_ratio: 
                continue
            if self.draft_model_path is None and speculative_num_draft_tokens_cpu != 1:
                continue
            if self.draft_model_path is None and speculative_num_draft_tokens_gpu != 1:
                continue
            if decode_gpu_attention_ratio == 0 and speculative_num_draft_tokens_gpu != 1: 
                continue
            if decode_gpu_attention_ratio != 0 and speculative_num_draft_tokens_gpu < speculative_num_draft_tokens_cpu:
                continue
            if draft_gpu_execution_ratio != 0 and speculative_num_draft_tokens_cpu == 1 and speculative_num_draft_tokens_gpu == 1:
                continue
            if not self.satisfy_gpu_memory([decode_micro_batch_num, decode_weight_cache_ratio, decode_gpu_attention_ratio, draft_gpu_execution_ratio, speculative_num_draft_tokens_cpu, speculative_num_draft_tokens_gpu], log=False):
                continue
            throughput = self.simulator.get_throughput((decode_micro_batch_num, decode_weight_cache_ratio, decode_gpu_attention_ratio, draft_gpu_execution_ratio, speculative_num_draft_tokens_cpu, speculative_num_draft_tokens_gpu), now_generation_length=now_generation_length, log=False)
            if throughput > max_throughput:
                max_throughput = throughput
                best_plan = (decode_micro_batch_num, decode_weight_cache_ratio, decode_gpu_attention_ratio, draft_gpu_execution_ratio, speculative_num_draft_tokens_cpu, speculative_num_draft_tokens_gpu)

        # print simulated basic information
        self.satisfy_gpu_memory(best_plan, log=True, record_cpu_kvcache_size=True)
        max_throughput = self.simulator.get_throughput(best_plan, now_generation_length=now_generation_length, log=True)

        return {
            'decode_micro_batch_num': best_plan[0],
            'decode_micro_batch_size': (self.plan.global_batch_size + best_plan[0] - 1) // best_plan[0],
            'decode_weight_cache_ratio': best_plan[1], 
            'decode_gpu_attention_ratio': best_plan[2],
            'draft_gpu_execution_ratio': best_plan[3],
            'speculative_num_steps': self.simulator.get_sd_policy_from_draftl(best_plan[4])[0],
            'speculative_eagle_topk': self.simulator.get_sd_policy_from_draftl(best_plan[4])[1],
            'speculative_num_draft_tokens': best_plan[4],
            'speculative_num_steps_gpu': self.simulator.get_sd_policy_from_draftl(best_plan[5])[0],
            'speculative_eagle_topk_gpu': self.simulator.get_sd_policy_from_draftl(best_plan[5])[1],
            'speculative_num_draft_tokens_gpu': best_plan[5],
            'decode_spec_policy': "SequentialCGCoop",
            'draft_model_placement': "CPU",
            'cpu_dram_for_kv_cache': self.cpu_dram_for_kv_cache,
            'throughput': max_throughput,
        }

    def solve(self) -> ExecutionPlan:
        """生成完整的执行计划"""
        prefill_plan = self.solve_prefill()
        # 设置prefill参数
        self.plan.global_batch_size = prefill_plan['global_batch_size']
        self.plan.prefill_micro_batch_size = prefill_plan['prefill_micro_batch_size']
        self.plan.prefill_micro_batch_num = prefill_plan['prefill_micro_batch_num']
        self.plan.prefill_weight_cache_ratio = prefill_plan['prefill_weight_cache_ratio']
        decode_plan = self.new_solve_decode()

        # 设置decode参数
        self.plan.decode_micro_batch_num = decode_plan['decode_micro_batch_num']
        self.plan.decode_micro_batch_size = decode_plan['decode_micro_batch_size']
        self.plan.decode_weight_cache_ratio = decode_plan['decode_weight_cache_ratio']
        self.plan.decode_gpu_attention_ratio = decode_plan['decode_gpu_attention_ratio']
        self.plan.decode_gpu_attention_micro_batch_size = decode_plan['decode_gpu_attention_micro_batch_size']
        self.plan.decode_gpu_attention_nano_batch_size = decode_plan['decode_gpu_attention_nano_batch_size']   
        self.plan.draft_gpu_execution_ratio = decode_plan['draft_gpu_execution_ratio']
        self.plan.speculative_num_steps = decode_plan['speculative_num_steps']
        self.plan.speculative_eagle_topk = decode_plan['speculative_eagle_topk']
        self.plan.speculative_num_draft_tokens = decode_plan['speculative_num_draft_tokens']
        self.plan.speculative_num_steps_gpu = decode_plan['speculative_num_steps_gpu']
        self.plan.speculative_eagle_topk_gpu = decode_plan['speculative_eagle_topk_gpu']
        self.plan.speculative_num_draft_tokens_gpu = decode_plan['speculative_num_draft_tokens_gpu']
        self.plan.decode_spec_policy = decode_plan['decode_spec_policy']
        self.plan.draft_model_placement = decode_plan['draft_model_placement']
        self.plan.cpu_dram_for_kv_cache = decode_plan['cpu_dram_for_kv_cache']
        self.plan.throughput = decode_plan['throughput']

        self.plan.target_cg_nano_kv_cache_slot = decode_plan['target_cg_nano_kv_cache_slot']
        self.plan.draft_kv_cache_slot = decode_plan['draft_kv_cache_slot']

        self.log_execution_plan(self.plan)
        return self.plan

    def log_execution_plan(self, plan: ExecutionPlan):
        """打印执行计划的详细信息"""
        logger.info("=" * 60)
        logger.info("生成的执行计划:")
        logger.info("=" * 60)
        logger.info("Prefill阶段:")
        logger.info(f"  global_batch_size: {plan.global_batch_size}")
        logger.info(f"  prefill_micro_batch_size: {plan.prefill_micro_batch_size}")
        logger.info(f"  prefill_micro_batch_num: {plan.prefill_micro_batch_num}")
        logger.info(f"  prefill_weight_cache_ratio: {plan.prefill_weight_cache_ratio:.3f}")
        logger.info("")
        logger.info("Decode阶段:")
        logger.info(f"  decode_micro_batch_num: {plan.decode_micro_batch_num}")
        logger.info(f"  decode_micro_batch_size: {plan.decode_micro_batch_size}")
        logger.info(f"  decode_weight_cache_ratio: {plan.decode_weight_cache_ratio:.3f}")
        logger.info(f"  decode_gpu_attention_ratio: {plan.decode_gpu_attention_ratio:.3f}")
        logger.info(f"  draft_gpu_execution_ratio: {plan.draft_gpu_execution_ratio:.3f}")
        logger.info(f"  decode_spec_policy: {plan.decode_spec_policy}")
        logger.info(f"  draft_model_placement: {plan.draft_model_placement}")
        logger.info("")
        logger.info("Speculative参数:")
        logger.info(f"  speculative_num_steps: {plan.speculative_num_steps}")
        logger.info(f"  speculative_eagle_topk: {plan.speculative_eagle_topk}")
        logger.info(f"  speculative_num_draft_tokens: {plan.speculative_num_draft_tokens}")
        logger.info(f"  speculative_num_steps_gpu: {plan.speculative_num_steps_gpu}")
        logger.info(f"  speculative_eagle_topk_gpu: {plan.speculative_eagle_topk_gpu}")
        logger.info(f"  speculative_num_draft_tokens_gpu: {plan.speculative_num_draft_tokens_gpu}")
        logger.info("Setting up")
        logger.info(f"  cpu_dram_for_kv_cache: {plan.cpu_dram_for_kv_cache:.3f} GB")
        logger.info(f"  target_cg_nano_kv_cache_slot: {plan.target_cg_nano_kv_cache_slot}")
        logger.info(f"  draft_kv_cache_slot: {plan.draft_kv_cache_slot}")
        logger.info("Result:")
        logger.info(f"  throughput: {plan.throughput:.3f}")
        logger.info("=" * 60)

    def get_real_target_kv_cache_size(self, sum_length: int, max_speculative_len: int = -1) -> float:
        if max_speculative_len == -1:
            max_speculative_len = self.max_speculative_len
        return self.target_kv_size_per_token_per_layer_gb * (sum_length + (self.max_output_length + max_speculative_len) * self.plan.global_batch_size) * self.target_model_layer_num


if __name__ == "__main__":
    # 示例用法
    ROOT = "/home/zzh/codes/specmoe"
    solver = Solver(
        target_model_path="/data1/zzh/huggingface/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/41bd4c9e7e4fb318ca40e721131d4933966c2cc1",
        draft_model_path="/data1/zzh/huggingface/hub/models--yuhuili--EAGLE-mixtral-instruct-8x7B/snapshots/f2e9cd1e1efaf0dec41c2da1b1fae4327727871d",
        average_seq_len=1205, 
        max_seq_len=2300,
        max_speculative_len=15,
        gpu_memory_gb=24,
        cpu_memory_gb=300,
        max_output_length=256,
        speculative_algorithm="EAGLE",
        tp_size=1,
        profile_root_path=f"{ROOT}/simulator_file",
        gpu_moe_fit_file_path=f"{ROOT}/simulator_file/fit_results/gpu_moe_fit.json",
        target_cpu_attn_fit_file_path=f"{ROOT}/simulator_file/fit_results/target_cpu_attn_fit.json", 
        target_gpu_attn_fit_file_path=f"{ROOT}/simulator_file/fit_results/target_gpu_attn_fit.json", 
        draft_cpu_attn_fit_file_path=f"{ROOT}/simulator_file/fit_results/draft_cpu_attn_fit.json", 
        draft_gpu_attn_fit_file_path=f"{ROOT}/simulator_file/fit_results/draft_gpu_attn_fit.json",
        htod_transfer_fit_file_path=f"{ROOT}/simulator_file/fit_results/htod_transfer_fit.json",
        log_level='info'
    )
