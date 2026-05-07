"""
分段线性拟合器 - 用于Roofline模型的自适应断点拟合
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from specmoe.optimizer.performance_simulator.models.base import BasePerformanceModel
from specmoe.optimizer.performance_simulator.fitter.base import BaseFitter
import logging

logger = logging.getLogger(__name__)

class PiecewiseLinearFitter(BaseFitter):
    """分段线性拟合器 - 用于Roofline模型的自适应断点拟合"""
    
    def __init__(self, min_samples_per_segment: int = 5, 
                 breakpoint_search_method: str = "exhaustive",
                 relative_error_weight: float = 0.5,
                 root_path: str = None,
                 output_dir: str = "fit_results"):
        """
        初始化分段线性拟合器
        
        Args:
            min_samples_per_segment: 每段最少样本数
            breakpoint_search_method: 断点搜索方法 ("exhaustive", "adaptive", "percentile")
            relative_error_weight: 相对误差权重，用于平衡两段的重要性
            output_dir: 输出目录
        """
        super().__init__("PiecewiseLinearFitter", root_path, output_dir)
        self.min_samples_per_segment = min_samples_per_segment
        self.breakpoint_search_method = breakpoint_search_method
        self.relative_error_weight = relative_error_weight
        
    def fit(self, model: BasePerformanceModel, profile_data: List[Dict[str, Any]], 
            **kwargs) -> Dict[str, Any]:
        """
        拟合分段线性模型
        
        Args:
            model: 性能模型
            profile_data: profile数据
            **kwargs: 额外参数
            
        Returns:
            拟合结果字典
        """
        logger.debug(f"开始分段线性拟合 {model.name} 模型参数...")
        
        # 准备训练数据
        valid_data = self.prepare_training_data(profile_data)
        
        # 提取token数量和延迟
        token_counts = []
        latencies = []
        
        for data_point in valid_data:
            token_counts.append(data_point['input_params']['num_tokens'])
            latencies.append(data_point['latency'])
        
        token_counts = np.array(token_counts)
        latencies = np.array(latencies)
        
        # 按token数量排序
        sorted_indices = np.argsort(token_counts)
        token_counts = token_counts[sorted_indices]
        latencies = latencies[sorted_indices]
        
        # 搜索最优断点
        best_breakpoint, best_score, all_results = self._search_optimal_breakpoint(
            token_counts, latencies, model
        )
        
        # 使用最优断点进行最终拟合
        final_result = self._fit_piecewise_model(
            token_counts, latencies, model, best_breakpoint
        )
        
        # 设置模型参数
        model.set_parameters(final_result['parameters'])
        
        # 构造完整的拟合结果
        fit_result = {
            'best_breakpoint': best_breakpoint,
            'best_score': best_score,
            'parameters': final_result['parameters'],
            'left_segment': final_result['left_segment'],
            'right_segment': final_result['right_segment'],
            'training_metrics': final_result['metrics'],
            'breakpoint_search_results': all_results,
            'num_samples': len(valid_data),
            'token_range': [float(token_counts.min()), float(token_counts.max())],
            'search_method': self.breakpoint_search_method
        }
        
        logger.debug(f"分段拟合完成，最优断点: {best_breakpoint}, 总体得分: {best_score:.4f}")
        return fit_result
    
    def _generate_breakpoint_candidates(self, token_counts: np.ndarray) -> List[int]:
        """生成断点候选"""
        min_tokens = token_counts.min()
        max_tokens = token_counts.max()
        
        if self.breakpoint_search_method == "exhaustive":
            # 穷举法：在有效范围内的每个可能token数
            unique_tokens = np.unique(token_counts)
            # 确保两段都有足够的样本
            valid_candidates = []
            for token in unique_tokens:
                left_count = np.sum(token_counts <= token)
                right_count = np.sum(token_counts > token)
                if (left_count >= self.min_samples_per_segment and 
                    right_count >= self.min_samples_per_segment):
                    valid_candidates.append(int(token))
            return valid_candidates
            
        elif self.breakpoint_search_method == "percentile":
            # 百分位法：使用数据的百分位数作为候选
            percentiles = np.arange(20, 81, 10)  # 20%, 30%, ..., 80%
            candidates = []
            for p in percentiles:
                candidate = int(np.percentile(token_counts, p))
                left_count = np.sum(token_counts <= candidate)
                right_count = np.sum(token_counts > candidate)
                if (left_count >= self.min_samples_per_segment and 
                    right_count >= self.min_samples_per_segment):
                    candidates.append(candidate)
            return candidates
            
        elif self.breakpoint_search_method == "adaptive":
            # 自适应法：基于数据分布特征
            # 使用K-means聚类的思想找到自然分割点
            from sklearn.cluster import KMeans
            
            # 对token数量进行聚类
            kmeans = KMeans(n_clusters=2, random_state=42)
            clusters = kmeans.fit_predict(token_counts.reshape(-1, 1))
            
            # 找到聚类边界附近的候选点
            boundary_region = []
            for i in range(len(token_counts) - 1):
                if clusters[i] != clusters[i + 1]:
                    boundary_region.extend([token_counts[i], token_counts[i + 1]])
            
            if boundary_region:
                # 在边界区域生成候选
                min_boundary = min(boundary_region)
                max_boundary = max(boundary_region)
                candidates = []
                for token in np.unique(token_counts):
                    if min_boundary <= token <= max_boundary:
                        left_count = np.sum(token_counts <= token)
                        right_count = np.sum(token_counts > token)
                        if (left_count >= self.min_samples_per_segment and 
                            right_count >= self.min_samples_per_segment):
                            candidates.append(int(token))
                return candidates
            else:
                # 回退到百分位法
                return self._generate_breakpoint_candidates(token_counts)
        
        return []
    
    def _search_optimal_breakpoint(self, token_counts: np.ndarray, 
                                  latencies: np.ndarray, 
                                  model: BasePerformanceModel) -> Tuple[int, float, List[Dict]]:
        """搜索最优断点"""
        candidates = self._generate_breakpoint_candidates(token_counts)
        
        if not candidates:
            raise ValueError("无法生成有效的断点候选")
        
        logger.debug(f"生成 {len(candidates)} 个断点候选: {candidates}")
        
        best_breakpoint = None
        best_score = -np.inf
        all_results = []
        
        for breakpoint in candidates:
            try:
                result = self._fit_piecewise_model(token_counts, latencies, model, breakpoint)
                score = result['overall_score']
                
                all_results.append({
                    'breakpoint': breakpoint,
                    'score': score,
                    'metrics': result['metrics'],
                    'left_segment': result['left_segment'],
                    'right_segment': result['right_segment']
                })
                
                if score > best_score:
                    best_score = score
                    best_breakpoint = breakpoint
                
                logger.debug(f"  断点 {breakpoint}: 得分 {score:.4f}")
                
            except Exception as e:
                logger.debug(f"  断点 {breakpoint}: 拟合失败 - {e}")
                continue
        
        if best_breakpoint is None:
            raise ValueError("所有断点候选都拟合失败")
        
        return best_breakpoint, best_score, all_results
    
    def _fit_piecewise_model(self, token_counts: np.ndarray, 
                            latencies: np.ndarray,
                            model: BasePerformanceModel,
                            breakpoint: int) -> Dict[str, Any]:
        """对给定断点拟合分段模型"""
        # 分割数据
        left_mask = token_counts <= breakpoint
        right_mask = token_counts > breakpoint
        
        left_tokens = token_counts[left_mask]
        left_latencies = latencies[left_mask]
        right_tokens = token_counts[right_mask]
        right_latencies = latencies[right_mask]
        
        if len(left_tokens) < 2 or len(right_tokens) < 2:
            raise ValueError("某段数据点过少，无法拟合")
        
        # 拟合左段 (memory bound)
        left_features = self._get_memory_features(left_tokens, model)
        left_reg = LinearRegression()
        left_reg.fit(left_features, left_latencies)
        
        # 拟合右段 (compute bound)  
        right_features = self._get_compute_features(right_tokens, model)
        right_reg = LinearRegression()
        right_reg.fit(right_features, right_latencies)
        
        # 计算预测值
        left_pred = left_reg.predict(left_features)
        right_pred = right_reg.predict(right_features)
        
        # 计算评估指标
        left_metrics = self._calculate_segment_metrics(left_latencies, left_pred, "left")
        right_metrics = self._calculate_segment_metrics(right_latencies, right_pred, "right")
        
        # 计算整体得分（使用最大相对误差作为评价指标）
        left_weight = len(left_tokens) / len(token_counts)
        right_weight = len(right_tokens) / len(token_counts)
        
        # 计算所有样本的相对误差
        all_relative_errors = []
        
        # 添加左段的相对误差
        left_relative_errors = np.abs((left_pred - left_latencies) / left_latencies)
        all_relative_errors.extend(left_relative_errors)
        
        # 添加右段的相对误差
        right_relative_errors = np.abs((right_pred - right_latencies) / right_latencies)
        all_relative_errors.extend(right_relative_errors)
        
        # 计算最大相对误差（越小越好）
        max_relative_error = np.max(all_relative_errors)
        
        # 将最大相对误差转换为得分（越小的误差对应越高的得分）
        # 使用 1 / (1 + max_error) 确保得分在 (0, 1] 范围内
        overall_score = 1.0 / (1.0 + max_relative_error)
        
        # 构造参数字典
        h = getattr(model, 'hidden_size', 4096)
        h_ff = getattr(model, 'expert_size', 16384)
        
        # 从线性回归结果提取参数
        # 左段: T = a1 × N + b1
        # 特征是 [N, 1]，系数是 [a1, b1]
        a1 = left_reg.coef_[0] if len(left_reg.coef_) > 0 else 0
        b1 = left_reg.intercept_
        
        # 右段: T = a2 × N + b2  
        # 特征是 [N, 1]，系数是 [a2, b2]
        a2 = right_reg.coef_[0] if len(right_reg.coef_) > 0 else 0
        b2 = right_reg.intercept_
        
        parameters = {
            'a1': float(a1),
            'b1': float(b1),
            'a2': float(a2), 
            'b2': float(b2),
            'breakpoint': int(breakpoint)
        }
        
        return {
            'parameters': parameters,
            'left_segment': {
                'coefficients': left_reg.coef_.tolist(),
                'intercept': float(left_reg.intercept_),
                'metrics': left_metrics,
                'sample_count': len(left_tokens)
            },
            'right_segment': {
                'coefficients': right_reg.coef_.tolist(), 
                'intercept': float(right_reg.intercept_),
                'metrics': right_metrics,
                'sample_count': len(right_tokens)
            },
            'overall_score': overall_score,
            'metrics': {
                'left_r2': left_metrics['r2'],
                'right_r2': right_metrics['r2'],
                'left_relative_error': left_metrics['mean_relative_error'],
                'right_relative_error': right_metrics['mean_relative_error'],
                'max_relative_error': max_relative_error,
                'overall_score': overall_score
            }
        }
    
    def _get_memory_features(self, tokens: np.ndarray, model: BasePerformanceModel) -> np.ndarray:
        """获取左段（小token数）的特征"""
        # 简化的线性特征: [num_tokens, 1]
        features = np.column_stack([
            tokens,                      # 线性项
            np.ones(len(tokens))         # 常数项
        ])
        
        return features
    
    def _get_compute_features(self, tokens: np.ndarray, model: BasePerformanceModel) -> np.ndarray:
        """获取右段（大token数）的特征"""
        # 简化的线性特征: [num_tokens, 1]
        features = np.column_stack([
            tokens,                      # 线性项
            np.ones(len(tokens))         # 常数项
        ])
        
        return features
    
    def _calculate_segment_metrics(self, actual: np.ndarray, 
                                  predicted: np.ndarray, 
                                  segment_name: str) -> Dict[str, float]:
        """计算段的评估指标"""
        mse = mean_squared_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        
        # 相对误差
        relative_errors = np.abs((predicted - actual) / actual)
        mean_relative_error = np.mean(relative_errors)
        
        return {
            'mse': float(mse),
            'r2': float(r2),
            'mean_relative_error': float(mean_relative_error),
            'segment': segment_name
        }
