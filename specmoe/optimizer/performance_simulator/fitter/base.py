"""
参数拟合基类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import numpy as np
import json
import os
from datetime import datetime
from specmoe.optimizer.performance_simulator.models.base import BasePerformanceModel


class BaseFitter(ABC):
    """参数拟合基类"""
    
    def __init__(self, name: str, root_path: str, output_dir: str = "fit_results"):
        self.name = name
        self.output_dir = os.path.join(root_path, output_dir)
        self.fit_results = {}
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
    
    @abstractmethod
    def fit(self, model: BasePerformanceModel, profile_data: List[Dict[str, Any]], 
            **kwargs) -> Dict[str, Any]:
        """
        拟合模型参数
        
        Args:
            model: 性能模型
            profile_data: profile数据
            **kwargs: 拟合配置参数
            
        Returns:
            拟合结果字典
        """
        pass
    
    def evaluate_fit(self, model: BasePerformanceModel, 
                    profile_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        评估拟合效果
        
        Args:
            model: 已拟合的性能模型
            profile_data: 测试数据
            
        Returns:
            评估结果
        """
        if not model.is_fitted:
            raise ValueError("模型尚未拟合参数")
        
        predictions = []
        actual_values = []
        
        for data_point in profile_data:
            if data_point.get('latency') is None:
                continue  # 跳过无效数据
                
            input_params = data_point['input_params']
            actual_latency = data_point['latency']
            
            try:
                predicted_latency = model.predict(**input_params)
                predictions.append(predicted_latency)
                actual_values.append(actual_latency)
            except Exception as e:
                print(f"预测失败: {input_params}, 错误: {e}")
                continue
        
        if not predictions:
            return {'error': '没有有效的预测数据'}
        
        predictions = np.array(predictions)
        actual_values = np.array(actual_values)
        
        # 计算评估指标
        mse = np.mean((predictions - actual_values) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actual_values))
        
        # R²相关系数
        ss_res = np.sum((actual_values - predictions) ** 2)
        ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # 相对误差
        relative_errors = np.abs((predictions - actual_values) / actual_values)
        mean_relative_error = np.mean(relative_errors)
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mean_relative_error': float(mean_relative_error),
            'num_samples': len(predictions),
            'predictions': predictions.tolist(),
            'actual_values': actual_values.tolist(),
            'relative_errors': relative_errors.tolist()
        }
    
    def save_fit_results(self, model: BasePerformanceModel, 
                        fit_result: Dict[str, Any],
                        evaluation: Dict[str, Any] = None,
                        filename: str = None, 
                        filepath: str = None) -> str:
        """
        保存拟合结果
        
        Args:
            model: 拟合后的模型
            fit_result: 拟合结果
            evaluation: 评估结果 (可选)
            filename: 输出文件名 (可选)
            
        Returns:
            保存的文件路径
        """
        if filepath is None:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{model.name}_fit_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
        
        # 构造输出数据
        output_data = {
            'model_name': model.name,
            'model_description': model.get_model_description(),
            'fitter_name': self.name,
            'timestamp': datetime.now().isoformat(),
            'parameters': model.get_parameters(),
            'parameter_names': model.get_parameter_names(),
            'fit_result': fit_result,
            'evaluation': evaluation
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"拟合结果已保存到: {filepath}")
        return filepath
    
    def load_fit_results(self, filepath: str) -> Dict[str, Any]:
        """
        从文件加载拟合结果
        
        Args:
            filepath: 结果文件路径
            
        Returns:
            拟合结果字典
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.fit_results = data
        print(f"已加载拟合结果: {data.get('model_name', '未知模型')}")
        return data
    
    def prepare_training_data(self, profile_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备训练数据，过滤无效数据点
        
        Args:
            profile_data: profile数据
            
        Returns:
            (特征矩阵, 标签向量)
        """
        valid_data = []
        for data_point in profile_data:
            if (data_point.get('latency') is not None and 
                data_point.get('input_params') is not None and
                data_point['latency'] > 0):  # 过滤无效延迟
                valid_data.append(data_point)
        
        if not valid_data:
            raise ValueError("没有有效的训练数据")
        
        print(f"准备训练数据: {len(valid_data)}/{len(profile_data)} 个有效样本")
        return valid_data
    
    def cross_validate(self, model: BasePerformanceModel, 
                      profile_data: List[Dict[str, Any]],
                      k_folds: int = 5, **kwargs) -> Dict[str, Any]:
        """
        交叉验证
        
        Args:
            model: 性能模型
            profile_data: profile数据
            k_folds: 折数
            **kwargs: 拟合参数
            
        Returns:
            交叉验证结果
        """
        valid_data = self.prepare_training_data(profile_data)
        n_samples = len(valid_data)
        
        if n_samples < k_folds:
            raise ValueError(f"样本数量({n_samples})少于折数({k_folds})")
        
        # 随机打乱数据
        indices = np.random.permutation(n_samples)
        fold_size = n_samples // k_folds
        
        cv_results = []
        
        for fold in range(k_folds):
            # 分割训练集和验证集
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < k_folds - 1 else n_samples
            
            val_indices = indices[start_idx:end_idx]
            train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
            
            train_data = [valid_data[i] for i in train_indices]
            val_data = [valid_data[i] for i in val_indices]
            
            # 在训练集上拟合
            fold_model = type(model)(model.name)  # 创建新的模型实例
            fold_fit_result = self.fit(fold_model, train_data, **kwargs)
            
            # 在验证集上评估
            fold_evaluation = self.evaluate_fit(fold_model, val_data)
            
            cv_results.append({
                'fold': fold,
                'train_size': len(train_data),
                'val_size': len(val_data),
                'fit_result': fold_fit_result,
                'evaluation': fold_evaluation
            })
        
        # 汇总交叉验证结果
        cv_metrics = ['mse', 'rmse', 'mae', 'r2', 'mean_relative_error']
        cv_summary = {}
        
        for metric in cv_metrics:
            values = [fold['evaluation'][metric] for fold in cv_results if metric in fold['evaluation']]
            if values:
                cv_summary[f'{metric}_mean'] = float(np.mean(values))
                cv_summary[f'{metric}_std'] = float(np.std(values))
        
        return {
            'k_folds': k_folds,
            'total_samples': n_samples,
            'cv_results': cv_results,
            'cv_summary': cv_summary
        }