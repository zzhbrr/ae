"""
线性拟合器 - 使用最小二乘法拟合性能模型参数
"""

import numpy as np
from typing import Dict, Any, List
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from specmoe.optimizer.performance_simulator.models.base import BasePerformanceModel
from specmoe.optimizer.performance_simulator.fitter.base import BaseFitter


class LinearFitter(BaseFitter):
    """线性拟合器 - 基于最小二乘法"""
    
    def __init__(self, regularization: str = None, alpha: float = 1.0, 
                 normalize_features: bool = True, root_path: str = None, output_dir: str = "fit_results"):
        """
        初始化线性拟合器
        
        Args:
            regularization: 正则化类型 ('ridge', 'lasso', None)
            alpha: 正则化强度
            normalize_features: 是否标准化特征
            output_dir: 输出目录
        """
        super().__init__("LinearFitter", root_path, output_dir)
        self.regularization = regularization
        self.alpha = alpha
        self.normalize_features = normalize_features
        self.scaler = StandardScaler() if normalize_features else None
    
    def fit(self, model: BasePerformanceModel, profile_data: List[Dict[str, Any]], 
            **kwargs) -> Dict[str, Any]:
        """
        拟合模型参数
        
        Args:
            model: 性能模型
            profile_data: profile数据
            **kwargs: 额外参数
            
        Returns:
            拟合结果字典
        """
        print(f"开始拟合 {model.name} 模型参数...")
        
        # 准备训练数据
        valid_data = self.prepare_training_data(profile_data)
        
        # 构造特征矩阵和标签向量
        X, y = self._build_feature_matrix(model, valid_data)
        
        # 特征标准化
        if self.normalize_features:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # 选择回归器
        regressor = self._get_regressor()
        
        # 拟合模型
        regressor.fit(X_scaled, y)
        
        # 获取参数
        coefficients = regressor.coef_
        intercept = regressor.intercept_
        
        # 如果有特征标准化，需要转换回原始尺度
        if self.normalize_features:
            # 转换系数到原始尺度
            original_coefficients = coefficients / self.scaler.scale_
            original_intercept = intercept - np.sum(coefficients * self.scaler.mean_ / self.scaler.scale_)
        else:
            original_coefficients = coefficients
            original_intercept = intercept
        
        # 构造参数字典
        param_names = model.get_parameter_names()
        parameters = {}
        
        for i, param_name in enumerate(param_names):
            if i < len(original_coefficients):
                parameters[param_name] = float(original_coefficients[i])
        
        # 设置模型参数
        model.set_parameters(parameters)
        
        # 计算拟合质量指标
        y_pred = regressor.predict(X_scaled)
        mse = np.mean((y - y_pred) ** 2)
        r2 = regressor.score(X_scaled, y)
        
        fit_result = {
            'parameters': parameters,
            'parameter_names': param_names,
            'coefficients': original_coefficients.tolist(),
            'intercept': float(original_intercept),
            'training_mse': float(mse),
            'training_r2': float(r2),
            'num_samples': len(valid_data),
            'regularization': self.regularization,
            'alpha': self.alpha,
            'normalize_features': self.normalize_features,
            'feature_importance': self._calculate_feature_importance(original_coefficients, param_names)
        }
        
        print(f"拟合完成，R² = {r2:.4f}, MSE = {mse:.6f}")
        return fit_result
    
    def _build_feature_matrix(self, model: BasePerformanceModel, 
                             valid_data: List[Dict[str, Any]]) -> tuple:
        """构造特征矩阵和标签向量"""
        X = []
        y = []
        
        for data_point in valid_data:
            input_params = data_point['input_params']
            latency = data_point['latency']
            
            # 获取特征向量
            try:
                feature_vector = model.get_feature_vector(**input_params)
                X.append(feature_vector)
                y.append(latency)
            except Exception as e:
                print(f"构造特征向量失败: {input_params}, 错误: {e}")
                continue
        
        if not X:
            raise ValueError("无法构造有效的特征矩阵")
        
        return np.array(X), np.array(y)
    
    def _get_regressor(self):
        """获取回归器"""
        if self.regularization == 'ridge':
            return Ridge(alpha=self.alpha, fit_intercept=False)  # 特征向量已包含bias项
        elif self.regularization == 'lasso':
            return Lasso(alpha=self.alpha, fit_intercept=False, max_iter=2000)
        else:
            return LinearRegression(fit_intercept=False)  # 特征向量已包含bias项
    
    def _calculate_feature_importance(self, coefficients: np.ndarray, 
                                    param_names: List[str]) -> Dict[str, float]:
        """计算特征重要性"""
        # 使用系数的绝对值作为重要性指标
        abs_coefficients = np.abs(coefficients)
        total_importance = np.sum(abs_coefficients)
        
        importance = {}
        for i, param_name in enumerate(param_names):
            if i < len(abs_coefficients):
                importance[param_name] = float(abs_coefficients[i] / total_importance) if total_importance > 0 else 0
        
        return importance
    
    def fit_with_validation(self, model: BasePerformanceModel, 
                           profile_data: List[Dict[str, Any]],
                           validation_split: float = 0.2, **kwargs) -> Dict[str, Any]:
        """
        使用验证集的拟合
        
        Args:
            model: 性能模型
            profile_data: profile数据
            validation_split: 验证集比例
            **kwargs: 额外参数
            
        Returns:
            拟合结果字典
        """
        valid_data = self.prepare_training_data(profile_data)
        n_samples = len(valid_data)
        
        # 分割训练集和验证集
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)
        
        train_data = [valid_data[i] for i in indices[n_val:]]
        val_data = [valid_data[i] for i in indices[:n_val]]
        
        print(f"训练集: {len(train_data)} 样本, 验证集: {len(val_data)} 样本")
        
        # 在训练集上拟合
        fit_result = self.fit(model, train_data, **kwargs)
        
        # 在验证集上评估
        if val_data:
            validation_result = self.evaluate_fit(model, val_data)
            fit_result['validation'] = validation_result
            print(f"验证集 R² = {validation_result['r2']:.4f}")
        
        return fit_result
    
    def grid_search(self, model: BasePerformanceModel, 
                   profile_data: List[Dict[str, Any]],
                   param_grid: Dict[str, List], **kwargs) -> Dict[str, Any]:
        """
        网格搜索最佳超参数
        
        Args:
            model: 性能模型
            profile_data: profile数据
            param_grid: 参数网格，如 {'alpha': [0.1, 1.0, 10.0]}
            **kwargs: 额外参数
            
        Returns:
            网格搜索结果
        """
        print("开始网格搜索...")
        
        # 生成参数组合
        param_combinations = []
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        def generate_combinations(index, current_combo):
            if index == len(param_names):
                param_combinations.append(current_combo.copy())
                return
            
            param_name = param_names[index]
            for value in param_values[index]:
                current_combo[param_name] = value
                generate_combinations(index + 1, current_combo)
        
        generate_combinations(0, {})
        
        # 测试每种参数组合
        best_score = -np.inf
        best_params = None
        best_result = None
        all_results = []
        
        for params in param_combinations:
            print(f"测试参数: {params}")
            
            # 临时设置参数
            old_params = {
                'regularization': self.regularization,
                'alpha': self.alpha
            }
            
            for key, value in params.items():
                setattr(self, key, value)
            
            try:
                # 使用交叉验证评估
                cv_result = self.cross_validate(model, profile_data, k_folds=3, **kwargs)
                score = cv_result['cv_summary'].get('r2_mean', -np.inf)
                
                result = {
                    'params': params.copy(),
                    'cv_score': score,
                    'cv_result': cv_result
                }
                all_results.append(result)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_result = result
                
                print(f"交叉验证 R² = {score:.4f}")
                
            except Exception as e:
                print(f"参数组合失败: {params}, 错误: {e}")
            
            # 恢复原参数
            for key, value in old_params.items():
                setattr(self, key, value)
        
        # 使用最佳参数重新拟合
        if best_params:
            for key, value in best_params.items():
                setattr(self, key, value)
            
            final_fit_result = self.fit(model, profile_data, **kwargs)
            
            print(f"网格搜索完成，最佳参数: {best_params}, 最佳得分: {best_score:.4f}")
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'best_result': best_result,
                'all_results': all_results,
                'final_fit_result': final_fit_result
            }
        else:
            raise ValueError("网格搜索未找到有效参数组合")