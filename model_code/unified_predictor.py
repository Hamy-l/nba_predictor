"""
统一预测接口
用于调用所有训练好的模型进行预测
"""

import pandas as pd
import numpy as np
import os
import sys

# 导入所有模型
from decision_tree_model import DecisionTreePredictor
from svm_model import SVMPredictor
from random_forest_model import RandomForestPredictor
from mlp_model import MLPPredictor
from lstm_model import LSTMPredictor
from gru_model import GRUPredictor


class UnifiedPredictor:
    """统一预测器，支持加载和调用所有模型"""

    def __init__(self):
        self.models = {}
        self.model_types = {
            'decision_tree': DecisionTreePredictor,
            'svm': SVMPredictor,
            'random_forest': RandomForestPredictor,
            'mlp': MLPPredictor,
            'lstm': LSTMPredictor,
            'gru': GRUPredictor
        }

    def load_model(self, model_name, model_path):
        """
        加载指定模型

        参数:
            model_name: 模型名称 ('decision_tree', 'svm', 'random_forest', 'mlp', 'lstm', 'gru')
            model_path: 模型文件路径
        """
        if model_name not in self.model_types:
            raise ValueError(f"不支持的模型类型: {model_name}")

        predictor_class = self.model_types[model_name]
        predictor = predictor_class()

        if os.path.exists(model_path):
            predictor.load_model(model_path)
            self.models[model_name] = predictor
            print(f"成功加载模型: {model_name}")
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

    def load_all_models(self, model_dir='trained_models'):
        """
        加载所有可用模型

        参数:
            model_dir: 模型文件目录
        """
        model_files = {
            'decision_tree': os.path.join(model_dir, 'decision_tree_model.pkl'),
            'svm': os.path.join(model_dir, 'svm_model.pkl'),
            'random_forest': os.path.join(model_dir, 'random_forest_model.pkl'),
            'mlp': os.path.join(model_dir, 'mlp_model.pth'),
            'lstm': os.path.join(model_dir, 'lstm_model.pth'),
            'gru': os.path.join(model_dir, 'gru_model.pth')
        }

        for model_name, model_path in model_files.items():
            try:
                self.load_model(model_name, model_path)
            except Exception as e:
                print(f"加载模型 {model_name} 失败: {str(e)}")

    def predict_single_model(self, model_name, data):
        """
        使用单个模型进行预测

        参数:
            model_name: 模型名称
            data: DataFrame，输入数据

        返回:
            predictions: 预测结果
            probabilities: 预测概率
        """
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 未加载")

        return self.models[model_name].predict(data)

    def predict_all_models(self, data):
        """
        使用所有已加载模型进行预测

        参数:
            data: DataFrame，输入数据

        返回:
            results: 字典，包含所有模型的预测结果
        """
        results = {}

        for model_name, predictor in self.models.items():
            try:
                predictions, probabilities = predictor.predict(data)
                results[model_name] = {
                    'predictions': predictions,
                    'probabilities': probabilities
                }
            except Exception as e:
                print(f"模型 {model_name} 预测失败: {str(e)}")
                results[model_name] = None

        return results

    def predict_with_ensemble(self, data, method='voting'):
        """
        使用集成方法进行预测

        参数:
            data: DataFrame，输入数据
            method: 集成方法 ('voting' 或 'averaging')

        返回:
            ensemble_predictions: 集成预测结果
            ensemble_probabilities: 集成预测概率
            individual_results: 各模型的预测结果
        """
        # 获取所有模型的预测
        all_results = self.predict_all_models(data)

        # 过滤掉失败的预测
        valid_results = {k: v for k, v in all_results.items() if v is not None}

        if not valid_results:
            raise ValueError("没有可用的预测结果")

        # 提取预测和概率
        all_predictions = []
        all_probabilities = []

        for model_name, result in valid_results.items():
            all_predictions.append(result['predictions'])
            all_probabilities.append(result['probabilities'])

        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)

        # 集成预测
        if method == 'voting':
            # 投票法：选择多数模型预测的类别
            ensemble_predictions = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(),
                axis=0,
                arr=all_predictions
            )
        elif method == 'averaging':
            # 平均法：对概率取平均后选择类别
            ensemble_probabilities = np.mean(all_probabilities, axis=0)
            ensemble_predictions = np.argmax(ensemble_probabilities, axis=1)
        else:
            raise ValueError(f"不支持的集成方法: {method}")

        # 计算集成概率（无论哪种方法都使用平均）
        ensemble_probabilities = np.mean(all_probabilities, axis=0)

        return ensemble_predictions, ensemble_probabilities, valid_results


def format_prediction_results(data, predictions, probabilities, model_name="模型"):
    """
    格式化预测结果

    参数:
        data: 原始数据DataFrame
        predictions: 预测结果
        probabilities: 预测概率
        model_name: 模型名称
    """
    print(f"\n{'='*60}")
    print(f"{model_name} 预测结果")
    print(f"{'='*60}\n")

    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        if i >= len(data):
            break

        home_team = data.iloc[i]['h_team_name']
        away_team = data.iloc[i]['o_team_name']
        actual_result = data.iloc[i]['home_win']

        winner = home_team if pred == 1 else away_team
        home_win_prob = prob[1] * 100
        away_win_prob = prob[0] * 100

        # 判断预测是否正确
        is_correct = (pred == actual_result)
        status = "✓" if is_correct else "✗"

        print(f"比赛 {i+1}: {home_team} (主) vs {away_team} (客)")
        print(f"  预测胜者: {winner} [{status}]")
        print(f"  主队胜率: {home_win_prob:.2f}%")
        print(f"  客队胜率: {away_win_prob:.2f}%")
        print()


# 使用示例
if __name__ == "__main__":
    # 创建统一预测器
    predictor = UnifiedPredictor()

    # 加载所有模型
    print("加载所有模型...")
    predictor.load_all_models('trained_models')

    # 加载测试数据
    print("\n加载测试数据...")
    test_df = pd.read_csv('nba_team_boxscores_features_2015_16_to_2025_26.csv').head(20)

    # 方案1：使用单个模型预测
    print("\n=== 方案1：单模型预测 ===")
    if 'random_forest' in predictor.models:
        predictions, probabilities = predictor.predict_single_model('random_forest', test_df)
        format_prediction_results(test_df, predictions, probabilities, "随机森林")

    # 方案2：使用所有模型预测并对比
    print("\n=== 方案2：所有模型预测对比 ===")
    all_results = predictor.predict_all_models(test_df)

    model_names_cn = {
        'decision_tree': '决策树',
        'svm': 'SVM',
        'random_forest': '随机森林',
        'mlp': 'MLP神经网络',
        'lstm': 'LSTM',
        'gru': 'GRU'
    }

    for model_name, result in all_results.items():
        if result is not None:
            format_prediction_results(
                test_df,
                result['predictions'][:10],  # 只显示前10条
                result['probabilities'][:10],
                model_names_cn.get(model_name, model_name)
            )

    # 方案3：集成预测
    print("\n=== 方案3：集成预测（投票法）===")
    try:
        ensemble_pred, ensemble_prob, individual = predictor.predict_with_ensemble(
            test_df, method='voting'
        )
        format_prediction_results(test_df, ensemble_pred, ensemble_prob, "集成模型（投票）")
    except Exception as e:
        print(f"集成预测失败: {str(e)}")

    # 计算各模型准确率
    print("\n=== 模型准确率对比 ===")
    print(f"{'模型':<20} {'准确率':<10}")
    print("-" * 30)

    actual = test_df['home_win'].values

    for model_name, result in all_results.items():
        if result is not None:
            predictions = result['predictions']
            # 对于时序模型，需要截取对应长度
            min_len = min(len(predictions), len(actual))
            accuracy = np.mean(predictions[:min_len] == actual[:min_len])
            model_name_cn = model_names_cn.get(model_name, model_name)
            print(f"{model_name_cn:<20} {accuracy*100:>6.2f}%")

    # 集成模型准确率
    if 'ensemble_pred' in locals():
        min_len = min(len(ensemble_pred), len(actual))
        ensemble_accuracy = np.mean(ensemble_pred[:min_len] == actual[:min_len])
        print(f"{'集成模型（投票）':<20} {ensemble_accuracy*100:>6.2f}%")
