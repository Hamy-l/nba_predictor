"""
SVM (Support Vector Machine) 模型
用于NBA比赛胜负预测
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib


class SVMPredictor:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        """
        初始化SVM模型

        参数:
            kernel: 核函数类型 ('linear', 'rbf', 'poly', 'sigmoid')
            C: 正则化参数
            gamma: 核函数系数
        """
        self.model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42)
        self.scaler = StandardScaler()
        self.feature_columns = None

    def prepare_features(self, df):
        """准备特征列"""
        # 排除非特征列
        exclude_cols = ['season', 'season_type', 'date', 'game_id',
                       'h_team_name', 'o_team_name', 'h_W/L', 'o_W/L',
                       'label', 'home_win']

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols

    def train(self, data_path):
        """
        训练SVM模型

        参数:
            data_path: CSV数据文件路径
        """
        # 加载数据
        df = pd.read_csv(data_path)

        # 准备特征和标签
        self.feature_columns = self.prepare_features(df)
        X = df[self.feature_columns].values
        y = df['home_win'].values

        # 处理缺失值
        X = np.nan_to_num(X, nan=0.0)

        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 训练模型
        print("开始训练SVM模型...")
        self.model.fit(X_train_scaled, y_train)

        # 评估模型
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)

        print(f"训练集准确率: {train_score:.4f}")
        print(f"测试集准确率: {test_score:.4f}")

        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score
        }

    def predict(self, data):
        """
        预测比赛结果

        参数:
            data: DataFrame或ndarray，包含特征数据

        返回:
            predictions: 预测结果 (0: 客队胜, 1: 主队胜)
            probabilities: 预测概率
        """
        if isinstance(data, pd.DataFrame):
            X = data[self.feature_columns].values
        else:
            X = data

        # 处理缺失值
        X = np.nan_to_num(X, nan=0.0)

        # 标准化
        X_scaled = self.scaler.transform(X)

        # 预测
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        return predictions, probabilities

    def save_model(self, model_path='svm_model.pkl'):
        """保存模型"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, model_path)
        print(f"模型已保存到: {model_path}")

    def load_model(self, model_path='svm_model.pkl'):
        """加载模型"""
        saved = joblib.load(model_path)
        self.model = saved['model']
        self.scaler = saved['scaler']
        self.feature_columns = saved['feature_columns']
        print(f"模型已从 {model_path} 加载")


# 使用示例
if __name__ == "__main__":
    # 训练模型
    predictor = SVMPredictor(kernel='rbf', C=1.0)

    # 训练
    results = predictor.train('nba_team_boxscores_features_2015_16_to_2025_26.csv')

    # 保存模型
    predictor.save_model('trained_models/svm_model.pkl')

    # 预测示例
    # 加载测试数据
    test_df = pd.read_csv('nba_team_boxscores_features_2015_16_to_2025_26.csv').head(10)
    predictions, probabilities = predictor.predict(test_df)

    print("\n预测结果:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        home_team = test_df.iloc[i]['h_team_name']
        away_team = test_df.iloc[i]['o_team_name']
        winner = home_team if pred == 1 else away_team
        home_win_prob = prob[1] * 100
        away_win_prob = prob[0] * 100

        print(f"{home_team} vs {away_team}")
        print(f"  预测胜者: {winner}")
        print(f"  主队胜率: {home_win_prob:.2f}%")
        print(f"  客队胜率: {away_win_prob:.2f}%")
