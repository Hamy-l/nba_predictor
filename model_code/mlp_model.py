"""
MLP (多层感知机) 全连接神经网络模型
用于NBA比赛胜负预测
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib


class MLPNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32]):
        """
        MLP网络结构

        参数:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
        """
        super(MLPNetwork, self).__init__()

        layers = []
        prev_dim = input_dim

        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, 2))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MLPPredictor:
    def __init__(self, hidden_dims=[128, 64, 32], learning_rate=0.001):
        """
        初始化MLP模型

        参数:
            hidden_dims: 隐藏层维度列表
            learning_rate: 学习率
        """
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare_features(self, df):
        """准备特征列"""
        exclude_cols = ['season', 'season_type', 'date', 'game_id',
                       'h_team_name', 'o_team_name', 'h_W/L', 'o_W/L',
                       'label', 'home_win']

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols

    def train(self, data_path, epochs=50, batch_size=64):
        """
        训练MLP模型

        参数:
            data_path: CSV数据文件路径
            epochs: 训练轮数
            batch_size: 批次大小
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

        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_test_tensor = torch.LongTensor(y_test)

        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 初始化模型
        input_dim = X_train_scaled.shape[1]
        self.model = MLPNetwork(input_dim, self.hidden_dims).to(self.device)

        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # 训练模型
        print("开始训练MLP模型...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # 前向传播
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            train_acc = correct / total

            # 验证
            self.model.eval()
            with torch.no_grad():
                X_test_device = X_test_tensor.to(self.device)
                y_test_device = y_test_tensor.to(self.device)
                outputs = self.model(X_test_device)
                _, predicted = torch.max(outputs.data, 1)
                test_acc = (predicted == y_test_device).sum().item() / y_test_device.size(0)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, "
                      f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

        print(f"\n最终训练集准确率: {train_acc:.4f}")
        print(f"最终测试集准确率: {test_acc:.4f}")

        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc
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
        self.model.eval()

        if isinstance(data, pd.DataFrame):
            X = data[self.feature_columns].values
        else:
            X = data

        # 处理缺失值
        X = np.nan_to_num(X, nan=0.0)

        # 标准化
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        # 预测
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()

        return predictions, probabilities

    def save_model(self, model_path='mlp_model.pth'):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'hidden_dims': self.hidden_dims,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, model_path)
        print(f"模型已保存到: {model_path}")

    def load_model(self, model_path='mlp_model.pth'):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.hidden_dims = checkpoint['hidden_dims']
        self.scaler = checkpoint['scaler']
        self.feature_columns = checkpoint['feature_columns']

        input_dim = len(self.feature_columns)
        self.model = MLPNetwork(input_dim, self.hidden_dims).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"模型已从 {model_path} 加载")


# 使用示例
if __name__ == "__main__":
    # 训练模型
    predictor = MLPPredictor(hidden_dims=[128, 64, 32], learning_rate=0.001)

    # 训练
    results = predictor.train('nba_team_boxscores_features_2015_16_to_2025_26.csv',
                              epochs=50, batch_size=64)

    # 保存模型
    predictor.save_model('trained_models/mlp_model.pth')

    # 预测示例
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
