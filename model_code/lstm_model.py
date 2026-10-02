"""
LSTM (长短期记忆网络) 时序模型
用于NBA比赛胜负预测
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib


class LSTMNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        """
        LSTM网络结构

        参数:
            input_dim: 输入特征维度
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            dropout: Dropout比例
        """
        super(LSTMNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        # LSTM层
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 取最后一个时间步的输出
        out = lstm_out[:, -1, :]

        # 全连接层
        out = self.fc(out)

        return out


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMPredictor:
    def __init__(self, sequence_length=5, hidden_dim=64, num_layers=2, learning_rate=0.001):
        """
        初始化LSTM模型

        参数:
            sequence_length: 时序窗口长度（使用最近N场比赛数据）
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            learning_rate: 学习率
        """
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
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

    def create_sequences(self, X, y):
        """
        创建时序序列数据

        参数:
            X: 特征数据
            y: 标签数据

        返回:
            X_seq: 时序特征 (samples, sequence_length, features)
            y_seq: 对应标签
        """
        X_seq, y_seq = [], []

        for i in range(len(X) - self.sequence_length + 1):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length - 1])

        return np.array(X_seq), np.array(y_seq)

    def train(self, data_path, epochs=50, batch_size=32):
        """
        训练LSTM模型

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

        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)

        # 创建时序序列
        X_seq, y_seq = self.create_sequences(X_scaled, y)

        print(f"时序数据形状: {X_seq.shape}")

        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.2, random_state=42
        )

        # 创建数据加载器
        train_dataset = TimeSeriesDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = TimeSeriesDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 初始化模型
        input_dim = X_seq.shape[2]
        self.model = LSTMNetwork(input_dim, self.hidden_dim, self.num_layers).to(self.device)

        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # 训练模型
        print("开始训练LSTM模型...")
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
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

            test_acc = correct / total

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
            data: DataFrame，包含时序特征数据（至少需要sequence_length条记录）

        返回:
            predictions: 预测结果
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

        # 创建序列
        if len(X_scaled) < self.sequence_length:
            raise ValueError(f"需要至少 {self.sequence_length} 条记录进行预测")

        X_seq, _ = self.create_sequences(X_scaled, np.zeros(len(X_scaled)))
        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        # 预测
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()

        return predictions, probabilities

    def save_model(self, model_path='lstm_model.pth'):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'sequence_length': self.sequence_length,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, model_path)
        print(f"模型已保存到: {model_path}")

    def load_model(self, model_path='lstm_model.pth'):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.sequence_length = checkpoint['sequence_length']
        self.hidden_dim = checkpoint['hidden_dim']
        self.num_layers = checkpoint['num_layers']
        self.scaler = checkpoint['scaler']
        self.feature_columns = checkpoint['feature_columns']

        input_dim = len(self.feature_columns)
        self.model = LSTMNetwork(input_dim, self.hidden_dim, self.num_layers).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"模型已从 {model_path} 加载")


# 使用示例
if __name__ == "__main__":
    # 训练模型
    predictor = LSTMPredictor(sequence_length=5, hidden_dim=64, num_layers=2)

    # 训练
    results = predictor.train('nba_team_boxscores_features_2015_16_to_2025_26.csv',
                              epochs=50, batch_size=32)

    # 保存模型
    predictor.save_model('trained_models/lstm_model.pth')

    # 预测示例（需要至少sequence_length条记录）
    test_df = pd.read_csv('nba_team_boxscores_features_2015_16_to_2025_26.csv').head(20)
    predictions, probabilities = predictor.predict(test_df)

    print("\n预测结果:")
    # 从第sequence_length条开始显示预测
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        idx = i + predictor.sequence_length - 1
        if idx < len(test_df):
            home_team = test_df.iloc[idx]['h_team_name']
            away_team = test_df.iloc[idx]['o_team_name']
            winner = home_team if pred == 1 else away_team
            home_win_prob = prob[1] * 100
            away_win_prob = prob[0] * 100

            print(f"{home_team} vs {away_team}")
            print(f"  预测胜者: {winner}")
            print(f"  主队胜率: {home_win_prob:.2f}%")
            print(f"  客队胜率: {away_win_prob:.2f}%")
