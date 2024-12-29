import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from typing import Dict, List, Tuple
from collect import StockDataCollector

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

class StockDataset(Dataset):
    def __init__(self, data: pd.DataFrame, sequence_length: int = 20):
        self.sequence_length = sequence_length
        
        # 创建数据的深拷贝以避免 SettingWithCopyWarning
        data = data.copy()
        
        # 计算收益率和平滑收益率
        data.loc[:, 'returns'] = data['Close'].pct_change()
        data.loc[:, 'smooth_returns'] = data['returns'].rolling(window=5).mean()
        
        # 准备特征
        features = []
        features.extend(['Open', 'High', 'Low', 'Close', 'Volume'])
        features.extend(['MA5', 'MA20', 'RSI', 'MACD'])
        
        # 标准化数据
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(data[features].values)
        
        # 创建序列数据
        self.X = []
        self.y = []
        
        # 设置阈值
        threshold = 0.001  # 0.1%的阈值
        
        for i in range(len(scaled_data) - sequence_length):
            # 确保有足够的数据并且smooth_returns不是NaN
            if i + sequence_length < len(data) and pd.notna(data['smooth_returns'].iloc[i + sequence_length]):
                return_value = data['smooth_returns'].iloc[i + sequence_length]
                # 只有当收益率超过阈值时才添加样本
                if abs(return_value) >= threshold:
                    self.X.append(scaled_data[i:(i + sequence_length)])
                    self.y.append(1 if return_value > 0 else 0)
        
        if len(self.X) == 0:
            raise ValueError("No valid sequences could be created from the data")
            
        # 转换为张量
        self.X = torch.FloatTensor(np.array(self.X))
        self.y = torch.LongTensor(np.array(self.y))
        
        print(f"Dataset created with {len(self.X)} samples")  # 添加调试信息

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TransformerPredictor(nn.Module):
    def __init__(self, input_dim: int, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        """
        初始化Transformer模型
        Args:
            input_dim: 输入特征维度
            num_heads: 注意力头数
            num_layers: Transformer层数
            dropout: Dropout率
        """
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, 64)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入数据，shape: (batch_size, seq_len, input_dim)
        Returns:
            预测结果
        """
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # 对序列取平均
        return self.fc(x)

class StockPredictor:
    def __init__(self, model_params: Dict, training_params: Dict):
        """
        初始化预测器
        Args:
            model_params: 模型参数
            training_params: 训练参数
        """
        self.model = TransformerPredictor(**model_params)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 使用Focal Loss
        self.criterion = FocalLoss(alpha=0.25, gamma=2)
        self.optimizer = optim.Adam(self.model.parameters(), lr=training_params['learning_rate'])
        
        self.batch_size = training_params['batch_size']
        self.num_epochs = training_params['num_epochs']

    def prepare_data(self, data: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
        """
        准备训练和验证数据
        """
        # 分割训练集和验证集
        train_size = int(0.8 * len(data))
        train_data = data[:train_size]
        val_data = data[train_size:]
        
        # 创建数据集
        train_dataset = StockDataset(train_data)
        val_dataset = StockDataset(val_data)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        return train_loader, val_loader

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        训练模型
        """
        # 检查数据加载器
        if len(train_loader) == 0:
            raise ValueError("Training data loader is empty")
        if len(val_loader) == 0:
            raise ValueError("Validation data loader is empty")
        
        print(f"Training with {len(train_loader)} batches per epoch")
        print(f"Validation with {len(val_loader)} batches per epoch")
    
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
        
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
            
                loss.backward()
                self.optimizer.step()
            
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += batch_y.size(0)
                train_correct += predicted.eq(batch_y).sum().item()
            
                if (batch_idx + 1) % 10 == 0:  # 每10个批次打印一次进度
                    print(f'Epoch [{epoch+1}/{self.num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] '
                        f'Loss: {loss.item():.4f}')
        
            # 验证
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
        
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += batch_y.size(0)
                    val_correct += predicted.eq(batch_y).sum().item()
        
            print(f'Epoch [{epoch+1}/{self.num_epochs}]')
            print(f'Train Loss: {train_loss/len(train_loader):.4f}, '
                f'Train Acc: {100.*train_correct/train_total:.2f}%')
            print(f'Val Loss: {val_loss/len(val_loader):.4f}, '
                f'Val Acc: {100.*val_correct/val_total:.2f}%')

    def predict(self, data: torch.Tensor) -> np.ndarray:
        """
        预测
        """
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            outputs = self.model(data)
            _, predicted = outputs.max(1)
            return predicted.cpu().numpy()

def prepare_prediction_data(collector: StockDataCollector) -> pd.DataFrame:
    """
    准备预测所需的价格数据
    """
    # 获取历史数据
    hist_data = collector.stock.history(period="max")
    
    # 计算技术指标
    hist_data.loc[:, 'MA5'] = hist_data['Close'].rolling(window=5).mean()
    hist_data.loc[:, 'MA20'] = hist_data['Close'].rolling(window=20).mean()
    hist_data.loc[:, 'RSI'] = calculate_rsi(hist_data['Close'])
    hist_data.loc[:, 'MACD'] = calculate_macd(hist_data['Close'])
    
    # 计算收益率和平滑收益率
    hist_data.loc[:, 'returns'] = hist_data['Close'].pct_change()
    hist_data.loc[:, 'smooth_returns'] = hist_data['returns'].rolling(window=5).mean()
    
    # 删除包含NaN的行
    hist_data = hist_data.dropna()
    
    print(f"Prepared data with {len(hist_data)} rows")  # 添加调试信息
    
    return hist_data

def calculate_rsi(prices, period=14):
    """计算RSI指标"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, slow=24, fast=12, signal=9):  # 调整MACD参数
    """计算MACD指标"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line

# 使用示例
def main():
    # 初始化数据收集器
    collector = StockDataCollector("AAPL", "mongodb://localhost:27017/")
    
    # 准备数据
    data = prepare_prediction_data(collector)
    
    # 模型参数
    model_params = {
        'input_dim': 9,  # 5个价格特征 + 4个技术指标
        'num_heads': 4,
        'num_layers': 2,
        'dropout': 0.1
    }
    
    # 训练参数
    training_params = {
        'learning_rate': 0.0005,  # 降低学习率
        'batch_size': 64,        # 增加批次大小
        'num_epochs': 100        # 增加训练轮数
    }
    
    # 初始化预测器
    predictor = StockPredictor(model_params, training_params)
    
    # 准备数据加载器
    train_loader, val_loader = predictor.prepare_data(data)
    
    # 训练模型
    predictor.train(train_loader, val_loader)

    # 保存模型
    torch.save(predictor.model.state_dict(), 'transformer_model.pth')

if __name__ == "__main__":
    main()