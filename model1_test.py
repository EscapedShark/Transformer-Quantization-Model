import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from model1 import StockPredictor, prepare_prediction_data, StockDataset, StockDataCollector

class ModelTester:
    def __init__(self, stock_symbol: str, n_splits: int = 5):
        """
        初始化模型测试器
        Args:
            stock_symbol: 股票代码
            n_splits: 交叉验证折数
        """
        self.stock_symbol = stock_symbol
        self.n_splits = n_splits
        self.collector = StockDataCollector(stock_symbol, "mongodb://localhost:27017/")
        
        # 模型参数
        self.model_params = {
            'input_dim': 9,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1
        }
        
        # 训练参数
        self.training_params = {
            'learning_rate': 0.0005,
            'batch_size': 64,
            'num_epochs': 50  # 可以根据需要调整
        }

    def prepare_cv_data(self) -> pd.DataFrame:
        """准备交叉验证数据"""
        return prepare_prediction_data(self.collector)

    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        评估预测结果
        Returns:
            包含各种评估指标的字典
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, fold: int):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - Fold {fold}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_fold_{fold}.png')
        plt.close()

    def run_cross_validation(self) -> Tuple[List[Dict], Dict]:
        """
        运行交叉验证
        Returns:
            每折的评估指标列表和平均评估指标
        """
        data = self.prepare_cv_data()
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        fold_metrics = []
        all_y_true = []
        all_y_pred = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(data), 1):
            print(f"\nFold {fold}/{self.n_splits}")
            
            # 分割数据
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            # 初始化预测器
            predictor = StockPredictor(self.model_params, self.training_params)
            
            # 准备数据加载器
            train_loader, val_loader = predictor.prepare_data(train_data)
            
            # 训练模型
            predictor.train(train_loader, val_loader)
            
            # 准备测试数据
            test_dataset = StockDataset(test_data)
            test_loader = torch.utils.data.DataLoader(
                test_dataset, 
                batch_size=self.training_params['batch_size']
            )
            
            # 收集预测结果
            y_true = []
            y_pred = []
            predictor.model.eval()
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X = batch_X.to(predictor.device)
                    outputs = predictor.model(batch_X)
                    _, predicted = outputs.max(1)
                    y_true.extend(batch_y.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())
            
            # 计算并存储评估指标
            metrics = self.evaluate_predictions(y_true, y_pred)
            fold_metrics.append(metrics)
            
            # 绘制混淆矩阵
            self.plot_confusion_matrix(y_true, y_pred, fold)
            
            # 保存所有预测结果
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            
            print(f"Fold {fold} metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
        
        # 计算平均指标
        avg_metrics = {
            metric: np.mean([fold[metric] for fold in fold_metrics])
            for metric in fold_metrics[0].keys()
        }
        
        # 绘制整体混淆矩阵
        self.plot_confusion_matrix(all_y_true, all_y_pred, fold='overall')
        
        return fold_metrics, avg_metrics

    def plot_metrics(self, fold_metrics: List[Dict]):
        """绘制评估指标的变化趋势"""
        metrics = list(fold_metrics[0].keys())
        folds = range(1, len(fold_metrics) + 1)
        
        plt.figure(figsize=(12, 6))
        for metric in metrics:
            values = [m[metric] for m in fold_metrics]
            plt.plot(folds, values, marker='o', label=metric)
        
        plt.title('Metrics Across Folds')
        plt.xlabel('Fold')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.savefig('metrics_across_folds.png')
        plt.close()

def main():
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 初始化测试器
    tester = ModelTester("AAPL", n_splits=5)
    
    # 运行交叉验证
    fold_metrics, avg_metrics = tester.run_cross_validation()
    
    # 绘制指标变化趋势
    tester.plot_metrics(fold_metrics)
    
    # 打印平均指标
    print("\nAverage metrics across all folds:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()