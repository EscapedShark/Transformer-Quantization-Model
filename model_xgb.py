import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient
import warnings
from datetime import datetime, timedelta
import traceback
import sys
from collect import StockDataCollector

# 忽略特定警告以保持输出整洁
# ignore specific warnings to keep output clean
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def safe_float_to_int(x):
    """安全地将浮点数转换为整数，处理NaN和无穷大"""
    if np.isnan(x) or np.isinf(x):
        return 0
    else:
        return int(x)

class StockDatasetXGB:
    """
    Dataset class for XGBoost model, preparing data from pandas DataFrame
    """
    def __init__(self, data: pd.DataFrame, sequence_length: int = 20, is_train: bool = True):
        self.sequence_length = sequence_length
        self.is_train = is_train
        
        # 创建数据的深拷贝以避免 SettingWithCopyWarning
        # create a deep copy to avoid SettingWithCopyWarning
        try:
            data = data.copy()
            
            # 快速检查是否有缺失值
            # quick check for missing values
            if data.isnull().any().any():
                print("Warning: Input data contains NaN values. Handling...")
                data = data.fillna(method='ffill').fillna(method='bfill')
            
            # 计算收益率和平滑收益率
            # calculate returns and smooth returns
            if 'returns' not in data.columns:
                data.loc[:, 'returns'] = data['Close'].pct_change()
            if 'smooth_returns' not in data.columns:
                data.loc[:, 'smooth_returns'] = data['returns'].rolling(window=5).mean()
            
            # 准备特征 - 使用简化高效的方式
            # prepare features - use simplified efficient way
            feature_cols = []
            feature_data = {}
            
            # 基础特征
            base_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20', 'RSI', 'MACD']
            
            # 添加基础技术指标特征，一次性创建所有切片
            for feature in base_features:
                # 获取原始序列
                orig_series = data[feature].ffill().bfill().values
                if np.isnan(orig_series).any():
                    print(f"Warning: NaN values in {feature} even after ffill/bfill")
                    orig_series = np.nan_to_num(orig_series)
                    
                # 创建滞后特征 (lag_1, lag_2, ..., lag_n)
                for i in range(1, min(sequence_length + 1, 10)):  # 限制最大滞后为10，减少复杂性
                    col_name = f"{feature}_lag_{i}"
                    lagged = np.zeros_like(orig_series)
                    lagged[i:] = orig_series[:-i]
                    feature_data[col_name] = lagged
                    feature_cols.append(col_name)
            
            # 添加几个简单的技术指标，避免复杂计算
            feature_data['Price_Range'] = data['High'].values - data['Low'].values
            feature_data['Price_Change'] = data['Close'].values - data['Open'].values
            feature_cols.extend(['Price_Range', 'Price_Change'])
            
            # 创建特征DataFrame
            feature_df = pd.DataFrame(feature_data, index=data.index)
            
            # 创建二元目标变量（涨或不涨）
            # create binary target variable (up or not up)
            threshold = 0.001  # 0.1%的阈值 threshold of 0.1%
            
            # 设置目标变量，确保没有NaN值
            target = np.where(data['smooth_returns'] > threshold, 1, 0)
            
            # 检查target中是否有NaN值
            if np.isnan(target).any():
                print("Warning: NaN in target. Replacing with 0.")
                target = np.nan_to_num(target, nan=0)
                
            feature_df['target'] = target
            
            # 移除包含NaN的行
            feature_df = feature_df.dropna()
            
            # 检查是否有足够的数据点
            if len(feature_df) < 10:  # 设置一个最小有效数据量
                raise ValueError(f"Insufficient data points after cleaning: only {len(feature_df)} rows left")
            
            # 提取特征和目标变量
            # extract features and target
            X_raw = feature_df[feature_cols].values
            y_raw = feature_df['target'].values
            
            # 清理X中的异常值
            # clean outliers in X
            X_cleaned = np.clip(X_raw, -1e6, 1e6)  # 剪裁极端值
            X_cleaned = np.nan_to_num(X_cleaned, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 清理y中的NaN并转换为整数
            # clean NaN in y and convert to integer
            y_cleaned = np.array([safe_float_to_int(val) for val in y_raw])
            
            # 应用标准化
            # apply normalization
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X_cleaned)
            
            # 最后一次检查NaN值
            # final check for NaN values
            if np.isnan(X_scaled).any():
                print("Warning: NaN values in X_scaled after processing. Replacing with 0.")
                X_scaled = np.nan_to_num(X_scaled)
                
            if np.isnan(y_cleaned).any():
                print("Warning: NaN values in y_cleaned after processing. Replacing with 0.")
                y_cleaned = np.nan_to_num(y_cleaned).astype(int)
            
            # 存储处理后的数据
            # store processed data
            self.X = X_scaled
            self.y = y_cleaned
            self.feature_names = feature_cols
            
            print(f"XGBoost dataset created with {len(self.X)} samples, {len(feature_cols)} features")
            
        except Exception as e:
            print(f"Error in StockDatasetXGB.__init__: {str(e)}")
            print(traceback.format_exc())
            # 创建空数据集以避免程序崩溃
            self.X = np.array([]).reshape(0, max(1, min(sequence_length, 10)) * len(base_features) + 2)
            self.y = np.array([])
            self.feature_names = []
            raise ValueError("Failed to create dataset")

    def get_data(self):
        """Return the prepared data"""
        # 最后一次确认没有NaN值
        if np.isnan(self.X).any():
            print("Warning: NaN values detected in X before returning. Fixing...")
            self.X = np.nan_to_num(self.X)
            
        if np.isnan(self.y).any():
            print("Warning: NaN values detected in y before returning. Fixing...")
            self.y = np.array([safe_float_to_int(val) for val in self.y])
            
        return self.X, self.y

class StockPredictorXGB:
    """
    XGBoost model for stock price movement prediction
    """
    def __init__(self, model_params: Dict, training_params: Dict):
        """
        Initialize the XGBoost predictor
        Args:
            model_params: XGBoost model parameters
            training_params: Training parameters
        """
        # 添加安全参数，防止处理异常值时出错
        # add safety parameters to prevent errors when handling outliers
        default_params = {
            'max_depth': 3,  # 更保守的树深度
            'learning_rate': 0.05,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',  # 更稳健的评估指标
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'alpha': 1.0,
            'lambda': 1.0,
            'gamma': 1.0,
            'min_child_weight': 2,
            'scale_pos_weight': 1,
            'missing': 0.0        # 明确指定如何处理缺失值
        }
        
        # 合并默认参数和用户提供的参数
        # merge default parameters with user-supplied ones
        self.model_params = {**default_params, **model_params}
        self.training_params = training_params
        self.model = None
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Train the XGBoost model
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        try:
            # 确保输入数据的完整性
            # ensure input data integrity
            if len(X_train) == 0 or len(y_train) == 0:
                raise ValueError("Empty training data")
                
            if X_val is not None and (len(X_val) == 0 or len(y_val) == 0):
                print("Warning: Empty validation data. Training without validation.")
                X_val, y_val = None, None
            
            # 再次检查和清洗数据
            # double check and clean data
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            y_train = np.array([safe_float_to_int(val) for val in y_train])
            
            if X_val is not None:
                X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
                y_val = np.array([safe_float_to_int(val) for val in y_val])
            
            # 创建DMatrix数据结构
            # create DMatrix data structure
            dtrain = xgb.DMatrix(X_train, label=y_train)
            
            # 开始训练模型
            # start training model
            training_args = {
                'params': self.model_params,
                'dtrain': dtrain,
                'num_boost_round': self.training_params.get('num_boost_round', 100),
                'verbose_eval': self.training_params.get('verbose_eval', 10)
            }
            
            # 如果提供了验证集，则使用早停
            # use early stopping if validation set is provided
            if X_val is not None and y_val is not None:
                dval = xgb.DMatrix(X_val, label=y_val)
                training_args.update({
                    'evals': [(dtrain, 'train'), (dval, 'validation')],
                    'early_stopping_rounds': self.training_params.get('early_stopping_rounds', 20)
                })
            
            try:
                # 训练模型
                self.model = xgb.train(**training_args)
                print("XGBoost model training completed successfully")
            except Exception as e:
                print(f"Error during XGBoost training: {str(e)}")
                print("Trying with more conservative parameters...")
                
                # 使用极其保守的参数重试
                # retry with extremely conservative parameters
                conservative_params = {
                    'max_depth': 2,
                    'learning_rate': 0.01,
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'subsample': 0.5,
                    'colsample_bytree': 0.5,
                    'alpha': 5.0,
                    'lambda': 5.0,
                    'gamma': 5.0,
                    'min_child_weight': 5,
                    'scale_pos_weight': 1,
                    'missing': 0.0
                }
                
                training_args['params'] = conservative_params
                training_args['num_boost_round'] = min(50, training_args['num_boost_round'])
                
                self.model = xgb.train(**training_args)
                print("XGBoost model training completed with conservative parameters")
                
            # 尝试打印特征重要性
            # try to print feature importance
            self._print_feature_importance()
            
        except Exception as e:
            print(f"Fatal error in train method: {str(e)}")
            print(traceback.format_exc())
            raise
    
    def _print_feature_importance(self):
        """Print feature importance"""
        if self.model is None:
            print("Model is not trained yet")
            return
        
        try:    
            importance = self.model.get_score(importance_type='gain')
            if not importance:
                print("No feature importance available")
                return
                
            # 绘制特征重要性图
            # plot feature importance
            plt.figure(figsize=(12, 8))
            importance_df = pd.DataFrame(
                {'Feature': list(importance.keys()), 
                 'Importance': list(importance.values())}
            )
            importance_df = importance_df.sort_values('Importance', ascending=False).head(20)
            
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title('XGBoost Feature Importance (Top 20)')
            plt.tight_layout()
            plt.savefig('xgb_feature_importance.png')
            plt.close()
            
            # 打印前10个重要特征
            # print top 10 important features
            print("Top 10 important features:")
            for i, (feature, score) in enumerate(sorted(importance.items(), 
                                                       key=lambda x: x[1], reverse=True)[:10]):
                print(f"{i+1}. {feature}: {score:.4f}")
        except Exception as e:
            print(f"Error printing feature importance: {str(e)}")
            print("Continuing with training process...")
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        Args:
            X: Features
        Returns:
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model is not trained yet")
        
        try:
            # 清理输入数据
            # clean input data
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            dtest = xgb.DMatrix(X)
            y_pred_prob = self.model.predict(dtest)
            y_pred = (y_pred_prob > 0.5).astype(int)
            return y_pred
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            # 返回全0预测作为备选
            return np.zeros(len(X), dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities
        Args:
            X: Features
        Returns:
            Predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model is not trained yet")
        
        try:
            # 清理输入数据
            # clean input data
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            dtest = xgb.DMatrix(X)
            return self.model.predict(dtest)
            
        except Exception as e:
            print(f"Error during probability prediction: {str(e)}")
            # 返回全0.5概率作为备选
            return np.ones(len(X)) * 0.5
        
    def save_model(self, filepath: str):
        """Save the model to a file"""
        if self.model is None:
            raise ValueError("No model to save")
            
        try:
            self.model.save_model(filepath)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
        
    def load_model(self, filepath: str):
        """Load the model from a file"""
        try:
            self.model = xgb.Booster()
            self.model.load_model(filepath)
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = None

def prepare_prediction_data_from_db(symbol: str, mongodb_uri="mongodb://localhost:27017/") -> pd.DataFrame:
    """
    Prepare price data required for prediction from MongoDB database
    """
    try:
        # 连接到MongoDB
        # connect to MongoDB
        client = MongoClient(mongodb_uri)
        db = client['stock_data']
        
        # 从基本信息集合中获取价格数据
        # get price data from basic_info collection
        basic_info = db['basic_info'].find_one({'symbol': symbol})
        
        if not basic_info or 'price_data' not in basic_info:
            raise ValueError(f"No price data found for {symbol} in the database.")
        
        # 将价格数据转换为DataFrame
        # convert price data to DataFrame
        price_data = pd.DataFrame(basic_info['price_data'])
        
        # 将Date列转换为datetime类型
        # convert Date column to datetime type
        if 'Date' in price_data.columns:
            price_data['Date'] = pd.to_datetime(price_data['Date'])
            price_data.set_index('Date', inplace=True)
        
        print(f"Successfully retrieved {len(price_data)} rows of historical data for {symbol} from database")
        
        # 清理数据 - 移除无穷大和NaN值
        # clean data - remove infinity and NaN values
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in price_data.columns:
                # 检查并替换无穷大值
                # check and replace infinity values
                price_data[col] = price_data[col].replace([np.inf, -np.inf], np.nan)
                
                # 使用前向填充处理NaN值 (使用推荐的方法)
                # use forward fill to handle NaN values (use recommended method)
                price_data[col] = price_data[col].ffill()
                
                # 剩余的NaN值（通常是开始的几行）使用后向填充
                # remaining NaN values (usually first few rows) use backward fill
                price_data[col] = price_data[col].bfill()
                
                # 检查是否还有NaN值
                if price_data[col].isnull().any():
                    print(f"Warning: Column {col} still has NaN values after filling")
                    price_data[col] = price_data[col].fillna(0)
        
        # 计算技术指标 calculate technical indicators
        price_data['MA5'] = price_data['Close'].rolling(window=5).mean().fillna(price_data['Close'])
        price_data['MA20'] = price_data['Close'].rolling(window=20).mean().fillna(price_data['Close'])
        price_data['RSI'] = calculate_rsi(price_data['Close'])
        price_data['MACD'] = calculate_macd(price_data['Close'])
        
        # 计算收益率和平滑收益率 calculate returns and smooth returns
        price_data['returns'] = price_data['Close'].pct_change().fillna(0)
        price_data['smooth_returns'] = price_data['returns'].rolling(window=5).mean().fillna(price_data['returns'])
        
        # 确保没有无穷大值和NaN值
        # ensure no infinity values and NaN values
        price_data = price_data.replace([np.inf, -np.inf], np.nan)
        price_data = price_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        print(f"Prepared data with {len(price_data)} rows after processing")
        
        return price_data
        
    except Exception as e:
        print(f"Error retrieving data from MongoDB: {str(e)}")
        print(traceback.format_exc())
        raise

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator with NaN handling"""
    try:
        # 检查输入是否有NaN
        if prices.isnull().any():
            prices = prices.fillna(method='ffill').fillna(method='bfill')
        
        delta = prices.diff().fillna(0)
        
        # 分离正负变化
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        # 计算平均值
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        # 计算相对强度
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)  # 避免除以零
        
        # 计算RSI
        rsi = 100 - (100 / (1 + rs))
        
        # 填充缺失值
        rsi = rsi.fillna(50)  # 使用中性值50填充NaN
        
        return rsi
        
    except Exception as e:
        print(f"Error calculating RSI: {str(e)}")
        # 返回全50作为备用
        return pd.Series(50, index=prices.index)

def calculate_macd(prices, slow=24, fast=12, signal=9):
    """Calculate MACD indicator with enhanced safety"""
    try:
        # 检查输入是否有NaN
        if prices.isnull().any():
            prices = prices.fillna(method='ffill').fillna(method='bfill')
        
        # 计算快速和慢速EMA
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        
        # 计算MACD线
        macd = exp1 - exp2
        
        # 计算信号线
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        
        # 计算柱状图
        histogram = macd - signal_line
        
        # 确保没有无穷大值和NaN值
        histogram = histogram.replace([np.inf, -np.inf], np.nan)
        histogram = histogram.fillna(0)
        
        return histogram
        
    except Exception as e:
        print(f"Error calculating MACD: {str(e)}")
        # 返回全零
        return pd.Series(0, index=prices.index)

class XGBTester:
    """
    Class to test the XGBoost model with cross-validation
    """
    def __init__(self, stock_symbol: str, n_splits: int = 5, mongodb_uri: str = "mongodb://localhost:27017/"):
        """
        Initialize the XGBoost model tester
        Args:
            stock_symbol: Stock symbol
            n_splits: Number of cross-validation splits
            mongodb_uri: MongoDB connection URI
        """
        self.stock_symbol = stock_symbol
        self.n_splits = n_splits
        self.mongodb_uri = mongodb_uri
        
        # XGBoost模型参数 - 使用保守设置
        # XGBoost model parameters - use conservative settings
        self.model_params = {
            'max_depth': 3, 
            'learning_rate': 0.05,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'alpha': 1.0,
            'lambda': 1.0,
            'gamma': 1.0,
            'min_child_weight': 2
        }
        
        # 训练参数
        # training parameters
        self.training_params = {
            'num_boost_round': 100,  # 减少以加快训练
            'early_stopping_rounds': 20,
            'verbose_eval': 10
        }
        
        # 序列长度参数
        # sequence length parameter
        self.sequence_length = 10  # 减少以简化模型
        
    def run_cross_validation(self) -> Tuple[List[Dict], Dict]:
        """
        Run cross-validation using TimeSeriesSplit
        Returns:
            List of metrics for each fold, and average metrics
        """
        try:
            # 准备数据
            # prepare data from MongoDB
            data = prepare_prediction_data_from_db(self.stock_symbol, self.mongodb_uri)
            
            # 确保数据足够进行交叉验证
            # ensure we have enough data for cross-validation
            if len(data) < self.n_splits * 2:
                raise ValueError(f"Not enough data points ({len(data)}) for {self.n_splits}-fold cross-validation. Need at least {self.n_splits*2} data points.")
            
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            
            fold_metrics = []
            all_y_true = []
            all_y_pred = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(data), 1):
                print(f"\n{'='*40}\nFold {fold}/{self.n_splits}\n{'='*40}")
                
                try:
                    # 分割数据 split data
                    train_data = data.iloc[train_idx]
                    test_data = data.iloc[test_idx]
                    
                    # 划分验证集
                    # create validation set
                    val_size = int(0.2 * len(train_data))
                    if val_size > 0:
                        val_data = train_data.iloc[-val_size:]
                        train_data = train_data.iloc[:-val_size]
                    else:
                        val_data = None
                    
                    print(f"Train size: {len(train_data)}, Val size: {val_size if val_size > 0 else 0}, Test size: {len(test_data)}")
                    
                    # 准备数据集
                    # prepare datasets
                    train_dataset = StockDatasetXGB(train_data, sequence_length=self.sequence_length)
                    X_train, y_train = train_dataset.get_data()
                    
                    # 验证数据集准备
                    if val_size > 0 and val_data is not None:
                        val_dataset = StockDatasetXGB(val_data, sequence_length=self.sequence_length)
                        X_val, y_val = val_dataset.get_data()
                    else:
                        X_val, y_val = None, None
                    
                    # 测试数据集准备
                    test_dataset = StockDatasetXGB(test_data, sequence_length=self.sequence_length, is_train=False)
                    X_test, y_test = test_dataset.get_data()
                    
                    # 检查数据
                    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
                    if X_val is not None:
                        print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
                    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
                    
                    # 检查是否有足够的训练和测试数据
                    if len(X_train) == 0 or len(y_train) == 0 or len(X_test) == 0 or len(y_test) == 0:
                        print(f"Warning: Insufficient data in fold {fold}. Skipping.")
                        continue
                    
                    # 初始化并训练模型
                    # initialize and train the model
                    predictor = StockPredictorXGB(self.model_params, self.training_params)
                    predictor.train(X_train, y_train, X_val, y_val)
                    
                    # 预测和评估
                    # prediction and evaluation
                    y_pred = predictor.predict(X_test)
                    
                    # 检查y_test和y_pred长度
                    # check y_test and y_pred length
                    if len(y_test) == 0 or len(y_pred) == 0:
                        print(f"Warning: Empty prediction results in fold {fold}. Skipping this fold.")
                        continue
                    
                    # 计算指标
                    # calculate metrics
                    metrics = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred, zero_division=0),
                        'recall': recall_score(y_test, y_pred, zero_division=0),
                        'f1': f1_score(y_test, y_pred, zero_division=0)
                    }
                    
                    fold_metrics.append(metrics)
                    
                    # 保存预测结果
                    # save prediction results
                    all_y_true.extend(y_test)
                    all_y_pred.extend(y_pred)
                    
                    # 打印当前折的指标
                    # print metrics for current fold
                    print(f"Fold {fold} metrics:")
                    for metric, value in metrics.items():
                        print(f"{metric}: {value:.4f}")
                        
                    # 保存当前折的模型
                    # save the model for current fold
                    predictor.save_model(f"xgb_model_fold_{fold}.json")
                    
                    # 绘制混淆矩阵
                    # plot confusion matrix
                    self.plot_confusion_matrix(y_test, y_pred, fold)
                    
                except Exception as e:
                    print(f"Error in fold {fold}: {str(e)}")
                    print(traceback.format_exc())
                    print("Skipping this fold and continuing with next fold...")
                    continue
            
            if not fold_metrics:
                raise ValueError("No valid folds completed. Cannot calculate average metrics.")
                
            # 计算平均指标
            # calculate average metrics
            avg_metrics = {
                metric: np.mean([fold[metric] for fold in fold_metrics])
                for metric in fold_metrics[0].keys()
            }
            
            # 绘制整体混淆矩阵
            # plot overall confusion matrix
            if all_y_true and all_y_pred:
                self.plot_confusion_matrix(all_y_true, all_y_pred, fold='overall')
            
            # 绘制指标变化趋势
            # plot the trend of metrics
            self.plot_metrics(fold_metrics)
            
            return fold_metrics, avg_metrics
            
        except Exception as e:
            print(f"Error in run_cross_validation: {str(e)}")
            print(traceback.format_exc())
            raise
            
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, fold):
        """Plot confusion matrix"""
        try:
            from sklearn.metrics import confusion_matrix
            
            # 确保输入数据有效
            if len(y_true) == 0 or len(y_pred) == 0:
                print(f"Warning: Empty data for confusion matrix in fold {fold}")
                return
                
            # 确保没有NaN值
            if np.isnan(y_true).any() or np.isnan(y_pred).any():
                print(f"Warning: NaN values in confusion matrix data for fold {fold}")
                y_true = np.nan_to_num(y_true, nan=0).astype(int)
                y_pred = np.nan_to_num(y_pred, nan=0).astype(int)
            
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'XGBoost Confusion Matrix - Fold {fold}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(f'xgb_confusion_matrix_fold_{fold}.png')
            plt.close()
            
        except Exception as e:
            print(f"Error plotting confusion matrix: {str(e)}")
    
    def plot_metrics(self, fold_metrics: List[Dict]):
        """Plot the trend of evaluation metrics"""
        try:
            if not fold_metrics:
                print("Warning: No metrics to plot")
                return
                
            metrics = list(fold_metrics[0].keys())
            folds = range(1, len(fold_metrics) + 1)
            
            plt.figure(figsize=(12, 6))
            for metric in metrics:
                values = [m[metric] for m in fold_metrics]
                plt.plot(folds, values, marker='o', label=metric)
            
            plt.title('XGBoost Metrics Across Folds')
            plt.xlabel('Fold')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True)
            plt.savefig('xgb_metrics_across_folds.png')
            plt.close()
            
        except Exception as e:
            print(f"Error plotting metrics: {str(e)}")
        
    def run_feature_importance_analysis(self):
        """Run feature importance analysis on the full dataset"""
        try:
            # 准备全部数据
            # prepare full dataset from MongoDB
            data = prepare_prediction_data_from_db(self.stock_symbol, self.mongodb_uri)
            
            # 使用较小的序列长度简化模型
            dataset = StockDatasetXGB(data, sequence_length=self.sequence_length)
            X, y = dataset.get_data()
            
            if len(X) == 0 or len(y) == 0:
                print("Warning: Empty dataset for feature importance analysis")
                return None
            
            # 训练模型
            # train model
            predictor = StockPredictorXGB(self.model_params, self.training_params)
            predictor.train(X, y)
            
            # 保存模型
            # save model
            predictor.save_model(f"xgb_model_{self.stock_symbol}_full.json")
            
            return predictor
            
        except Exception as e:
            print(f"Error in feature importance analysis: {str(e)}")
            print(traceback.format_exc())
            return None

def get_available_stock_symbols_from_db(mongodb_uri="mongodb://localhost:27017/"):
    """Get a list of stock symbols available in the MongoDB database"""
    try:
        # 连接到MongoDB
        # connect to MongoDB
        client = MongoClient(mongodb_uri)
        db = client['stock_data']
        
        # 从basic_info集合中查找所有可用的股票代码
        # find all available stock symbols from basic_info collection
        symbols = db['basic_info'].distinct('symbol')
        
        if not symbols:
            print("Warning: No stock symbols found in the database.")
            return ["AAPL"]  # 默认返回AAPL作为备用
            
        print(f"Found {len(symbols)} stock symbols in database: {', '.join(symbols)}")
        return symbols
        
    except Exception as e:
        print(f"Error connecting to MongoDB: {str(e)}")
        print("Returning default symbol 'AAPL'")
        return ["AAPL"]

def run_simple_prediction(stock_symbol, mongodb_uri):
    """
    Run a simple prediction without cross-validation
    """
    print(f"\n{'='*50}")
    print(f"Running simple prediction for {stock_symbol}")
    print(f"{'='*50}\n")
    
    try:
        # 获取数据
        data = prepare_prediction_data_from_db(stock_symbol, mongodb_uri)
        
        if len(data) <= 100:
            print(f"Insufficient data points ({len(data)}) for {stock_symbol}.")
            return
            
        # 使用简单的训练/测试分割
        train_size = int(0.8 * len(data))
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        # 准备数据
        train_dataset = StockDatasetXGB(train_data, sequence_length=10)
        test_dataset = StockDatasetXGB(test_data, sequence_length=10, is_train=False)
        
        X_train, y_train = train_dataset.get_data()
        X_test, y_test = test_dataset.get_data()
        
        # 检查数据有效性
        if len(X_train) == 0 or len(y_train) == 0 or len(X_test) == 0 or len(y_test) == 0:
            print("Error: Invalid dataset after preparation")
            return
        
        # 确保数据中没有NaN或无穷大
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 确保目标变量是整数
        y_train = np.array([safe_float_to_int(val) for val in y_train])
        y_test = np.array([safe_float_to_int(val) for val in y_test])
        
        # 超级保守的模型参数
        model_params = {
            'max_depth': 2,
            'learning_rate': 0.01,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'subsample': 0.5,
            'colsample_bytree': 0.5,
            'alpha': 5.0,
            'lambda': 5.0,
            'gamma': 5.0,
            'min_child_weight': 5
        }
        
        training_params = {
            'num_boost_round': 50,
            'verbose_eval': 10
        }
        
        # 训练模型
        predictor = StockPredictorXGB(model_params, training_params)
        predictor.train(X_train, y_train)
        
        # 评估模型
        y_pred = predictor.predict(X_test)
        
        # 确保预测结果有效
        if len(y_pred) == 0:
            print("Error: Empty prediction results")
            return
        
        # 打印结果
        from sklearn.metrics import accuracy_score, classification_report
        print("\nModel Evaluation Results:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # 保存模型
        predictor.save_model(f"xgb_model_{stock_symbol}_simple.json")
        print(f"Simple model saved as 'xgb_model_{stock_symbol}_simple.json'")
        
    except Exception as e:
        print(f"Error in simple prediction: {str(e)}")
        print(traceback.format_exc())

def main():
    """Main function"""
    # 设置随机种子
    # set random seed
    np.random.seed(42)
    
    print("\n" + "="*70)
    print("STOCK PRICE MOVEMENT PREDICTION WITH XGBOOST".center(70))
    print("="*70)
    
    # MongoDB连接URI
    mongodb_uri = "mongodb://localhost:27017/"
    
    # 从数据库获取可用的股票符号
    # get available stock symbols from database
    available_symbols = get_available_stock_symbols_from_db(mongodb_uri)
    
    if not available_symbols:
        print("Error: No available stock symbols found in the database.")
        return
        
    # 使用第一个可用的股票符号
    # use the first available stock symbol
    stock_symbol = available_symbols[0]
    print(f"Selected {stock_symbol} for model testing")
    
    # 首先尝试简单预测，避免交叉验证问题
    # first try simple prediction to avoid cross-validation issues
    run_simple_prediction(stock_symbol, mongodb_uri)
    
    print(f"\nStarting XGBoost cross-validation testing for {stock_symbol}...")
    
    # 捕获所有异常以确保脚本不会崩溃
    # catch all exceptions to ensure script won't crash
    try:
        # 初始化测试器
        # initialize tester
        tester = XGBTester(stock_symbol, n_splits=5, mongodb_uri=mongodb_uri)
        
        # 先确认有足够的数据
        # first confirm we have enough data
        data = prepare_prediction_data_from_db(stock_symbol, mongodb_uri)
        if len(data) < 10:  # 设置一个合理的最小数据量
            raise ValueError(f"Insufficient data points ({len(data)}) for {stock_symbol}. Need more data for cross-validation.")
        
        # 根据数据量调整折数
        # adjust number of folds based on data size
        n_splits = min(5, len(data) // 50)  # 每折至少需要50个样本
        if n_splits < 2:
            print(f"Too few data points for cross-validation. Using 2 folds.")
            n_splits = 2
            
        if n_splits != tester.n_splits:
            print(f"Adjusting number of folds from {tester.n_splits} to {n_splits} based on data size")
            tester.n_splits = n_splits
        
        # 运行交叉验证
        # run cross-validation
        fold_metrics, avg_metrics = tester.run_cross_validation()
        
        # 输出平均指标
        # print average metrics
        print("\nAverage metrics across all folds:")
        for metric, value in avg_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # 分析特征重要性
        # analyze feature importance
        print("\nRunning feature importance analysis...")
        tester.run_feature_importance_analysis()
        
        print("XGBoost model testing completed!")
        
    except Exception as e:
        print(f"\nError occurred during model testing: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)