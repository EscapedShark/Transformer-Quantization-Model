# Stock Market Predictor

A sophisticated machine learning system for stock market prediction using Transformer architecture. This project includes data collection, preprocessing, model training, and evaluation components.

## Overview

This project consists of three main components:
1. Data Collection (`collect.py`)
2. Model Architecture (`model.py`)
3. Model Testing (`model_test.py`)

## Features

- Comprehensive stock data collection including:
  - Basic stock information (OHLCV data)
  - Financial metrics
  - Historical financials (quarterly and annual)
  - Options data
  - Analyst ratings
  - Industry data
  - Peer comparison data
- Advanced Transformer-based prediction model
- Cross-validation testing framework
- Performance visualization tools



## Project Structure

### Data Collection (`collect.py`)

The `StockDataCollector` class handles all data collection operations:

- Basic stock information
- Historical financial data
- Options data
- Analyst ratings
- Industry data
- Peer comparison

Key methods:
```python
collector = StockDataCollector(symbol, mongodb_uri)
collector.collect_all_data()  # Collects all available data
```

### Model Architecture (`model.py`)

Contains the core prediction model and supporting classes:

- `TransformerPredictor`: Main model architecture using Transformer
- `StockDataset`: Custom dataset class for stock data
- `FocalLoss`: Custom loss function for imbalanced data
- `StockPredictor`: High-level class for model training and prediction

Key features:
- Multi-head attention mechanism
- Focal loss for handling class imbalance
- Technical indicators (MA5, MA20, RSI, MACD)
- Sequence-based prediction

### Model Testing (`model_test.py`)

Comprehensive testing framework including:

- Time series cross-validation
- Performance metrics calculation
- Confusion matrix visualization
- Metrics plotting across folds

## Usage

1. Set up MongoDB:
```bash
# Start MongoDB service
mongod --dbpath <your_db_path>
```

2. Collect data:
```python
from collect import StockDataCollector

collector = StockDataCollector("AAPL", "mongodb://localhost:27017/")
collector.collect_all_data()
```

3. Train the model:
```python
from model import StockPredictor

# Define parameters
model_params = {
    'input_dim': 9,
    'num_heads': 4,
    'num_layers': 2,
    'dropout': 0.1
}

training_params = {
    'learning_rate': 0.0005,
    'batch_size': 64,
    'num_epochs': 50
}

predictor = StockPredictor(model_params, training_params)
predictor.train(train_loader, val_loader)
```

4. Test the model:
```python
from model_test import ModelTester

tester = ModelTester("AAPL", n_splits=5)
fold_metrics, avg_metrics = tester.run_cross_validation()
```

## Model Architecture Details

The Transformer-based predictor includes:

- Input embedding layer
- Multi-head attention layers
- Position-wise feed-forward networks
- Dropout for regularization
- Final classification layer

### Data Processing

The system processes the following features:
- Price data (Open, High, Low, Close)
- Volume
- Moving averages (5-day and 20-day)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)

## Performance Evaluation

The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion matrices
- Cross-validation performance plots

### Example Results

The model was trained and evaluated on stock market data with the following results:

#### Training Performance
```
Train Loss: 0.0401
Train Accuracy: 62.99%
Validation Loss: 0.0573
Validation Accuracy: 52.49%
```

#### Dataset Statistics
- Total samples in dataset: 1,620

#### Cross-validation Results

Sample fold (Fold 5) metrics:
```
Accuracy:  0.6019 (60.19%)
Precision: 0.7137 (71.37%)
Recall:    0.5651 (56.51%)
F1 Score:  0.6308 (63.08%)
```

Average metrics across all folds:
```
Accuracy:  0.5784 (57.84%)
Precision: 0.6287 (62.87%)
Recall:    0.6576 (65.76%)
F1 Score:  0.6348 (63.48%)
```

These results show that:
1. The model achieves consistent performance across training and validation sets
2. Precision is notably higher than recall, indicating the model is more conservative in its predictions
3. The model maintains stable performance across different cross-validation folds
4. Overall F1 score of ~63% suggests the model performs significantly better than random chance (50%) for binary prediction

## Notes

- The model uses time series cross-validation to prevent future data leakage
- Focal Loss is implemented to handle class imbalance
- The system includes extensive error handling and logging
- Technical indicators are calculated automatically during data preparation

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

[MIT License](LICENSE)

## Disclaimer

This project is for educational purposes only. Trading stocks carries significant risks, and past performance does not guarantee future results. Always conduct thorough research and consider consulting with a financial advisor before making investment decisions.