# Gold Price Forecasting Project

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-1.3+-orange.svg)](https://pandas.pydata.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive machine learning project for gold price forecasting using multiple statistical and deep learning approaches including ARIMA, LSTM neural networks, and ensemble methods.

## 🚀 Quick Start

```bash
# Clone or download the project
# Install dependencies
pip install pandas numpy matplotlib seaborn statsmodels tensorflow scikit-learn xgboost

# Run basic data inspection
python src/main.py

# Execute full forecasting pipeline
python gold_price_forecasting_pipeline.py
```

## 📊 Features

- **Multiple Forecasting Models**: ARIMA, LSTM, XGBoost, Random Forest, Ensemble Stacking
- **Multi-Timeframe Support**: Daily, hourly, 30-minute, 15-minute, 5-minute, 4-hour, weekly, monthly data
- **Comprehensive Evaluation**: MAE, MSE, RMSE, MAPE metrics with directional accuracy
- **Data Visualization**: Interactive plots for analysis and results
- **Modular Architecture**: Easy to extend with new models or data sources
- **Automated Setup**: Use `setup.py` for easy environment configuration

## 📁 Project Structure

```
├── src/
│   ├── currency_conversion.py    # Currency conversion utilities
│   └── main.py                   # Basic data loading and preprocessing
├── .continue/
│   └── rules/
│       └── CONTINUE.md           # Detailed project guide
├── __pycache__/
│   └── gold_price_forecasting_pipeline.cpython-311.pyc  # Compiled Python cache
├── Core Forecasting Scripts/
│   ├── gold_price_arima.py       # ARIMA model implementation
│   ├── gold_price_comparison.py  # ARIMA vs LSTM comparison
│   ├── gold_price_data_loader.py # Data loading utilities with inspection
│   ├── gold_price_forecasting_pipeline.py  # Complete forecasting pipeline
│   ├── gold_price_inference.py   # Inference utilities for models
│   ├── gold_price_metrics_only.py # Simplified ARIMA metrics
│   ├── gold-price-forecasting-using-lstm.py  # LSTM implementation
│   └── quant-directional-forecasting-xgb-rf-ensemble.py  # Ensemble methods
├── Jupyter Notebooks/
│   ├── crypto-technical-indicator-and-ml-prediction.ipynb
│   ├── gold-price-forecasting-using-lstm.ipynb
│   ├── gold-price-prediction-by-using-lstm-0d563d.ipynb
│   ├── gold-price-prediction-lstm-96-accuracy (2).ipynb
│   ├── introduction-to-time-series-arima-fbprophet-lstm.ipynb
│   ├── mitsui-co-commodity-prediction-challenge-lstm.ipynb
│   └── quant-directional-forecasting-xgb-rf-ensemble.ipynb
├── Utility Scripts/
│   ├── check_csv_columns.py      # CSV column inspection utility
│   ├── complete_forecast_function.py  # Forecasting function utilities
│   ├── test_file.py              # Test script
│   └── test_yfinance.py          # Yahoo Finance testing script
├── Data Files/
│   ├── XAU_15m_data.csv          # 15-minute gold price data
│   ├── XAU_1d_data.csv           # Daily gold price data
│   ├── XAU_1h_data.csv           # Hourly gold price data
│   ├── XAU_1Month_data.csv       # Monthly gold price data
│   ├── XAU_1m_data.csv           # 1-minute gold price data
│   ├── XAU_1w_data.csv           # Weekly gold price data
│   ├── XAU_30m_data.csv          # 30-minute gold price data
│   ├── XAU_4h_data.csv           # 4-hour gold price data
│   └── XAU_5m_data.csv           # 5-minute gold price data
├── Archives/
│   ├── archive (1).zip           # Backup archive 1
│   ├── archive (2).zip           # Backup archive 2
│   ├── archive (3).zip           # Backup archive 3
│   └── archive (4).zip           # Backup archive 4
├── Documentation/
│   ├── README.md                 # Project documentation
│   ├── TODO.md                   # Testing plan and progress
│   └── TODO_updated.md           # Updated testing plan
├── Configuration/
│   ├── requirements.txt          # Python dependencies
│   └── setup.py                  # Setup script
└── Miscellaneous/
    ├── v1.html                   # HTML version 1
    ├── v1.py                     # Python version 1
    └── v2.py                     # Python version 2
```

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Dependencies
```bash
pip install pandas numpy matplotlib seaborn statsmodels tensorflow scikit-learn xgboost
```

### Optional Dependencies (for enhanced functionality)
```bash
pip install jupyter  # For notebook-style execution
```

## 📖 Usage

### Basic Data Exploration
```bash
# Inspect and visualize data
python gold_price_data_loader.py
```

### Model Training
```bash
# ARIMA forecasting
python gold_price_arima.py

# LSTM forecasting
python gold-price-forecasting-using-lstm.py

# Model comparison
python gold_price_comparison.py

# Full pipeline with EDA
python gold_price_forecasting_pipeline.py

# Ensemble directional forecasting
python quant-directional-forecasting-xgb-rf-ensemble.py
```

### Testing
Follow the comprehensive testing plan in `TODO.md`:
- Test each script individually
- Verify data loading across different timeframes
- Validate model performance metrics
- Check visualization outputs

## 🎯 Models Overview

| Model | Type | Best For | Key Metrics |
|-------|------|----------|-------------|
| ARIMA | Statistical | Trend analysis | MSE, RMSE, MAPE |
| LSTM | Deep Learning | Complex patterns | MAE, directional accuracy |
| XGBoost | Ensemble | Feature-rich data | Accuracy, feature importance |
| Random Forest | Ensemble | Robust predictions | MAE, feature stability |
| Ensemble Stacking | Hybrid | Best performance | Combined metrics |

## 📈 Data Sources

The project uses XAU (Gold vs US Dollar) price data in various timeframes:
- **XAU_1d_data.csv**: Daily prices
- **XAU_1h_data.csv**: Hourly prices
- **XAU_30m_data.csv**: 30-minute intervals
- And more granular timeframes (15m, 5m, 4h, 1w, 1Month)

Data format: CSV with semicolon delimiter, columns include Open, High, Low, Close, Volume.

## 🔧 Development

### Adding New Models
1. Create a new Python script following the existing patterns
2. Implement data preprocessing, model training, and evaluation
3. Add visualization for results
4. Update TODO.md with testing requirements
5. Document in CONTINUE.md

### Code Standards
- Follow PEP 8 style guidelines
- Include comprehensive docstrings
- Add error handling and logging
- Test with multiple data timeframes

## 📊 Performance Metrics

Typical results (may vary by dataset and parameters):
- **ARIMA**: MAPE ~2-5%, good for short-term forecasts
- **LSTM**: MAPE ~1-3%, better for longer sequences
- **Ensemble**: Directional accuracy >70%, MAE <0.1% of price

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-model`)
3. Commit changes (`git commit -am 'Add new forecasting model'`)
4. Push to branch (`git push origin feature/new-model`)
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

- [Time Series Forecasting Guide](https://www.machinelearningplus.com/time-series/)
- [LSTM for Time Series](https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)
- [ARIMA Model Tutorial](https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/)

## 🙋 Support

For questions or issues:
1. Check the detailed guide in `.continue/rules/CONTINUE.md`
2. Review TODO.md for testing procedures
3. Examine individual script docstrings
4. Test with provided data files first

---

*Built with ❤️ for quantitative finance and machine learning research*
# jubilant-fortnight
