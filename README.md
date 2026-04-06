Forex & Multi‑Asset Analysis – Trend, Volatility, and Structure
This project builds a simple but realistic market analysis and prediction tool for three different markets:

EURUSD (major FX pair)
XAUUSD (gold, via GC=F futures proxy)
BTCUSD (Bitcoin)
Instead of relying on random indicators, the project focuses on three pillars: trend, volatility, and market structure, and uses them to train basic models that try to predict the direction of the next candle.

Project Goals
Download and prepare historical OHLC data for EURUSD, XAUUSD, and BTCUSD.
Engineer features that reflect trend, volatility, and structure.
Train a supervised model to answer:
“Will the next candle close higher than the current one?”


Compare behavior and model performance across FX, gold, and crypto.
Show clearly why short‑term direction prediction is difficult and why understanding the market structure matters.
Data
Historical price data is downloaded from Yahoo Finance using yfinance:

EURUSD: EURUSD=X
Gold (XAUUSD proxy): GC=F
Bitcoin: BTC-USD
Each dataset provides Open, High, Low, Close, Volume on a daily timeframe (this can be extended to other timeframes with different data sources).

Feature Engineering
For each asset, the pipeline builds the same set of features:

1. Trend
Moving averages: MA20, MA50, MA100 of the closing price.
Slopes of moving averages: MA20_slope, MA50_slope.
Simple trend flag: trend_up = 1 if MA20 > MA50, else 0.
2. Volatility
Daily return: ret = Close.pct_change().
Rolling volatility: vol_20 = std(ret, window=20).
Average True Range (ATR14): based on High–Low, High–PrevClose, Low–PrevClose.
3. Market Structure
Rolling support and resistance using a 50‑period window:
rolling_low = min(Low, window=50)
rolling_high = max(High, window=50)
Position in the range:
pos_in_range = (Close - rolling_low) / (rolling_high - rolling_low)
Clipped to [0, 1].
Simple structure zoning:
structure_zone = 1 near support (pos_in_range < 0.2)
structure_zone = -1 near resistance (pos_in_range > 0.8)
structure_zone = 0 in the middle of the range
Prediction Target
For each asset, the model predicts the direction of the next candle:

Next close: close_next = Close.shift(-1)
Future return: future_ret_1 = close_next / Close - 1
Binary label:
target_up = 1 if future_ret_1 > 0
target_up = 0 otherwise
This directly encodes: “Will the next candle close higher than the current one?”

Modeling
Features: trend, volatility, and structure variables:
ma_20, ma_50, ma_100
ma_20_slope, ma_50_slope
vol_20, atr_14
pos_in_range, structure_zone
Model: RandomForestClassifier from scikit‑learn.
Split: time‑series aware train/test split with shuffle=False to avoid look‑ahead bias.
For each asset, the code:

Builds the features.
Splits into train/test in time order.
Trains a Random Forest with class_weight='balanced'.
Evaluates accuracy, precision, recall, F1, and confusion matrix.
Plots:
Feature importance bar chart.
Price chart with MA20, MA50, and support/resistance zones.
Results & Comparison
EURUSD:

Smoother trends, moderate volatility.
Accuracy for next‑day direction is close to 50%, which is expected in an efficient FX pair.
XAUUSD (Gold):

More spiky, driven by macro/risk sentiment.
Volatility (ATR) often ranks high in importance.
Short‑term direction remains hard to predict reliably.
BTCUSD:

Highest volatility and largest swings.
Volatility and range position are critical features.
Prediction accuracy may deviate slightly from 50%, but simple directional models are still fragile given the size of crypto moves.
Key takeaway:
Short‑term direction prediction from daily candles is challenging across all three markets. The value of this project is in building a clear, reusable framework: data → trend/volatility/structure features → supervised model → visual + quantitative evaluation, and in comparing how different asset classes behave under the same logic.

Files
forex_multi_asset_analysis.ipynb – Main Jupyter/Colab notebook with data download, feature engineering, modeling, and visualizations.
How to Run
Clone the repository:
git clone <your-repo-url>
Install dependencies (for example):
pandas
numpy
yfinance
scikit-learn
matplotlib
seaborn
Open the notebook in Jupyter or Google Colab.
Run cells from top to bottom.
Possible Extensions
Use intraday data (4H, 15M, 5M) instead of daily and compare results.
Try other models (e.g., Gradient Boosting, XGBoost, or LSTM with proper time‑series validation).
Incorporate transaction costs and basic backtesting instead of only next‑candle direction.
Add macro or sentiment features (news, risk‑on/risk‑off proxies).
