# Financial Time Series Forecasting Using Deep Learning and News Analysis

A financial forecasting project that combines historical market prices, financial news, sentiment analysis, and text embeddings to explore the prediction of cryptocurrency and commodity prices using deep learning.

The project investigates how numerical market indicators and textual information can be incorporated into time-series forecasting models, with a focus on **GRU-based regression, LSTM networks, and multi-horizon price forecasting**.

## Overview

Financial markets are influenced by historical price movements as well as economic events, geopolitical developments, and market sentiment. This project explores different approaches to financial forecasting by combining numerical market data with information extracted from daily financial news.

The implementation includes data preparation, chronological dataset splitting, feature engineering, neural network training, model evaluation, and visualization of predicted versus actual prices.

## Key Features

* **Financial News Processing:** Extracts and combines relevant daily headlines using finance-related keywords.
* **Market Data Collection:** Retrieves historical opening and closing prices using Yahoo Finance.
* **Multi-Asset Analysis:** Works with Bitcoin, gold futures, and crude oil futures.
* **Feature Engineering:** Generates intraday returns and daily returns from historical prices.
* **Sentiment Analysis:** Incorporates sentiment information extracted from financial news.
* **Text Embeddings:** Explores the use of sentence embeddings as additional model inputs.
* **Deep Learning Models:** Implements GRU-based regression and LSTM-based forecasting.
* **Time-Series Validation:** Supports chronological train, validation, and test splits.
* **Model Evaluation:** Uses Root Mean Squared Error (RMSE) and Directional Accuracy.
* **Visualization:** Plots actual and predicted prices and compares forecasting results across different model configurations and prediction horizons.

## Assets

The project uses the following financial instruments:

| Ticker    | Asset               |
| --------- | ------------------- |
| `BTC-USD` | Bitcoin / US Dollar |
| `GC=F`    | Gold Futures        |
| `CL=F`    | Crude Oil Futures   |

The selected assets provide an opportunity to investigate forecasting across cryptocurrency and commodity markets.

## Methodology

### 1. Financial News Processing

The project processes a financial news dataset containing daily headlines.

The preprocessing pipeline includes:

* Identifying the date column.
* Detecting available news headline columns.
* Cleaning whitespace and normalizing text formatting.
* Identifying finance-related headlines using a predefined keyword list.
* Combining selected headlines into a daily news text.

When no finance-related keywords are detected, the pipeline uses a fallback strategy to retain selected headlines.

### 2. Market Data Collection and Integration

Historical opening and closing prices are retrieved through the Yahoo Finance interface provided by `yfinance`.

The news data and market prices are aligned using their date columns to construct a unified dataset.

The resulting dataset contains daily news information and numerical market features for the selected assets.

### 3. Feature Engineering

Numerical features are constructed from historical opening and closing prices.

The main derived features include:

* **Intraday Return:** Measures the relative change between the opening and closing prices of a trading session.
* **Daily Return:** Measures the relative change between consecutive closing prices.

These features are combined with historical price values to represent market behavior over time.

### 4. Sentiment Analysis and Text Embeddings

The project explores two approaches to incorporating information from financial news.

**Sentiment features**

Sentiment information is extracted using VADER sentiment analysis and incorporated into a forecasting configuration alongside numerical market data.

**Text embeddings**

Sentence embeddings are used to represent the semantic information contained in news text as numerical vectors.

These representations provide an alternative to relying exclusively on numerical market indicators.

### 5. Chronological Data Splitting

The notebook implements multiple strategies for separating the data into training, validation, and testing subsets.

These include:

* **Calendar-based split:** Uses a calendar boundary to reserve the final nine months for testing.
* **Percentage-based split:** Reserves the final 20% of observations for testing.
* **Time-based split:** Provides an additional chronological splitting strategy.

The validation subset is separated from the earlier observations to support model selection and evaluation.

Chronological splitting is appropriate for time-series experiments because it preserves the temporal order of observations.

### 6. GRU-Based Regression

The GRU component predicts the next day's closing price using historical sequences.

The model uses a configurable lookback window to construct sequential input samples.

Its architecture includes:

* A Gated Recurrent Unit (GRU) network.
* Layer normalization.
* Dropout regularization.
* Fully connected layers.
* A regression output for the predicted closing price.

The notebook evaluates three input configurations:

| Configuration        | Input Features                                    |
| -------------------- | ------------------------------------------------- |
| Numeric              | Numerical market features                         |
| Numeric + Embeddings | Numerical market features and sentence embeddings |
| Numeric + Sentiment  | Numerical market features and sentiment features  |

This setup allows the contribution of different information sources to be investigated through comparative experiments.

### 7. LSTM-Based Multi-Horizon Forecasting

The LSTM component provides another approach to sequential price prediction.

A separate LSTM model is trained for each selected asset using historical numerical features.

The model learns from a configurable lookback window and predicts the next closing price.

The notebook then uses **recursive forecasting** to generate predictions for multiple future horizons:

* 3 days
* 5 days
* 10 days

In recursive forecasting, each predicted step is used to construct the input for the next prediction. Because future observations are unavailable, the implementation carries forward selected features for other assets and updates the derived return features.

The resulting forecasts are visualized individually and in cross-asset comparisons.

## Evaluation Metrics

The project uses the following evaluation metrics.

### Root Mean Squared Error (RMSE)

RMSE measures the magnitude of the difference between actual and predicted prices.

A lower RMSE indicates smaller prediction errors in the evaluated price scale.

### Directional Accuracy (DA)

Directional Accuracy measures how often the predicted direction of price movement matches the actual direction.

For the GRU evaluation, the direction is determined by comparing the predicted and actual next-day closing prices with the current closing price.

A higher Directional Accuracy indicates better agreement between predicted and actual movement directions under this definition.

## Visualization

The notebook generates visual comparisons to support model evaluation, including:

* Actual versus predicted closing prices.
* RMSE comparisons across GRU configurations.
* Directional Accuracy comparisons.
* Individual LSTM forecasts for different assets.
* Three-day, five-day, and ten-day recursive forecasts.
* Cross-asset comparisons of actual and predicted price trajectories.

These visualizations help illustrate differences between model configurations and forecasting horizons.

## Technologies Used

* **Python**
* **Pandas** — Data manipulation and preparation
* **NumPy** — Numerical operations
* **PyTorch** — Neural network implementation and training
* **Scikit-learn** — Feature scaling and preprocessing
* **Matplotlib** — Data visualization
* **yfinance** — Historical financial market data
* **NLTK** — Natural language processing utilities
* **VADER Sentiment** — Sentiment analysis
* **Sentence Transformers** — Text embeddings
* **Hugging Face Transformers** — Transformer-based NLP components
* **Jupyter Notebook** — Interactive development and experimentation

## Project Structure

The repository can be organized as follows:

```text
financial-time-series-forecasting/
│
├── README.md
├── Final Project.ipynb
├── requirements.txt
└── data/
    └── WorldNewsData.csv
```

The structure above is a suggested repository layout. The `data/` directory should contain the news dataset only if you are permitted to redistribute it.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/GhazalCh/financial-time-series-forecasting.git
cd financial-time-series-forecasting
```


### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

Otherwise, install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn torch yfinance nltk sentence-transformers transformers vaderSentiment tqdm python-dateutil jupyter
```

The installed versions should be checked for compatibility with the environment in which the notebook is executed.

### 4. Prepare the Dataset

The notebook expects a financial news CSV file.

Update the `NEWS_CSV` variable to point to the location of your dataset:

```python
NEWS_CSV = "data/WorldNewsData.csv"
```

The dataset should contain a date column and news headline columns, such as `Top1`, `Top2`, and subsequent headline fields, or other supported headline column names.

### 5. Run the Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open `Final Project.ipynb` and execute the cells in order.

**Note:** The project requires access to the internet to retrieve market prices through Yahoo Finance. Depending on the notebook's execution path, pretrained NLP models may also need to be downloaded or available locally.

## Limitations

* Financial markets are inherently volatile, and historical patterns do not guarantee future performance.
* News headline selection depends partly on a predefined finance-related keyword list.
* The quality and availability of market data and news data can affect the results.
* Recursive forecasts can accumulate prediction errors over longer horizons.
* Predictions for other assets and future market features may rely on simplified assumptions.
* Forecasting performance must be established through actual experimental results rather than assumed from the model architecture alone.

## Future Improvements

Potential extensions include:

* Comparing GRU and LSTM models under identical evaluation conditions.
* Adding more financial and macroeconomic indicators.
* Experimenting with alternative sentiment analysis methods.
* Investigating different sentence embedding models.
* Introducing stronger baselines for comparison.
* Evaluating performance across multiple market regimes.
* Improving recursive forecasting by modeling future input features more explicitly.
* Adding reproducible experiment configurations and automated evaluation reports.

## Disclaimer

This project is intended for educational and research purposes only. It does not provide financial advice, investment recommendations, or guarantees of future market performance.

## Author

**Ghazal Cheraghi**

[GitHub Profile](https://github.com/GhazalCh)
