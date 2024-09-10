# Quantitative Stock Price Prediction Using Machine Learning

## Project Overview
This project aims to develop a **machine learning model** to predict the **next day’s stock prices** for a given company using various **technical indicators** such as **Simple Moving Averages (SMA)**, **Relative Strength Index (RSI)**, and **Bollinger Bands**. It involves fetching historical stock data, applying data preprocessing techniques, generating technical indicators, and building a **Linear Regression** model to forecast future stock prices.

The project leverages **Python**, **Google Colab** as the coding environment, and key Python libraries such as **pandas**, **scikit-learn**, **Plotly**, and **pandas-ta** (a library for technical analysis). The predictions are evaluated by comparing them to the actual stock prices using visualizations.

This project can be extended to include more sophisticated machine learning models like **LSTMs (Long Short-Term Memory networks)** for time-series forecasting, reinforcement learning models for adaptive strategies, and **backtesting** strategies to validate the effectiveness of trading models.

## Objectives
- Develop a model to predict future stock prices using **technical indicators**.
- Visualize stock price predictions alongside actual prices for comparative analysis.
- Provide insights into stock trends using **technical indicators** like SMA, RSI, and Bollinger Bands.
- Understand the relationship between various stock market indicators and price movements.
- Offer practical trading insights based on machine learning predictions.

## Technologies Used
- **Google Colab**: Cloud-based Jupyter Notebook platform used to write and execute Python code.
- **Python**: Main programming language for data analysis, visualization, and machine learning.
- **pandas**: Used for data manipulation and analysis.
- **pandas-ta**: A library for adding technical indicators such as SMA, RSI, and Bollinger Bands.
- **scikit-learn**: For building the machine learning model (Linear Regression in this case).
- **matplotlib & Plotly**: For creating visualizations, with Plotly being used for interactive charts.
- **yfinance**: For fetching historical stock data from Yahoo Finance.

## Dataset Used
- **Source**: Yahoo Finance API, accessed through the **yfinance** Python library.
- **Stock**: Historical data for **NVIDIA (NVDA)** stock was used.
- **Data Period**: Historical stock data from 2020 to 2024 was fetched, which included:
  - Open, High, Low, Close, Adjusted Close prices
  - Volume traded
- **Indicators Used**:
  - **Simple Moving Average (SMA)**: 50-day and 200-day moving averages were used.
  - **Relative Strength Index (RSI)**: To measure momentum and identify overbought/oversold conditions.
  - **Bollinger Bands**: To capture volatility in the stock price.

## Process Description

### 1. Data Collection
- **yfinance** was used to collect historical data for **NVIDIA (NVDA)** stock from 2020 to 2024.
- This data includes the open, high, low, close prices, and volume.

### 2. Data Preprocessing
- The data is first cleaned by removing missing values (NaN values) from the stock price columns and technical indicators.
- Key technical indicators such as **SMA50**, **SMA200**, **RSI**, and **Bollinger Bands** were generated using the **pandas-ta** library. These indicators are critical to understand market trends and are used as features in the model.

### 3. Feature Engineering
- **Simple Moving Averages (SMA)**: We calculated 50-day and 200-day moving averages to smoothen the price data and highlight trends over medium and long periods.
- **Relative Strength Index (RSI)**: The RSI indicator helped in identifying overbought (RSI > 70) and oversold (RSI < 30) conditions in the stock price, which can be crucial in making trading decisions.
- **Bollinger Bands**: This indicator uses standard deviations to provide a sense of price volatility.

### 4. Building the Predictive Model
- The **Linear Regression** model was used as a first step in building a predictive system. Linear regression fits a straight line to predict future prices based on past values.
- The features used for prediction included **SMA50**, **SMA200**, and **RSI**, while the target variable was the next day's **closing price**.
- **Train-Test Split**: The dataset was split into **80% training data** and **20% test data** to evaluate model performance.

### 5. Model Evaluation
- The model predictions were compared against actual stock prices to see how well the model performed.
- Both the predicted and actual prices were visualized using **matplotlib** and **Plotly** to offer a clear comparison of model performance.
  
### 6. Visualization
- An interactive candlestick chart was created using **Plotly**, displaying the stock price over time along with the **SMA**, **RSI**, and **Bollinger Bands**.
- A separate chart showed **Predicted Prices vs Actual Prices**, giving an overview of the model's performance.

## Challenges and Solutions

- **Messy Visualizations**: Initially, the plot was cluttered due to plotting all data points at once. This was resolved by smoothing the data using a **rolling mean** and plotting over smaller time ranges.
- **KeyError**: While aligning the data, a mismatch in indices caused errors. This was fixed by ensuring proper alignment between features and labels using the `align` method and removing any NaN values.
- **Accuracy**: Linear regression is a basic model. The next step would be to implement more advanced models like **LSTM** for time-series prediction, which is better suited for stock data.

## Future Enhancements
- **Incorporating Advanced Models**: Implementing more advanced models like **LSTM (Long Short-Term Memory)** and **ARIMA** for time-series analysis to improve the prediction accuracy.
- **Automated Trading Strategy**: Integrating this model into an automated trading bot, using APIs from stock and crypto exchanges to execute trades based on predictions.
- **Backtesting**: Running **backtests** on historical data to validate the effectiveness of the trading model over time.
- **Predicting Cryptocurrencies**: Expanding the model to include cryptocurrencies like **Bitcoin (BTC)** or **Ethereum (ETH)**, given their volatile nature and the importance of predictive models in this space.
- **Feature Expansion**: Incorporating more features such as **MACD (Moving Average Convergence Divergence)**, **ADX (Average Directional Index)**, and **Volume Indicators** to improve the predictive power of the model.
  
## Conclusion
This project demonstrates a practical approach to using machine learning for stock price prediction, utilizing key technical indicators to make predictions. By providing both data insights and visual comparisons, it showcases how machine learning can assist in making informed financial decisions. While the Linear Regression model provides a simple yet effective start, the project also lays the groundwork for more advanced predictive systems and strategies in both traditional stock markets and the emerging field of cryptocurrency trading.

This foundation can now be extended to include more sophisticated models, backtesting, and integration with real-world trading platforms.

## Author
**Sheikh Faizan**

- **LinkedIn**: [https://www.linkedin.com/in/ifzsheikh/](https://www.linkedin.com/in/ifzsheikh/)
- **Website**: [www.sheikhfaizan.com](https://www.sheikhfaizan.com)
