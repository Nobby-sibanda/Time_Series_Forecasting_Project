# Time_Series_Forecasting_Project

This project focuses on financial time series analysis involving stock price forecasting, volatility estimation, and pairs trading strategy development. The project focuses in sector/economic cycle stocks and dually listed stocks.
It includes loading and exploring financial data, applying stationarity tests, implementing classical econometric models (ARIMA, GARCH, Cointegration), developing a pairs trading strategy, and building machine learning models (LSTM, Prophet) for forecasting. Finally, compare the models and visualize the results in an interactive dashboard to summarize findings and insights.

# STEPS TAKEN IN CARRYING OUT THE PROJECT
# 1. Data loading and exploration

Load the necessary financial time series data for economic cycle stocks and dually listed stocks from yfinance, then perform initial exploratory data analysis (EDA) to understand data characteristics, distributions, and potential outliers.

# 2. Stationarity Testing and Preprocessing#

Import the necessary libraries for stationarity tests (ADF and KPSS) and define helper functions to interpret their results, which will make the output clearer and more consistent. Conduct stationarity tests on the stock price series. Apply appropriate transformations to achieve stationarity.

# 3. ARIMA Model for Price Forecasting 

Implement and train ARIMA models for price forecasting on selected stock series. Determine optimal ARIMA(p,d,q) orders using AIC/BIC criteria or ACF/PACF plots.
The first step is to calculate the daily returns for each stock from their closing prices. This transformation often helps in achieving stationarity, which is a prerequisite for identifying appropriate p and q orders from ACF/PACF plots.

# 4. GARCH(1,1) Model for Volatility Estimation

Implement and train GARCH(1,1) models to estimate and forecast volatility for the selected stock series, considering the squared residuals from the ARIMA models or returns. For each stock, I will instantiate and fit a GARCH(1,1) model and print its summary, which will address the first steps of the subtask.

# 5. Cointegration Analysis for Dual Listings

Apply a cointegration framework (Engle-Granger two-step or Johansen test) to identify cointegrated pairs among the dually listed stocks. Model the spread between these pairs. The task is to combine the 'Close' price series of the dually listed stocks (BHP and RY) into a single DataFrame and ensure they are aligned by date, which is necessary before performing cointegration tests.

# 6. Pairs Trading Strategy Development

Develop a simple mean-reversion based pairs trading strategy using the cointegrated spreads identified in the previous step, generating entry and exit signals.

# 7. Walk-Forward Validation Setup

Implement a robust walk-forward validation framework for all models (ARIMA, GARCH, Cointegration-based strategy, LSTM, Prophet). Define the parameters for the walk-forward validation, including selecting a stock, setting initial training and testing window sizes, and defining the step size (re-calibration frequency). Next, iterate through the time series, splitting the data into training and testing sets for each step and printing the window dates for demonstration purposes.

# 8. Model Evaluation with Financial KPIs

Evaluate the performance of all forecasting models and the pairs trading strategy using finance-specific Key Performance Indicators (KPIs) such as directional accuracy, Sharpe ratio, Sortino ratio, forecast interval coverage, and maximum drawdown.
Initialize empty dictionaries to store the ARIMA model's forecasts, actual values, and directional accuracy results for each walk-forward iteration. This sets up the data structures needed for evaluation.

# 9. Machine Learning Forecasting (LSTM/Prophet)- part 1

Prepare stock data for LSTM model training by creating supervised learning sequences. Define a function to convert the time series data into a supervised learning format suitable for LSTM models, where input sequences (X) are created from a 'look_back' number of previous days' prices, and the output (y) is the next day's price. This function is crucial for preparing the data for the next steps in LSTM model training.

# 10. Machine Learning Forecasting (LSTM/Prophet)- part 2

Implement and train an LSTM model for time-series price forecasting on a selected stock series, leveraging the walk-forward validation setup. Import the necessary Keras layers to build the LSTM model and then define the model architecture, including an LSTM layer and a Dense output layer. This sets up the model structure as required by the subtask.

# 11. Calculate Directional Accuracy for LSTM

Calculate the directional accuracy for the LSTM model's forecasts for each walk-forward iteration which involves comparing the predicted price movement with the actual price movement.

{Reasoning: The previous code failed due to an unterminated f-string literal and an incomplete iteration. I will modify the LSTM walk-forward validation loop to correctly calculate and store directional accuracy, as well as fix the f-string syntax in the print statements and ensure that the evaluation dictionaries are reset for each run. I will also incorporate the new lstm_directional_accuracy dictionary and related calculations.} 

# 12. Calculate Directional Accuracy for LSTM

{Re-run the LSTM walk-forward validation loop to calculate and store the directional accuracy for each iteration. This involves retraining the LSTM model on each training set, generating forecasts, and comparing the predicted price movements with actual price movements. The average directional accuracy will then be computed and displayed.
Reasoning: I will re-run the code in cell ad1f6384 to execute the LSTM walk-forward validation loop, which includes retraining the model, generating forecasts, and calculating directional accuracy for each iteration as requested by the subtask.}

# 13. Calculate Sharpe Ratio for LSTM

Calculate the Sharpe Ratio for the hypothetical trading strategy based on LSTM forecasts for each walk-forward iteration, using the previously defined helper function.

{Reasoning: The subtask requires calculating the Sharpe Ratio for the LSTM model's hypothetical trading strategy. I will modify the previous code block to integrate the calculate_sharpe_ratio function with the strategy_returns already derived from the LSTM forecasts within the walk-forward validation loop. I'll also initialize lstm_sharpe_ratios and include the output in the print statements for demonstration.}

# 14. 










