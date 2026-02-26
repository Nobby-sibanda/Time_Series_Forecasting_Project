# Time_Series_Forecasting_Project

This project focuses on financial time series analysis involving stock price forecasting, volatility estimation, and pairs trading strategy development. The project focuses in sector/economic cycle stocks and dually listed stocks.
It includes loading and exploring financial data, applying stationarity tests, implementing classical econometric models (ARIMA, GARCH, Cointegration), developing a pairs trading strategy, and building machine learning models (LSTM, Prophet) for forecasting. Finally, compare the models and visualize the results in an interactive dashboard to summarize findings and insights.

#Steps in carrying out the project 
#1. Data loading and exploration

Load the necessary financial time series data for economic cycle stocks and dually listed stocks from yfinance, then perform initial exploratory data analysis (EDA) to understand data characteristics, distributions, and potential outliers.

#2. Stationarity Testing and Preprocessing

Import the necessary libraries for stationarity tests (ADF and KPSS) and define helper functions to interpret their results, which will make the output clearer and more consistent. Conduct stationarity tests on the stock price series. Apply appropriate transformations to achieve stationarity.

#3. ARIMA Model for Price Forecasting

Implement and train ARIMA models for price forecasting on selected stock series. Determine optimal ARIMA(p,d,q) orders using AIC/BIC criteria or ACF/PACF plots.
The first step is to calculate the daily returns for each stock from their closing prices. This transformation often helps in achieving stationarity, which is a prerequisite for identifying appropriate p and q orders from ACF/PACF plots.











