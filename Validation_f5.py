
#Walk-Forward Validation Setup (1)
#Implement a robust walk-forward validation framework for all models (ARIMA, GARCH, Cointegration-based strategy, LSTM, Prophet). Define training and testing windows and re-calibration frequency.
#I will define the parameters for the walk-forward validation, including selecting a stock, setting initial training and testing window sizes, and defining the step size (re-calibration frequency). 
# Then, I will iterate through the time series, splitting the data into training and testing sets for each step and printing the window dates for demonstration purposes.

import pandas as pd

print("Setting up Walk-Forward Validation Framework...")

# 1. Choose one stock (e.g., 'AAPL') for demonstration
selected_ticker = 'AAPL'
stock_data = all_stocks_data[selected_ticker][f'{selected_ticker}_Close']

# Calculate total number of observations
n_observations = len(stock_data)
print(f"Total observations for {selected_ticker}: {n_observations}")

# 2. Define the initial training window size (e.g., 70% of the data)
initial_train_size = int(n_observations * 0.70)
print(f"Initial training size: {initial_train_size} observations")

# 3. Define the testing window size (e.g., 30 periods)
test_window_size = 30 # For example, forecast 30 days ahead
print(f"Testing window size: {test_window_size} observations")

# 4. Define the re-calibration frequency or step size (e.g., re-train every 10 periods)
step_size = 10 # Shift windows by 10 periods for each iteration
print(f"Walk-forward step size (re-calibration frequency): {step_size} observations")

# Initialize a list to store predictions (not required for this subtask but good practice)
# all_predictions = []
# all_actuals = []

print("\nDemonstrating Walk-Forward Window Splits (first 5 iterations):\n")

# Implement a loop that iterates through the time series
# The loop will continue as long as there's enough data for at least one test period
for i in range(0, n_observations - initial_train_size - test_window_size + 1, step_size):
    # Define the end of the current training window
    train_end = initial_train_size + i

    # Define the start and end of the current testing window
    test_start = train_end
    test_end = train_end + test_window_size

    # Ensure test_end does not exceed total observations
    if test_end > n_observations:
        test_end = n_observations # Adjust last test window if it goes beyond data
        if test_start >= test_end: # If no data left for testing
            break

    # Split data into training and testing sets
    train_data = stock_data.iloc[:train_end]
    test_data = stock_data.iloc[test_start:test_end]

    # 8. Print the start and end dates of the current training and testing windows
    if i < 5 * step_size: # Print for first 5 iterations to demonstrate
        print(f"Iteration {i // step_size + 1}:")
        print(f"  Training Window: {train_data.index.min().strftime('%Y-%m-%d')} to {train_data.index.max().strftime('%Y-%m-%d')} (Size: {len(train_data)})")
        print(f"  Testing Window:  {test_data.index.min().strftime('%Y-%m-%d')} to {test_data.index.max().strftime('%Y-%m-%d')} (Size: {len(test_data)})")

    # In a full implementation, models would be trained and evaluated here:
    # model = ARIMA(train_data, order=(p,d,q)).fit()
    # forecast = model.predict(start=test_data.index[0], end=test_data.index[-1])
    # Evaluate forecast against test_data

print("\nWalk-forward validation framework setup demonstrated.")
print("Note: Actual model training and evaluation are placeholders for this subtask.")



#Model Evaluation with Financial KPIs (2)
#Evaluate the performance of all forecasting models and the pairs trading strategy using finance-specific Key Performance Indicators 
# (KPIs) such as directional accuracy, Sharpe ratio, Sortino ratio, forecast interval coverage, and maximum drawdown.
#The first step is to initialize empty dictionaries to store the ARIMA model's forecasts, actual values, and directional accuracy results for each walk-forward iteration.
# This sets up the data structures needed for evaluation.


import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
import numpy as np

# Suppress specific warnings from statsmodels ARIMA fitting for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Dictionaries to store ARIMA evaluation results
arima_forecasts = {}
arima_actuals = {}
arima_directional_accuracy = {}

print("Initialized dictionaries for ARIMA evaluation results.")

#Now that the evaluation result dictionaries are initialized, I will iterate through the walk-forward validation windows,
#train the ARIMA model on each training set, generate forecasts, calculate directional accuracy, and store the results as specified in the instructions.


print("Performing Walk-Forward Validation and ARIMA Forecasting...")

# --- Walk-Forward Validation Loop ---
# Reusing parameters from the previous walk-forward setup
selected_ticker = 'AAPL'
stock_data = all_stocks_data[selected_ticker][f'{selected_ticker}_Close']
n_observations = len(stock_data)
initial_train_size = int(n_observations * 0.70)
test_window_size = 30
step_size = 10

iteration_num = 0
for i in range(0, n_observations - initial_train_size - test_window_size + 1, step_size):
    iteration_num += 1
    train_end = initial_train_size + i
    test_start = train_end
    test_end = train_end + test_window_size

    if test_end > n_observations:
        test_end = n_observations
        if test_start >= test_end:
            break

    train_data = stock_data.iloc[:train_end]
    test_data = stock_data.iloc[test_start:test_end]

    # 3. For each iteration, within the loop:
    # a. Retrain the ARIMA model on the current `train_data`
    #    (Using pre-determined arima_order = (1,1,1) for simplicity as per previous step)
    try:
        model = ARIMA(train_data, order=arima_order)
        model_fit = model.fit()

        # b. Generate a forecast for the `test_window_size` period
        # Forecast step starts from the end of train_data.index + 1
        # and ends at the end of test_data.index
        forecast_index = pd.date_range(start=train_data.index[-1] + pd.Timedelta(days=1), periods=len(test_data), freq='B')
        forecast = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)

        # Align forecast index with actual test_data index for proper comparison
        forecast.index = test_data.index

        # Store forecast and actuals
        arima_forecasts[iteration_num] = forecast
        arima_actuals[iteration_num] = test_data

        # c. Calculate the actual price movements from `test_data`.
        actual_movements = np.sign(test_data.diff().dropna())

        # d. Calculate key performance indicators (KPIs) for the ARIMA model's forecast
        # Directional Accuracy
        forecast_movements = np.sign(forecast.diff().dropna())
        
        # Ensure both series are aligned for comparison
        common_index = actual_movements.index.intersection(forecast_movements.index)
        
        if len(common_index) > 0:
            correct_direction = (forecast_movements.loc[common_index] == actual_movements.loc[common_index]).sum()
            total_predictions = len(common_index)
            directional_accuracy = correct_direction / total_predictions if total_predictions > 0 else 0
            arima_directional_accuracy[iteration_num] = directional_accuracy
        else:
            arima_directional_accuracy[iteration_num] = np.nan # No overlapping movements to compare

        if iteration_num <= 5: # Print for first 5 iterations to demonstrate
            print(f"Iteration {iteration_num}:")
            print(f"  Training Window: {train_data.index.min().strftime('%Y-%m-%d')} to {train_data.index.max().strftime('%Y-%m-%d')}")
            print(f"  Testing Window:  {test_data.index.min().strftime('%Y-%m-%d')} to {test_data.index.max().strftime('%Y-%m-%d')}")
            print(f"  Forecasted values (first 5):\n{forecast.head().to_string()}")
            print(f"  Actual values (first 5):\n{test_data.head().to_string()}")
            print(f"  Directional Accuracy: {directional_accuracy:.2f}")
            print("--------------------------------------------------")

    except Exception as e:
        print(f"Error in iteration {iteration_num} for {selected_ticker}: {e}")
        arima_directional_accuracy[iteration_num] = np.nan


print("Walk-forward validation and ARIMA forecasting complete for directional accuracy.")

# 4. After the walk-forward loop, aggregate and analyze the collected KPIs for the ARIMA model
print(f"\nAverage Directional Accuracy for ARIMA on {selected_ticker}: {np.nanmean(list(arima_directional_accuracy.values())):.2f}")


#To further evaluate the ARIMA model's performance as requested by the subtask, I will now implement the calculation of Sharpe Ratio and Sortino Ratio. 
# This involves defining helper functions for these metrics, calculating daily returns from the ARIMA forecasts and actual prices, and integrating these calculations into the walk-forward validation loop to store and aggregate the results.


import numpy as np
import pandas as pd

# Assuming a risk-free rate of 0% for simplicity in daily returns calculation
# For a more robust financial analysis, a proper risk-free rate should be used.
RISK_FREE_RATE = 0.0

def calculate_sharpe_ratio(returns, risk_free_rate=RISK_FREE_RATE, annualization_factor=252):
    """Calculates the annualized Sharpe Ratio."""
    if returns.empty or returns.std() == 0:
        return np.nan
    excess_returns = returns - risk_free_rate / annualization_factor
    return np.sqrt(annualization_factor) * excess_returns.mean() / excess_returns.std()

def calculate_sortino_ratio(returns, risk_free_rate=RISK_FREE_RATE, annualization_factor=252):
    """Calculates the annualized Sortino Ratio."""
    if returns.empty:
        return np.nan
    excess_returns = returns - risk_free_rate / annualization_factor
    negative_returns = excess_returns[excess_returns < 0]
    if negative_returns.empty or negative_returns.std() == 0:
        return np.nan
    downside_deviation = negative_returns.std()
    return np.sqrt(annualization_factor) * excess_returns.mean() / downside_deviation

# Dictionaries to store additional KPIs for ARIMA evaluation
arima_sharpe_ratios = {}
arima_sortino_ratios = {}

print("Continuing Walk-Forward Validation for ARIMA Forecasting to calculate Sharpe and Sortino Ratios...")

# Reusing parameters from the previous walk-forward setup
selected_ticker = 'AAPL'
stock_data = all_stocks_data[selected_ticker][f'{selected_ticker}_Close']
n_observations = len(stock_data)
initial_train_size = int(n_observations * 0.70)
test_window_size = 30
step_size = 10

iteration_num = 0
for i in range(0, n_observations - initial_train_size - test_window_size + 1, step_size):
    iteration_num += 1
    train_end = initial_train_size + i
    test_start = train_end
    test_end = train_end + test_window_size

    if test_end > n_observations:
        test_end = n_observations
        if test_start >= test_end:
            break

    train_data = stock_data.iloc[:train_end]
    test_data = stock_data.iloc[test_start:test_end]

    try:
        # Retrain the ARIMA model on the current `train_data`
        model = ARIMA(train_data, order=arima_order)
        model_fit = model.fit()

        # Generate a forecast for the `test_window_size` period
        forecast = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)
        forecast.index = test_data.index

        # Store forecast and actuals (already done in previous step, but re-confirming)
        arima_forecasts[iteration_num] = forecast
        arima_actuals[iteration_num] = test_data

        # Calculate actual and forecasted price movements for Directional Accuracy (already done)
        actual_movements = np.sign(test_data.diff().dropna())
        forecast_movements = np.sign(forecast.diff().dropna())
        common_index = actual_movements.index.intersection(forecast_movements.index)
        if len(common_index) > 0:
            correct_direction = (forecast_movements.loc[common_index] == actual_movements.loc[common_index]).sum()
            total_predictions = len(common_index)
            directional_accuracy = correct_direction / total_predictions if total_predictions > 0 else 0
            arima_directional_accuracy[iteration_num] = directional_accuracy
        else:
            arima_directional_accuracy[iteration_num] = np.nan

        # --- New KPI Calculations --- 
        # Calculate daily returns for both actual and forecasted prices to evaluate Sharpe/Sortino
        actual_returns = test_data.pct_change().dropna()
        forecast_returns = forecast.pct_change().dropna()

        # To calculate Sharpe/Sortino, we need a series of 'returns generated by the forecast'
        # For simplicity, we assume if the forecast predicts an increase, we 'buy' and if it predicts a decrease, we 'sell' (or go short).
        # The 'return' of our strategy is then the actual return if our prediction was correct, or negative actual return if it was wrong (simplified).
        # A more rigorous approach would involve backtesting a trading strategy based on forecasts.
        # For now, let's consider the returns of merely 'following' the forecast direction (long if forecast up, short if forecast down).
        
        # Create a series of returns based on forecast direction applied to actual returns
        strategy_returns = pd.Series(index=actual_returns.index, dtype=float)
        for idx in actual_returns.index:
            if idx in forecast_movements.index:
                if forecast_movements.loc[idx] == 1: # Forecasts increase, so we assume long position
                    strategy_returns.loc[idx] = actual_returns.loc[idx]
                elif forecast_movements.loc[idx] == -1: # Forecasts decrease, so we assume short position
                    strategy_returns.loc[idx] = -actual_returns.loc[idx]
                else: # Forecasts no change
                    strategy_returns.loc[idx] = 0 # No return
            else:
                strategy_returns.loc[idx] = 0 # No forecast for this day

        strategy_returns = strategy_returns.dropna()
        
        arima_sharpe_ratios[iteration_num] = calculate_sharpe_ratio(strategy_returns)
        arima_sortino_ratios[iteration_num] = calculate_sortino_ratio(strategy_returns)


        if iteration_num <= 5: # Print for first 5 iterations to demonstrate
            print(f"Iteration {iteration_num}:")
            print(f"  Training Window: {train_data.index.min().strftime('%Y-%m-%d')} to {train_data.index.max().strftime('%Y-%m-%d')}")
            print(f"  Testing Window:  {test_data.index.min().strftime('%Y-%m-%d')} to {test_data.index.max().strftime('%Y-%m-%d')}")
            print(f"  Forecasted values (first 5):\n{forecast.head().to_string()}")
            print(f"  Actual values (first 5):\n{test_data.head().to_string()}")
            print(f"  Directional Accuracy: {directional_accuracy:.2f}")
            print(f"  Sharpe Ratio: {arima_sharpe_ratios[iteration_num]:.2f}")
            print(f"  Sortino Ratio: {arima_sortino_ratios[iteration_num]:.2f}")
            print("--------------------------------------------------")

    except Exception as e:
        print(f"Error in iteration {iteration_num} for {selected_ticker}: {e}")
        arima_directional_accuracy[iteration_num] = np.nan
        arima_sharpe_ratios[iteration_num] = np.nan
        arima_sortino_ratios[iteration_num] = np.nan

print("Walk-forward validation and ARIMA forecasting complete for all specified KPIs.")

# After the walk-forward loop, aggregate and analyze the collected KPIs for the ARIMA model
print(f"\nAverage Directional Accuracy for ARIMA on {selected_ticker}: {np.nanmean(list(arima_directional_accuracy.values())):.2f}")
print(f"Average Sharpe Ratio for ARIMA on {selected_ticker}: {np.nanmean(list(arima_sharpe_ratios.values())):.2f}")
print(f"Average Sortino Ratio for ARIMA on {selected_ticker}: {np.nanmean(list(arima_sortino_ratios.values())):.2f}")


#To further evaluate the ARIMA model as required by the subtask, I will now calculate the forecast interval coverage. This involves generating prediction intervals from the 
#ARIMA model's forecast and then checking how many of the actual values fall within these intervals for each walk-forward iteration.


import numpy as np
import pandas as pd

# Dictionary to store Forecast Interval Coverage for ARIMA
arima_interval_coverage = {}

print("Continuing Walk-Forward Validation for ARIMA Forecasting to calculate Forecast Interval Coverage...")

# Reusing parameters from the previous walk-forward setup
selected_ticker = 'AAPL'
stock_data = all_stocks_data[selected_ticker][f'{selected_ticker}_Close']
n_observations = len(stock_data)
initial_train_size = int(n_observations * 0.70)
test_window_size = 30
step_size = 10

iteration_num = 0
for i in range(0, n_observations - initial_train_size - test_window_size + 1, step_size):
    iteration_num += 1
    train_end = initial_train_size + i
    test_start = train_end
    test_end = train_end + test_window_size

    if test_end > n_observations:
        test_end = n_observations
        if test_start >= test_end:
            break

    train_data = stock_data.iloc[:train_end]
    test_data = stock_data.iloc[test_start:test_end]

    try:
        # Retrain the ARIMA model on the current `train_data`
        model = ARIMA(train_data, order=arima_order)
        model_fit = model.fit()

        # Generate a forecast with prediction intervals (e.g., 95% confidence interval)
        forecast_results = model_fit.get_forecast(steps=len(test_data))
        forecast_mean = forecast_results.predicted_mean
        conf_int = forecast_results.conf_int(alpha=0.05) # 95% confidence interval

        # Align forecast index with actual test_data index for proper comparison
        forecast_mean.index = test_data.index
        conf_int.index = test_data.index

        # Calculate Forecast Interval Coverage
        # Check how many actual values fall within the confidence interval
        lower_bound = conf_int.iloc[:, 0]
        upper_bound = conf_int.iloc[:, 1]

        within_interval = ((test_data >= lower_bound) & (test_data <= upper_bound)).sum()
        total_points = len(test_data)
        coverage = within_interval / total_points if total_points > 0 else np.nan
        arima_interval_coverage[iteration_num] = coverage

        if iteration_num <= 5: # Print for first 5 iterations to demonstrate
            print(f"Iteration {iteration_num}:")
            print(f"  Training Window: {train_data.index.min().strftime('%Y-%m-%d')} to {train_data.index.max().strftime('%Y-%m-%d')}")
            print(f"  Testing Window:  {test_data.index.min().strftime('%Y-%m-%d')} to {test_data.index.max().strftime('%Y-%m-%d')}")
            print(f"  Forecast Mean (first 5):\n{forecast_mean.head().to_string()}")
            print(f"  Actual values (first 5):\n{test_data.head().to_string()}")
            print(f"  95% Confidence Interval (first 5):\n{conf_int.head().to_string()}")
            print(f"  Forecast Interval Coverage: {coverage:.2f}")
            print("--------------------------------------------------")

    except Exception as e:
        print(f"Error in iteration {iteration_num} for {selected_ticker}: {e}")
        arima_interval_coverage[iteration_num] = np.nan

print("Walk-forward validation and ARIMA forecasting complete for Forecast Interval Coverage.")

# Aggregate and analyze the collected KPI for the ARIMA model
print(f"\nAverage Forecast Interval Coverage for ARIMA on {selected_ticker}: {np.nanmean(list(arima_interval_coverage.values())):.2f}")



#To complete the ARIMA model evaluation as requested by the subtask, I will now calculate the maximum drawdown. 
#This involves defining a helper function for this metric and integrating its calculation into the walk-forward validation loop to store and aggregate the results.


import numpy as np
import pandas as pd

# Assuming a risk-free rate of 0% for simplicity in daily returns calculation
# For a more robust financial analysis, a proper risk-free rate should be used.
RISK_FREE_RATE = 0.0

def calculate_sharpe_ratio(returns, risk_free_rate=RISK_FREE_RATE, annualization_factor=252):
    """Calculates the annualized Sharpe Ratio."""
    if returns.empty or returns.std() == 0:
        return np.nan
    excess_returns = returns - risk_free_rate / annualization_factor
    return np.sqrt(annualization_factor) * excess_returns.mean() / excess_returns.std()

def calculate_sortino_ratio(returns, risk_free_rate=RISK_FREE_RATE, annualization_factor=252):
    """Calculates the annualized Sortino Ratio."""
    if returns.empty:
        return np.nan
    excess_returns = returns - risk_free_rate / annualization_factor
    negative_returns = excess_returns[excess_returns < 0]
    if negative_returns.empty or negative_returns.std() == 0:
        return np.nan
    downside_deviation = negative_returns.std()
    return np.sqrt(annualization_factor) * excess_returns.mean() / downside_deviation

def calculate_max_drawdown(returns):
    """Calculates the Maximum Drawdown from a series of returns."""
    if returns.empty:
        return np.nan
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

# Dictionaries to store additional KPIs for ARIMA evaluation
arima_sharpe_ratios = {}
arima_sortino_ratios = {}
arima_interval_coverage = {}
arima_max_drawdowns = {}

print("Continuing Walk-Forward Validation for ARIMA Forecasting to calculate all KPIs...")

# Reusing parameters from the previous walk-forward setup
selected_ticker = 'AAPL'
stock_data = all_stocks_data[selected_ticker][f'{selected_ticker}_Close']
n_observations = len(stock_data)
initial_train_size = int(n_observations * 0.70)
test_window_size = 30
step_size = 10

iteration_num = 0
for i in range(0, n_observations - initial_train_size - test_window_size + 1, step_size):
    iteration_num += 1
    train_end = initial_train_size + i
    test_start = train_end
    test_end = train_end + test_window_size

    if test_end > n_observations:
        test_end = n_observations
        if test_start >= test_end:
            break

    train_data = stock_data.iloc[:train_end]
    test_data = stock_data.iloc[test_start:test_end]

    try:
        # Retrain the ARIMA model on the current `train_data`
        model = ARIMA(train_data, order=arima_order)
        model_fit = model.fit()

        # Generate a forecast for the `test_window_size` period
        forecast_results = model_fit.get_forecast(steps=len(test_data))
        forecast_mean = forecast_results.predicted_mean
        conf_int = forecast_results.conf_int(alpha=0.05) # 95% confidence interval

        # Align forecast index with actual test_data index for proper comparison
        forecast_mean.index = test_data.index
        conf_int.index = test_data.index

        # Store forecast and actuals (already done in previous step, but re-confirming)
        arima_forecasts[iteration_num] = forecast_mean # Use forecast_mean as the point forecast
        arima_actuals[iteration_num] = test_data

        # Calculate actual and forecasted price movements for Directional Accuracy
        actual_movements = np.sign(test_data.diff().dropna())
        forecast_movements = np.sign(forecast_mean.diff().dropna())
        common_index = actual_movements.index.intersection(forecast_movements.index)
        if len(common_index) > 0:
            correct_direction = (forecast_movements.loc[common_index] == actual_movements.loc[common_index]).sum()
            total_predictions = len(common_index)
            directional_accuracy = correct_direction / total_predictions if total_predictions > 0 else 0
            arima_directional_accuracy[iteration_num] = directional_accuracy
        else:
            arima_directional_accuracy[iteration_num] = np.nan

        # Calculate daily returns for both actual and forecasted prices to evaluate Sharpe/Sortino
        actual_returns = test_data.pct_change().dropna()
        # Create a series of returns based on forecast direction applied to actual returns
        strategy_returns = pd.Series(index=actual_returns.index, dtype=float)
        for idx in actual_returns.index:
            if idx in forecast_movements.index:
                if forecast_movements.loc[idx] == 1: # Forecasts increase, so we assume long position
                    strategy_returns.loc[idx] = actual_returns.loc[idx]
                elif forecast_movements.loc[idx] == -1: # Forecasts decrease, so we assume short position
                    strategy_returns.loc[idx] = -actual_returns.loc[idx]
                else: # Forecasts no change
                    strategy_returns.loc[idx] = 0 # No return
            else:
                strategy_returns.loc[idx] = 0 # No forecast for this day

        strategy_returns = strategy_returns.dropna()

        arima_sharpe_ratios[iteration_num] = calculate_sharpe_ratio(strategy_returns)
        arima_sortino_ratios[iteration_num] = calculate_sortino_ratio(strategy_returns)

        # Calculate Forecast Interval Coverage
        lower_bound = conf_int.iloc[:, 0]
        upper_bound = conf_int.iloc[:, 1]

        within_interval = ((test_data >= lower_bound) & (test_data <= upper_bound)).sum()
        total_points = len(test_data)
        coverage = within_interval / total_points if total_points > 0 else np.nan
        arima_interval_coverage[iteration_num] = coverage

        # Calculate Maximum Drawdown for the strategy returns
        arima_max_drawdowns[iteration_num] = calculate_max_drawdown(strategy_returns)

        if iteration_num <= 5: # Print for first 5 iterations to demonstrate
            print(f"Iteration {iteration_num}:")
            print(f"  Training Window: {train_data.index.min().strftime('%Y-%m-%d')} to {train_data.index.max().strftime('%Y-%m-%d')}")
            print(f"  Testing Window:  {test_data.index.min().strftime('%Y-%m-%d')} to {test_data.index.max().strftime('%Y-%m-%d')}")
            print(f"  Forecast Mean (first 5):\n{forecast_mean.head().to_string()}")
            print(f"  Actual values (first 5):\n{test_data.head().to_string()}")
            print(f"  Directional Accuracy: {directional_accuracy:.2f}")
            print(f"  Sharpe Ratio: {arima_sharpe_ratios[iteration_num]:.2f}")
            print(f"  Sortino Ratio: {arima_sortino_ratios[iteration_num]:.2f}")
            print(f"  Forecast Interval Coverage: {coverage:.2f}")
            print(f"  Maximum Drawdown: {arima_max_drawdowns[iteration_num]:.2f}")
            print("--------------------------------------------------")

    except Exception as e:
        print(f"Error in iteration {iteration_num} for {selected_ticker}: {e}")
        arima_directional_accuracy[iteration_num] = np.nan
        arima_sharpe_ratios[iteration_num] = np.nan
        arima_sortino_ratios[iteration_num] = np.nan
        arima_interval_coverage[iteration_num] = np.nan
        arima_max_drawdowns[iteration_num] = np.nan

print("Walk-forward validation and ARIMA forecasting complete for all specified KPIs.")

# After the walk-forward loop, aggregate and analyze the collected KPIs for the ARIMA model
print(f"\nAverage Directional Accuracy for ARIMA on {selected_ticker}: {np.nanmean(list(arima_directional_accuracy.values())):.2f}")
print(f"Average Sharpe Ratio for ARIMA on {selected_ticker}: {np.nanmean(list(arima_sharpe_ratios.values())):.2f}")
print(f"Average Sortino Ratio for ARIMA on {selected_ticker}: {np.nanmean(list(arima_sortino_ratios.values())):.2f}")
print(f"Average Forecast Interval Coverage for ARIMA on {selected_ticker}: {np.nanmean(list(arima_interval_coverage.values())):.2f}")
print(f"Average Maximum Drawdown for ARIMA on {selected_ticker}: {np.nanmean(list(arima_max_drawdowns.values())):.2f}")









