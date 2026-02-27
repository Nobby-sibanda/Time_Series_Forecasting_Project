
# Data Loading and Initial Exploration
# import necessary libraries 
#step 1

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define dictionaries to store stock data
sector_stocks = {
    'PG': 'Consumer Staples (Procter & Gamble)',
    'CAT': 'Industrials/Cyclical (Caterpillar)',
    'JPM': 'Financials (JPMorgan Chase)',
    'AAPL': 'Technology (Apple)'
}

dually_listed_stocks = {
    'BHP': 'Dually Listed (BHP Group - NYSE)', # Also listed on ASX
    'RY': 'Dually Listed (Royal Bank of Canada - NYSE)' # Also listed on TSX
}

all_stocks_data = {}


# Download data for sector/economic cycle stocks
print("Downloading data for sector/economic cycle stocks...")
for ticker, name in sector_stocks.items():
    print(f"Fetching {ticker} - {name}")
    data = yf.download(ticker, start='2010-01-01', end='2023-01-01') # Example date range
    if not data.empty:
        all_stocks_data[ticker] = data[['Close']].copy()
        all_stocks_data[ticker].rename(columns={'Close': f'{ticker}_Close'}, inplace=True)
        print(f"First 5 rows of {ticker} data:")
        print(all_stocks_data[ticker].head())
        print("\n")
    else:
        print(f"Could not retrieve data for {ticker}.\n")

# Download data for dually listed stocks
print("Downloading data for dually listed stocks...")
for ticker, name in dually_listed_stocks.items():
    print(f"Fetching {ticker} - {name}")
    data = yf.download(ticker, start='2010-01-01', end='2023-01-01') # Example date range
    if not data.empty:
        all_stocks_data[ticker] = data[['Close']].copy()
        all_stocks_data[ticker].rename(columns={'Close': f'{ticker}_Close'}, inplace=True)
        print(f"First 5 rows of {ticker} data:")
        print(all_stocks_data[ticker].head())
        print("\n")
    else:
        print(f"Could not retrieve data for {ticker}.\n")

print("Data loading complete. All stock data is stored in the 'all_stocks_data' dictionary.")


#The next step in the subtask is to perform initial exploratory data analysis (EDA).
# I will start by checking the data types, handling any missing values, generating descriptive statistics, and then plotting the closing prices for each stock to visually inspect trends and anomalies. 
# I'll use ffill for missing values as it's suitable for time series.


print("Starting Exploratory Data Analysis...")

for ticker, df in all_stocks_data.items():
    print(f"\n--- EDA for {ticker} ---")

    # 1. Check data types and convert index to datetime if not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    print(f"Data types for {ticker}:\n{df.info()}")

    # 2. Identify and handle missing values
    initial_missing = df.isnull().sum().sum()
    if initial_missing > 0:
        print(f"Missing values detected for {ticker}: {initial_missing}")
        df.fillna(method='ffill', inplace=True)
        print(f"Missing values after ffill for {ticker}: {df.isnull().sum().sum()}")
    else:
        print(f"No missing values detected for {ticker}.")

    # 3. Generate descriptive statistics
    print(f"Descriptive statistics for {ticker}_Close:\n{df[f'{ticker}_Close'].describe()}")

    # 4. Plot the time series of 'Close' prices
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[f'{ticker}_Close'], label=f'{ticker} Close Price', color='blue')
    plt.title(f'{ticker} Close Price Time Series')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

print("EDA complete for all stocks.")


# After inspecting the time series trends, the next step in EDA is to analyze the distribution of the closing prices for each stock.
#  Histograms will show the frequency distribution, and box plots will help identify potential outliers and the spread of the data, which is crucial for 
# understanding data characteristics as requested by the subtask.


print("Generating distribution plots (histograms and box plots) for each stock...")

for ticker, df in all_stocks_data.items():
    print(f"\n--- Distribution Analysis for {ticker} ---")

    plt.figure(figsize=(15, 6))

    # Histogram
    plt.subplot(1, 2, 1) # 1 row, 2 columns, first plot
    sns.histplot(df[f'{ticker}_Close'], kde=True, color='skyblue')
    plt.title(f'{ticker} Close Price Distribution (Histogram)')
    plt.xlabel('Close Price')
    plt.ylabel('Frequency')

    # Box Plot
    plt.subplot(1, 2, 2) # 1 row, 2 columns, second plot
    sns.boxplot(y=df[f'{ticker}_Close'], color='lightcoral')
    plt.title(f'{ticker} Close Price Distribution (Box Plot)')
    plt.ylabel('Close Price')

    plt.tight_layout()
    plt.show()

print("Distribution analysis complete.")



#The previous error 'ValueError: If using all scalar values, you must pass an index' in seaborn.boxplot was caused by column names in the DataFrames being a MultiIndex, 
# which seaborn did not handle as expected. 
# The yf.download function often returns a DataFrame with MultiIndex columns (e.g., ('Close', 'PG')), and the original renaming df.rename(columns={'Close': f'{ticker}_Close'}) failed to flatten this. 
# I will modify the data loading step to correctly extract the 'Close' price as a single-level column name, ensuring consistency for subsequent EDA steps.


import yfinance as yf
import pandas as pd

# Define dictionaries to store stock data
sector_stocks = {
    'PG': 'Consumer Staples (Procter & Gamble)',
    'CAT': 'Industrials/Cyclical (Caterpillar)',
    'JPM': 'Financials (JPMorgan Chase)',
    'AAPL': 'Technology (Apple)'
}

dually_listed_stocks = {
    'BHP': 'Dually Listed (BHP Group - NYSE)', # Also listed on ASX
    'RY': 'Dually Listed (Royal Bank of Canada - NYSE)' # Also listed on TSX
}

all_stocks_data = {}

# Download data for sector/economic cycle stocks
print("Downloading data for sector/economic cycle stocks...")
for ticker, name in sector_stocks.items():
    print(f"Fetching {ticker} - {name}")
    # yfinance often returns a MultiIndex for columns. Extract 'Close' and ensure single-level column name.
    data = yf.download(ticker, start='2010-01-01', end='2023-01-01')
    if not data.empty:
        # Extract the 'Close' price series, which will be named after the ticker if MultiIndex is present
        close_series = data['Close']
        # Convert the Series to a DataFrame and assign the desired single-level column name
        df_to_store = pd.DataFrame(close_series)
        df_to_store.columns = [f'{ticker}_Close']

        all_stocks_data[ticker] = df_to_store
        print(f"First 5 rows of {ticker} data:")
        print(all_stocks_data[ticker].head())
        print("\n")
    else:
        print(f"Could not retrieve data for {ticker}.\n")

# Download data for dually listed stocks
print("Downloading data for dually listed stocks...")
for ticker, name in dually_listed_stocks.items():
    print(f"Fetching {ticker} - {name}")
    data = yf.download(ticker, start='2010-01-01', end='2023-01-01')
    if not data.empty:
        # Extract the 'Close' price series
        close_series = data['Close']
        # Convert the Series to a DataFrame and assign the desired single-level column name
        df_to_store = pd.DataFrame(close_series)
        df_to_store.columns = [f'{ticker}_Close']

        all_stocks_data[ticker] = df_to_store
        print(f"First 5 rows of {ticker} data:")
        print(all_stocks_data[ticker].head())
        print("\n")
    else:
        print(f"Could not retrieve data for {ticker}.\n")

print("Data loading complete. All stock data is stored in the 'all_stocks_data' dictionary.")



#The previous codeblock addressed the MultiIndex column issue in the all_stocks_data dictionary. 
# Now that the data is correctly structured with single-level column names, I will re-run the distribution plotting code to generate the histograms and box plots, 
# which should resolve the ValueError and complete the distribution analysis part of the EDA.

print("Generating distribution plots (histograms and box plots) for each stock...")

for ticker, df in all_stocks_data.items():
    print(f"\n--- Distribution Analysis for {ticker} ---")

    plt.figure(figsize=(15, 6))

    # Histogram
    plt.subplot(1, 2, 1) # 1 row, 2 columns, first plot
    sns.histplot(df[f'{ticker}_Close'], kde=True, color='skyblue')
    plt.title(f'{ticker} Close Price Distribution (Histogram)')
    plt.xlabel('Close Price')
    plt.ylabel('Frequency')

    # Box Plot
    plt.subplot(1, 2, 2) # 1 row, 2 columns, second plot
    sns.boxplot(y=df[f'{ticker}_Close'], color='lightcoral')
    plt.title(f'{ticker} Close Price Distribution (Box Plot)')
    plt.ylabel('Close Price')

    plt.tight_layout()
    plt.show()

print("Distribution analysis complete.")



#Stationarity Testing and Preprocessing
#Conduct stationarity tests (e.g., Augmented Dickey-Fuller, KPSS) on the stock price series. 
# Apply appropriate transformations (e.g., differencing) to achieve stationarity if required for classical models.
#The first step is to import the necessary libraries for stationarity tests (ADF and KPSS) and define helper functions to interpret their results, which will make the output clearer and more consistent.

from statsmodels.tsa.stattools import adfuller, kpss
import numpy as np

def interpret_adf(adf_output):
    dickey_fuller_test = adf_output[0]
    p_value = adf_output[1]
    critical_values = adf_output[4]

    print(f'\tADF Statistic: {dickey_fuller_test:.2f}')
    print(f'\tP-value: {p_value:.3f}')
    print('\tCritical Values:')
    for key, value in critical_values.items():
        print(f'\t\t{key}: {value:.2f}')
    
    if p_value <= 0.05 and dickey_fuller_test < critical_values['5%']:
        print('\tConclusion: Series is likely stationary (reject H0)')
    else:
        print('\tConclusion: Series is likely non-stationary (fail to reject H0)')

def interpret_kpss(kpss_output):
    kpss_statistic = kpss_output[0]
    p_value = kpss_output[1]
    critical_values = kpss_output[3]

    print(f'\tKPSS Statistic: {kpss_statistic:.2f}')
    print(f'\tP-value: {p_value:.3f}')
    print('\tCritical Values:')
    for key, value in critical_values.items():
        print(f'\t\t{key}: {value:.2f}')
    
    if p_value >= 0.05:
        print('\tConclusion: Series is likely stationary (fail to reject H0)')
    else:
        print('\tConclusion: Series is likely non-stationary (reject H0)')

print("Stationarity test functions imported and defined.")


# Now that the helper functions for interpreting stationarity tests are defined, I will iterate through each stock, perform ADF and KPSS tests on the original series, 
# apply first-order differencing if the series is non-stationary, re-run the tests on the differenced series, and then plot both the original and differenced series for visual inspection.


import matplotlib.pyplot as plt

# Dictionary to store differenced series if needed
differenced_stocks_data = {}

print("Performing stationarity tests and applying differencing if necessary...")

for ticker, df in all_stocks_data.items():
    print(f"\n--- Stationarity Analysis for {ticker} ---")
    
    close_prices = df[f'{ticker}_Close']
    
    # Store the original series for plotting later
    original_series = close_prices.copy()

    # 1. ADF Test on Original Series
    print(f"\nADF Test on Original {ticker} Close Prices:")
    adf_result_original = adfuller(close_prices)
    interpret_adf(adf_result_original)

    # 2. KPSS Test on Original Series
    print(f"\nKPSS Test on Original {ticker} Close Prices:")
    kpss_result_original = kpss(close_prices, regression='c') # 'c' for constant
    interpret_kpss(kpss_result_original)
    
    is_stationary_adf = (adf_result_original[1] <= 0.05 and adf_result_original[0] < adf_result_original[4]['5%'])
    is_stationary_kpss = (kpss_result_original[1] >= 0.05)

    # Determine if differencing is needed based on common interpretations:
    # - ADF fails to reject H0 (non-stationary) AND KPSS rejects H0 (non-stationary) -> non-stationary
    # - ADF rejects H0 (stationary) AND KPSS fails to reject H0 (stationary) -> stationary
    # - Both reject H0 -> differencing might be needed (contradictory, often means trend stationarity for ADF and difference stationarity for KPSS)
    # - Both fail to reject H0 -> differencing might be needed (contradictory, often means non-stationary for both)
    
    # For simplicity, we'll difference if ADF indicates non-stationarity or KPSS indicates non-stationarity.
    # A more robust approach would consider the interaction, but for stock prices, differencing is a common first step.
    
    perform_differencing = False
    if not is_stationary_adf or not is_stationary_kpss: # If either test suggests non-stationarity
        perform_differencing = True
        print(f"\nOriginal series for {ticker} appears non-stationary. Applying first-order differencing...")
        
        # Apply first-order differencing
        differenced_series = close_prices.diff().dropna()
        differenced_stocks_data[f'{ticker}_Diff'] = differenced_series

        # 3. Re-run ADF Test on Differenced Series
        print(f"\nADF Test on Differenced {ticker} Close Prices:")
        adf_result_diff = adfuller(differenced_series)
        interpret_adf(adf_result_diff)

        # 4. Re-run KPSS Test on Differenced Series
        print(f"\nKPSS Test on Differenced {ticker} Close Prices:")
        kpss_result_diff = kpss(differenced_series, regression='c')
        interpret_kpss(kpss_result_diff)
    else:
        print(f"\nOriginal series for {ticker} appears stationary. No differencing applied.")

    # 5. Plot original and differenced series (if differenced)
    plt.figure(figsize=(14, 7))
    plt.plot(original_series.index, original_series, label=f'{ticker} Original Close Price', color='blue')
    
    if perform_differencing:
        plt.plot(differenced_series.index, differenced_series, label=f'{ticker} First Difference', color='red', linestyle='--')
        plt.title(f'{ticker} Original vs. First Differenced Close Prices')
    else:
        plt.title(f'{ticker} Original Close Prices (Stationary)')
        
    plt.xlabel('Date')
    plt.ylabel('Price / Difference')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

print("Stationarity testing and differencing complete for all stocks.")


#The first step is to import the necessary libraries for stationarity tests (ADF and KPSS) and define helper functions to interpret their results, 
# which will make the output clearer and more consistent.


from statsmodels.tsa.stattools import adfuller, kpss
import numpy as np

def interpret_adf(adf_output):
    dickey_fuller_test = adf_output[0]
    p_value = adf_output[1]
    critical_values = adf_output[4]

    print(f'\tADF Statistic: {dickey_fuller_test:.2f}')
    print(f'\tP-value: {p_value:.3f}')
    print('\tCritical Values:')
    for key, value in critical_values.items():
        print(f'\t\t{key}: {value:.2f}')
    
    if p_value <= 0.05 and dickey_fuller_test < critical_values['5%']:
        print('\tConclusion: Series is likely stationary (reject H0)')
    else:
        print('\tConclusion: Series is likely non-stationary (fail to reject H0)')

def interpret_kpss(kpss_output):
    kpss_statistic = kpss_output[0]
    p_value = kpss_output[1]
    critical_values = kpss_output[3]

    print(f'\tKPSS Statistic: {kpss_statistic:.2f}')
    print(f'\tP-value: {p_value:.3f}')
    print('\tCritical Values:')
    for key, value in critical_values.items():
        print(f'\t\t{key}: {value:.2f}')
    
    if p_value >= 0.05:
        print('\tConclusion: Series is likely stationary (fail to reject H0)')
    else:
        print('\tConclusion: Series is likely non-stationary (reject H0)')

print("Stationarity test functions imported and defined.")


 #Now that the helper functions for interpreting stationarity tests are defined, I will iterate through each stock, perform ADF and KPSS tests on the original series, 
 # apply first-order differencing if the series is non-stationary, re-run the tests on the differenced series, and then plot both the original and differenced series for visual inspection, as per the subtask instructions.


import matplotlib.pyplot as plt

# Dictionary to store differenced series if needed
differenced_stocks_data = {}

print("Performing stationarity tests and applying differencing if necessary...")

for ticker, df in all_stocks_data.items():
    print(f"\n--- Stationarity Analysis for {ticker} ---")
    
    close_prices = df[f'{ticker}_Close']
    
    # Store the original series for plotting later
    original_series = close_prices.copy()

    # 1. ADF Test on Original Series
    print(f"\nADF Test on Original {ticker} Close Prices:")
    adf_result_original = adfuller(close_prices)
    interpret_adf(adf_result_original)

    # 2. KPSS Test on Original Series
    print(f"\nKPSS Test on Original {ticker} Close Prices:")
    kpss_result_original = kpss(close_prices, regression='c') # 'c' for constant
    interpret_kpss(kpss_result_original)
    
    is_stationary_adf = (adf_result_original[1] <= 0.05 and adf_result_original[0] < adf_result_original[4]['5%'])
    is_stationary_kpss = (kpss_result_original[1] >= 0.05)

    # Determine if differencing is needed based on common interpretations:
    # For simplicity, we'll difference if ADF indicates non-stationarity or KPSS indicates non-stationarity.
    # A more robust approach would consider the interaction, but for stock prices, differencing is a common first step.
    
    perform_differencing = False
    if not is_stationary_adf or not is_stationary_kpss: # If either test suggests non-stationarity
        perform_differencing = True
        print(f"\nOriginal series for {ticker} appears non-stationary. Applying first-order differencing...")
        
        # Apply first-order differencing
        differenced_series = close_prices.diff().dropna()
        differenced_stocks_data[f'{ticker}_Diff'] = differenced_series

        # 3. Re-run ADF Test on Differenced Series
        print(f"\nADF Test on Differenced {ticker} Close Prices:")
        adf_result_diff = adfuller(differenced_series)
        interpret_adf(adf_result_diff)

        # 4. Re-run KPSS Test on Differenced Series
        print(f"\nKPSS Test on Differenced {ticker} Close Prices:")
        kpss_result_diff = kpss(differenced_series, regression='c')
        interpret_kpss(kpss_result_diff)
    else:
        print(f"\nOriginal series for {ticker} appears stationary. No differencing applied.")

    # 5. Plot original and differenced series (if differenced)
    plt.figure(figsize=(14, 7))
    plt.plot(original_series.index, original_series, label=f'{ticker} Original Close Price', color='blue')
    
    if perform_differencing:
        plt.plot(differenced_series.index, differenced_series, label=f'{ticker} First Difference', color='red', linestyle='--')
        plt.title(f'{ticker} Original vs. First Differenced Close Prices')
    else:
        plt.title(f'{ticker} Original Close Prices (Stationary)')
        
    plt.xlabel('Date')
    plt.ylabel('Price / Difference')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

print("Stationarity testing and differencing complete for all stocks.")


#The previous code produced InterpolationWarning messages from the kpss function because the test statistics were outside the range of its internal p-value lookup table.
#To suppress these warnings and keep the output clean, I will add a warning filter specifically for InterpolationWarning before running the stationarity tests.

import matplotlib.pyplot as plt
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning

# Dictionary to store differenced series if needed
differenced_stocks_data = {}

print("Performing stationarity tests and applying differencing if necessary...")

for ticker, df in all_stocks_data.items():
    print(f"\n--- Stationarity Analysis for {ticker} ---")
    
    close_prices = df[f'{ticker}_Close']
    
    # Store the original series for plotting later
    original_series = close_prices.copy()

    # 1. ADF Test on Original Series
    print(f"\nADF Test on Original {ticker} Close Prices:")
    adf_result_original = adfuller(close_prices)
    interpret_adf(adf_result_original)

    # 2. KPSS Test on Original Series
    print(f"\nKPSS Test on Original {ticker} Close Prices:")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', InterpolationWarning)
        kpss_result_original = kpss(close_prices, regression='c') # 'c' for constant
    interpret_kpss(kpss_result_original)
    
    is_stationary_adf = (adf_result_original[1] <= 0.05 and adf_result_original[0] < adf_result_original[4]['5%'])
    is_stationary_kpss = (kpss_result_original[1] >= 0.05)

    # Determine if differencing is needed based on common interpretations:
    # For simplicity, we'll difference if ADF indicates non-stationarity or KPSS indicates non-stationarity.
    # A more robust approach would consider the interaction, but for stock prices, differencing is a common first step.
    
    perform_differencing = False
    if not is_stationary_adf or not is_stationary_kpss: # If either test suggests non-stationarity
        perform_differencing = True
        print(f"\nOriginal series for {ticker} appears non-stationary. Applying first-order differencing...")
        
        # Apply first-order differencing
        differenced_series = close_prices.diff().dropna()
        differenced_stocks_data[f'{ticker}_Diff'] = differenced_series

        # 3. Re-run ADF Test on Differenced Series
        print(f"\nADF Test on Differenced {ticker} Close Prices:")
        adf_result_diff = adfuller(differenced_series)
        interpret_adf(adf_result_diff)

        # 4. Re-run KPSS Test on Differenced Series
        print(f"\nKPSS Test on Differenced {ticker} Close Prices:")
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', InterpolationWarning)
            kpss_result_diff = kpss(differenced_series, regression='c')
        interpret_kpss(kpss_result_diff)
    else:
        print(f"\nOriginal series for {ticker} appears stationary. No differencing applied.")

    # 5. Plot original and differenced series (if differenced)
    plt.figure(figsize=(14, 7))
    plt.plot(original_series.index, original_series, label=f'{ticker} Original Close Price', color='blue')
    
    if perform_differencing:
        plt.plot(differenced_series.index, differenced_series, label=f'{ticker} First Difference', color='red', linestyle='--')
        plt.title(f'{ticker} Original vs. First Differenced Close Prices')
    else:
        plt.title(f'{ticker} Original Close Prices (Stationary)')
        
    plt.xlabel('Date')
    plt.ylabel('Price / Difference')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

print("Stationarity testing and differencing complete for all stocks.")













