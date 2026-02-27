
#GARCH(1,1) Model for Volatility Estimation

#Implement and train GARCH(1,1) models to estimate and forecast volatility for the selected stock series, considering the squared residuals from the ARIMA models or returns.
# I will import the necessary arch_model function from the arch library and then iterate through the previously calculated daily returns for each stock. 
# For each stock, I will instantiate and fit a GARCH(1,1) model and print its summary, which will address the first steps of the subtask.

from arch import arch_model
import warnings

# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore")

# Dictionary to store fitted GARCH models
garch_models = {}

print("Fitting GARCH(1,1) models for each stock's daily returns...")

for ticker, returns_series in stock_returns.items():
    print(f"\n--- Fitting GARCH(1,1) model for {ticker} Returns ---")

    try:
        # Instantiate and fit GARCH(1,1) model
        # p=1, q=1 for GARCH orders
        # mean='Zero' for modeling returns directly (often assumed to have zero mean)
        # dist='normal' for the error distribution
        model = arch_model(returns_series, mean='Zero', vol='Garch', p=1, q=1, dist='normal')
        model_fit = model.fit(disp='off') # disp='off' suppresses convergence messages
        garch_models[ticker] = model_fit

        print(f"GARCH(1,1) Model Summary for {ticker}:")
        print(model_fit.summary())
    except Exception as e:
        print(f"Could not fit GARCH model for {ticker}. Error: {e}")

print("GARCH(1,1) model fitting complete for all selected stocks.")


import sys
!{sys.executable} -m pip install arch

#Now that the arch library has been successfully installed, I will re-run the code to import arch_model and then iterate through the previously calculated daily returns for each stock.
#  For each stock, I will instantiate and fit a GARCH(1,1) model and print its summary, addressing the subtask requirements.


from arch import arch_model
import warnings

# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore")

# Dictionary to store fitted GARCH models
garch_models = {}

print("Fitting GARCH(1,1) models for each stock's daily returns...")

for ticker, returns_series in stock_returns.items():
    print(f"\n--- Fitting GARCH(1,1) model for {ticker} Returns ---")

    try:
        # Instantiate and fit GARCH(1,1) model
        # p=1, q=1 for GARCH orders
        # mean='Zero' for modeling returns directly (often assumed to have zero mean)
        # dist='normal' for the error distribution
        model = arch_model(returns_series, mean='Zero', vol='Garch', p=1, q=1, dist='normal')
        model_fit = model.fit(disp='off') # disp='off' suppresses convergence messages
        garch_models[ticker] = model_fit

        print(f"GARCH(1,1) Model Summary for {ticker}:")
        print(model_fit.summary())
    except Exception as e:
        print(f"Could not fit GARCH model for {ticker}. Error: {e}")

print("GARCH(1,1) model fitting complete for all selected stocks.")









