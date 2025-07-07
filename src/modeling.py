import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error


def train_sarima(train_series, order, seasonal_order, enforce_stationarity=True, enforce_invertibility=True):
    """
    Fits a SARIMA model to the training data.
    """
    model = SARIMAX(train_series,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=enforce_stationarity,
                    enforce_invertibility=enforce_invertibility)
    results = model.fit(disp=False)
    return results


def forecast_and_evaluate(model, test_series):
    """
    Generates forecast and calculates RMSE.
    """
    forecast = model.get_forecast(steps=len(test_series))
    predicted_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    rmse = np.sqrt(mean_squared_error(test_series, predicted_mean))
    return predicted_mean, forecast_ci, rmse


def plot_forecast(train, test, predicted, conf_int, title="", ylabel="Market Share (%)"):
    """
    Plots the forecast vs actual data.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train.index, train, label="Train")
    plt.plot(test.index, test, label="Actual", color="green")
    plt.plot(test.index, predicted, label="Forecast", color="red", linestyle="--")
    plt.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_full_forecast(ts, name, order, seasonal_order, split_date='2020'):
    """
    Splits the data, runs SARIMA, evaluates and plots results.
    """
    train = ts[:split_date]
    test = ts[pd.to_datetime(split_date).year + 1:]

    model = train_sarima(train, order, seasonal_order)
    predicted, conf_int, rmse = forecast_and_evaluate(model, test)

    title = f"{name} Forecast\nSARIMA{order}x{seasonal_order}, RMSE={rmse:.4f}"
    plot_forecast(train, test, predicted, conf_int, title)
    
    return model, rmse
