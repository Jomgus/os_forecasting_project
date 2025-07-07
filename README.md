# OS Forecasting Project
This notebook documents the exploration of optimizing three autoregression models through an understanding and tweaking of approaches and parameters. Using data from the steam hardware survey, trends for Linux, MacOS, and Windows are independently forecasted to an optimally low RMSE and visually satisfactory forecasting trend.


## Overview
This project aims to forecast the market share of three operating systems (Linux, macOS, and Windows) using monthly time series data from the [Steam Hardware Survey](https://github.com/myagues/steam-hss-data/releases). The dataset contains aggregated OS usage percentages collected from 2014 to 2021.

The approach decided on for this forecasting problem was from a time series regression task standpoint, treating each OS independently. Based on seasonal patterns in the data, Seasonal ARIMA (SARIMA) models are chosen as the core method. For each platform, SARIMA hyperparameters were tuned based on residual patterns and RMSE validation performance. RMSE was chosen as it is more sensitive to deviation from trends than MSE.

The best models for each OS achieved variances as low as

| OS          | RMSE   |
| ----------- | ------ |
| **Linux**   | 0.1016 |
| **macOS**   | 0.4356 |
| **Windows** | 0.3759 |

### Data
This project uses monthly OS market share data collected from the [Steam Hardware Survey](https://github.com/myagues/steam-hss-data/releases) between January 2014 to December 2021. The dataset was filtered and transformed to isolate three platforms: Linux, macOS, and Windows. The raw file is a compressed parquet file with a shape of 262,470 rows × 5 columns and a size of 700 KB.

#### Preprocessing
To transform the data to be used in our models we had to find an appropriate monthly aggregate, the "OS Version (total)" category, and then pivot it to form a tidy monthly timeseries per OS. Cleaned and processed datasets are saved to the data/ directory.

### Visuals
To assess the presence of seasonality in the OS market share data, seasonal decompositions using statsmodels were applied. This helped confirm the suitability of seasonal autoregressive models (SARIMA).

![Linux Decomp](reports/linux_decomposition)

Final Model Forecasts

![Linux Final](reports/linux_final)
![Windows Final](reports/windows_final)
![MacOS Final](reports/osx_final)

Visualized for each platform:
- Training data (up to 2020)
- True values from 2021
- SARIMA forecast with 95% confidence intervals
These plots help communicate model fit

### The Problem
What is given is the aggregated monthly data for the market share of each OS from January 2014 to December 2021. What is desired is a reliable forecast for the last year of the given data for every OS. 

The starting approach of detecting and implementing seasonality into an autoregression model was made possible by the pmdarima library. First it became clear that the model would not detect the data as seasonal automatically, which cast some doubts as to the efficacy of this approach. Without seasonality the model was flat, a lifeless straight line eminating from the last datapoint. Forcing seasonality into the model by writing in the parameters yielded impressive results. Firstly, it confirmed through residual plots that the model stuck much closer to the actual data. When testing the new model through training data, RMSE was lowered and the forecast seemed to weave a reliable narrative for every OS trend. 

Sticking with SARIMA and changing parameters where they made sense lowered our RMSE even still, further establishing the reliability of seasonal autoregression for this task. By narrowing the parameters down through these tables:

| Series  | Seasonal Order | RMSE     |
| ------- | -------------- | -------- |
| OSX     | (1, 1, 1, 12)  | 0.514426 |
| OSX     | (0, 1, 1, 12)  | 0.539559 |
| OSX     | (1, 0, 1, 12)  | 0.568680 |
| OSX     | (1, 1, 0, 12)  | 0.598024 |
| OSX     | (0, 1, 0, 12)  | 0.879010 |
| Windows | (1, 1, 1, 12)  | 0.518014 |
| Windows | (0, 1, 1, 12)  | 0.553314 |
| Windows | (1, 1, 0, 12)  | 0.566934 |
| Windows | (1, 0, 1, 12)  | 0.635486 |
| Windows | (0, 1, 0, 12)  | 0.884149 |

and

| Series  | Order     | Seasonal Order | Stationarity Enforced | Invertibility Enforced | RMSE     |
| ------- | --------- | -------------- | --------------------- | ---------------------- | -------- |
| OSX     | (1, 0, 1) | (1, 1, 1, 12)  | False                 | False                  | 0.435617 |
| OSX     | (1, 0, 1) | (1, 1, 1, 12)  | True                  | True                   | 0.514426 |
| Windows | (1, 0, 1) | (1, 1, 1, 12)  | False                 | False                  | 0.375858 |
| Windows | (1, 0, 1) | (1, 1, 1, 12)  | True                  | True                   | 0.518014 |

RMSE was able to go from:
Linux: ARIMA RMSE of 0.1294 to SARIMA RMSE of 0.1016
Windows: ARIMA RMSE of 0.5900 to SARIMA RMSE of 0.3759
macOS: ARIMA RMSE of 0.6436 to SARIMA RMSE of 0.4356

These results validate the use of seasonal autoregressive modeling for forecasting platform market share on Steam, particularly when seasonality is enforced and tuned explicitly. All three SARIMA models prove vastly more predictive than the original ARIMA models. These models stand as a testament to the impressive forecasting work that can be done using nothing but past data points. 

### Future Work

- **Forecasting beyond 2021**
Finding a dataset that scraped the annual steam hardware survey and was updated consistently proved difficult, those that were totally up to date were missing essential categories needed to forecast the data as was done here. The best next step for these models would be to have this dataset updated so that the forecasts can be extended and evaluate them against the survey for years 2022-2025.

- **exogenous variables**
It is interesting work to see what can be done with simple datapoints, however, SARIMAX does provide the ability to work with exogenous variables, market share is very nuanced and including some extra datapoints would help accuracy. Some examples could be:
  - Major OS updates (e.g., Windows 11 launch)
  - Steam client support changes
  - Gaming hardware trends (e.g., introduction of ARM-based Macs)
  - Global events (e.g., COVID-era supply chain disruptions)

## Reproduce Results

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Jomgus/os_forecasting_project.git
   ```
2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
3. **Run Preprocessing:**

  - Open and run notebooks/preprocessing.ipynb. This will:
  - Load raw parquet data from data/raw/
  - Filter for OS totals
  - Pivot and clean it
  - Save final data to data/processed/os_monthly_marketshare.csv
4. Train models and evaluate:

  Run notebooks/Production.ipynb. This notebook:
- Imports modeling functions from src/sarima_utils.py
- Trains SARIMA models on each OS
- Tests and compares seasonal configs
- Outputs diagnostics, RMSE tables, and plots to reports/

### Overview of Files in Repository

This project follows a modular structure to separate data, modeling, utilities, and outputs.

**Project Directory**
- data/raw/: Raw Steam OS data (.parquet)
- data/processed/: Cleaned monthly market share CSV
- models/: Trained .pkl models and rmse_summary.csv
- reports/: Plots and visuals (forecasts, diagnostics)
- src/sarima_utils.py: All forecasting and evaluation functions
- notebooks/preprocessing.ipynb: Loads and reshapes data
- notebooks/Production.ipynb: Main SARIMA training + analysis
- requirements.txt: Dependency list
- README.md: Documentation

#### File Roles

- **`sarima_utils.py`**: Defines `train_sarima`, `forecast_and_evaluate`, `compute_rmse`, and `plot_forecast`. Core logic is imported into notebooks.
- **`preprocessing.ipynb`**: Filters raw Steam data, pivots by OS, handles missing values, and exports clean data.
- **`Production.ipynb`**: Forecasts Linux, macOS, and Windows market share. Performs seasonal parameter tuning and visual diagnostics.
- **`rmse_summary.csv`**: Compares RMSEs from different seasonal configurations.


### Citations
Steam Hardware Survey Data:
https://github.com/myagues/steam-hss-data

pmdarima documentation:
https://alkaline-ml.com/pmdarima/

