# Electricity Futures Forecasting 🔌📈

**Goal:** Forecast electricity spot/futures prices using statistical models and ML (ARIMA, LightGBM, optional LSTM).  
**Why it matters for a hedge fund:** time series forecasting with exogenous variables (weather, demand, gas), model comparison, and out-of-sample metrics.

## Project Structure
- `data/` — raw & processed (ignored by git)
- `notebooks/` — EDA and modeling
- `src/` — pipeline, features, models, utils
- `results/` — charts and reports (ignored by git)

## Models
- Baseline: ARIMA / ARIMAX
- ML: LightGBM with lagged features & calendar features
- (Optional) DL: LSTM/GRU

## Metrics
RMSE, MAE, MAPE; forecast vs. actual plots; rolling window backtesting.

## Next Steps
- Add exogenous variables (weather, gas)
- Hyperparameter optimization (Bayesian)
- Extend to multi-market (DE/FR/IT)

*Data note:* the repository does not contain proprietary data. It uses public sources (e.g., ENTSO-E, or proxy data via yfinance for the MVP).
