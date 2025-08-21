# Electricity Futures Forecasting

**Goal:** Forecast electricity spot/futures prices using statistical models and ML (ARIMA, LightGBM, optional LSTM).  
**Why it matters for a hedge fund:** time series forecasting with exogenous variables (weather, demand, gas), model comparison, and out-of-sample metrics.

---

## Project Structure
- `data/` — sample of the dataset used
- `notebooks/` — EDA and modeling

---

## Models
- Baseline: ARIMA / ARIMAX
- ML: LightGBM with lagged features & calendar features
- (Optional) DL: LSTM/GRU

---

## Metrics
- RMSE, MAE, MAPE  
- Forecast vs. actual plots  
- Rolling window backtesting

---

## Next Steps
- Add exogenous variables (weather, gas)
- Hyperparameter optimization (Bayesian search)
- Extend to multi-market (DE/FR/IT)

*Data note:* the repository does not contain proprietary data. It uses public sources (e.g., ENTSO-E, or proxy data via yfinance for the MVP).

---

## Finance Note: Adjusted Close vs Close

When working with financial data, it’s important to understand the difference between **Close** and **Adjusted Close** prices.

### Close
- The raw last trading price of the day.  
- Does **not** account for dividends or stock splits.  
- Can create misleading “jumps” in the time series.

### Adjusted Close
- The closing price **corrected for dividends and stock splits**.  
- Reflects the **true economic value** of holding the stock/ETF over time.  
- Ensures the series is consistent for return calculations and modeling.

### Why Adjustments Matter
1. **Dividends**  
   - Example: A stock closes at \$100 and pays a \$2 dividend.  
   - Next day, raw Close ≈ \$98 (looks like a drop).  
   - Adjusted Close keeps continuity, showing that the investor’s wealth stayed at \$100 (price + dividend).

2. **Stock Splits**  
   - Example: A 2-for-1 split turns a \$200 share into 2 shares at \$100.  
   - Market cap unchanged.  
   - Adjusted Close prevents the chart from showing a “crash”.

### Why use Adjusted Close in this project?
- For ETFs like **XLE**, dividends are significant.  
- Hedge funds and quants typically use **Adjusted Close** to model **total return** series.  
- Using Adjusted Close avoids spurious jumps and makes forecasting more realistic.

In short:  
- **Close** = what was paid on that day.  
- **Adjusted Close** = what matters for long-term performance and modeling.

---

## Finance & Modeling Note: Lags and Trading Days

### What is a Lag?
In time series modeling, a **lag** means using past values of a variable to explain or forecast the present.

- **Lag 1** = yesterday’s value  
- **Lag 5** = value from 5 days ago  
- **Lag 22** = value from ~1 trading month ago  

Example:  
If today’s electricity price is correlated with the price from 7 days ago, then **lag 7** is useful as a predictor.

Lags help capture **momentum, seasonality, and autocorrelation** in financial data.

---

### What is a Trading Day?
A **trading day** is a day when financial markets are open.  
- In the U.S. stock market, this excludes weekends and holidays.  
- Roughly **252 trading days per year** (not 365).  
- Roughly **21–22 trading days per month**.  

---

### Trading Day vs Calendar Day
- **Calendar day (month day)** = all days in the calendar (30 or 31 days per month).  
- **Trading day (month)** = only the market-open days (~22 per month).  

That’s why in quant finance we often use **22 trading days** as a proxy for “1 month,” not 30 calendar days.  
This avoids mixing in weekends and holidays where no trading (and no price movement) happens.

---

### Why this matters for our project
- When we create lag features like **lag_22**, we are capturing the **effect of the past month of trading**, not the past calendar month.  
- This makes the features more consistent with how hedge funds and financial analysts think about time series.  
- For example: “What was the ETF price 22 trading days ago?” ≈ “What was the price 1 month ago (in trading terms)?”

---

## Theoretical Background — Time Series Forecasting

Time series forecasting is the process of using historical data (ordered in time) to predict future values.  
It is widely applied in **finance, energy markets, insurance, and economics**, where understanding future trends is critical for decision-making.

---

### 1. Key Concepts
- **Time Series (Yₜ):** a sequence of observations ordered in time (e.g., daily electricity prices).  
- **Trend:** long-term increase or decrease in the series.  
- **Seasonality:** repeating patterns (daily, weekly, yearly).  
- **Noise / Shocks:** random fluctuations not explained by structure.  
- **Stationarity:** a time series whose statistical properties (mean, variance, autocorrelation) do not change over time.  
- **Lag:** a past value of the series used as a predictor.  
  - Example: if today is \( t \), then \( Y_{t-1} \) is the value of yesterday (lag-1),  
    \( Y_{t-5} \) is last week (lag-5), and \( Y_{t-22} \) is last month (lag-22).  
  - Lags are crucial because financial and energy data often depend strongly on their past.

---

### 2. Forecasting Approaches
1. **Statistical models**  
   - Use assumptions about data-generating processes.  
   - Examples: ARIMA, SARIMA, VAR.  
   - Strengths: interpretable, mathematically grounded.  
   - Weaknesses: limited in capturing nonlinear dynamics.

2. **Machine Learning models**  
   - Learn patterns directly from data (often with engineered features like lags & calendar variables).  
   - Examples: Random Forests, Gradient Boosting (LightGBM, XGBoost).  
   - Strengths: flexible, capture nonlinear effects, scalable.  
   - Weaknesses: require more data, less interpretable.

3. **Deep Learning models**  
   - Neural networks adapted for sequential data.  
   - Examples: RNN, LSTM, GRU, Transformers.  
   - Strengths: capture long-term dependencies and nonlinearities.  
   - Weaknesses: data-hungry, computationally intensive.

---

### 3. Why This Matters for Energy & Finance
- **Energy markets**: prices depend on demand, weather, fuel costs, and policy. Forecasting helps with **hedging, risk management, and trading strategies**.  
- **Finance & Hedge Funds**: accurate forecasts allow for **better pricing models, arbitrage opportunities, and portfolio optimization**.  
- **Insurance & Risk Models**: time series underpin claims modeling, economic indicators, and solvency forecasting.

---

## Theoretical Background — ARIMA

**ARIMA (AutoRegressive Integrated Moving Average)** is a classical statistical model widely used in finance and econometrics for time series forecasting.  
It combines three components:

- **AR (Autoregressive)**: regression on past values of the series.  
- **I (Integrated)**: differencing to make the series stationary.  
- **MA (Moving Average)**: regression on past forecast errors.  

The model is written as **ARIMA(p, d, q)**, where:  
- *p* = number of autoregressive lags,  
- *d* = number of differences applied for stationarity,  
- *q* = number of moving average lags.

---

### 1. Stationarity
ARIMA assumes stationarity: the mean, variance, and autocorrelation are stable over time.  
If the series is non-stationary (common in prices), we apply **differencing**:  
$\Delta Y_t = Y_t - Y_{t-1}$

---

### 2. AR (AutoRegressive) Part
$Y_t = \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + ... + \phi_p Y_{t-p} + \epsilon_t$
- Current value depends on past values.  
- Coefficients \( \phi_i \) measure persistence.

---

### 3. MA (Moving Average) Part
$Y_t = \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + ... + \theta_q \epsilon_{t-q}$
- Current value depends on past forecast errors.  
- Coefficients \( \theta_j \) measure how shocks propagate.  

---

### 4. ARIMA(p, d, q)
General form after differencing:
$\Delta^d Y_t = \phi_1 \Delta^d Y_{t-1} + ... + \phi_p \Delta^d Y_{t-p} + \epsilon_t + \theta_1 \epsilon_{t-1} + ... + \theta_q \epsilon_{t-q}$

---

### 5. Model Identification
- **ACF/PACF plots** help select p and q.  
- **Information criteria (AIC, BIC)** compare candidate models.  
- Automated tools (`auto_arima`) streamline the process.  

---

### 6. Estimation
Parameters \( \phi_i, \theta_j \) are estimated via **Maximum Likelihood Estimation (MLE)**.

---

### 7. Forecasting
- AR terms propagate past values forward.  
- MA terms use expected future errors = 0.  
- Forecast uncertainty grows with horizon.  

---

### 8. Extensions
- **ARIMAX**: includes exogenous variables (e.g., weather, gas prices).  
- **SARIMA**: adds seasonal components.  
- **SARIMAX**: seasonal ARIMA with exogenous regressors.  
- **VAR**: multivariate extension.

---

### 9. Why ARIMA Matters in Energy & Finance
- Provides a **transparent, interpretable baseline**.  
- Captures autocorrelation and short-term dependencies.  
- Benchmark for more advanced models (LightGBM, LSTM).  
- Well-established in both academic literature and hedge fund practice.

*Limitations:* purely linear, struggles with regime shifts and nonlinear effects, requires stationarity.

---

## Modeling Note: Backtesting & Rolling Forecasts

### Why Backtesting?
In financial and energy markets, it is not enough to split the dataset once into train/test.  
Markets evolve, seasonality shifts, and shocks (e.g., weather, gas prices, policy changes) can strongly affect electricity prices.  

**Backtesting** simulates how a forecasting model would have performed historically if it had been deployed in real time.  
This gives hedge funds and traders a more reliable picture of **out-of-sample performance**.

---

### How Backtesting Works
Instead of training once, we use a **rolling time window**:

1. **Train** the model on a historical window (e.g., 2 years of daily electricity futures).  
2. **Forecast** the next horizon (e.g., 22 trading days ≈ 1 month).  
3. **Roll the window forward** in time and repeat.  
   - Either **fixed length** (rolling window), or  
   - **Expanding length** (always include all past data).  

At each step, the model only uses **past information**, never “peeking into the future.”

---

### Why Not Random Cross-Validation?
- Standard ML cross-validation shuffles the data, which would leak **future prices into the past** — unrealistic in finance.  
- Time series requires **time-aware validation**: train on the past → test on the future.  

This prevents information leakage and makes evaluation closer to **real-world trading conditions**.

---

### Example (Electricity Futures)
Suppose we forecast daily futures prices from **2021 to 2023**:

1. Train: 2021–2022 → Test: Jan 2023  
2. Train: 2021–Feb 2023 → Test: Mar 2023  
3. Train: 2021–Mar 2023 → Test: Apr 2023  
… and so on, until the end of the dataset.  

Each step mimics how a hedge fund would actually use the model: **fit on history, trade on the next period**.

---

### Why This Matters for Hedge Funds
- Provides **robust performance evaluation** across multiple regimes.  
- Shows how the model behaves during **shocks** (spikes, crashes) vs **stable markets**.  
- Enables **strategy stress-testing** before real capital is deployed.  

---

### Key Takeaway
Backtesting in this project uses a **rolling-window forecast evaluation**.  
It is the industry-standard technique in **quant finance and energy trading** to assess whether a forecasting model (ARIMA, ARIMAX, LightGBM) is truly reliable out-of-sample.