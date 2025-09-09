# 📊 Crypto & Energy Market Dashboard

This is a fully interactive **financial analytics dashboard** developed using **Streamlit**, designed to analyze the relationship between **cryptocurrencies** and **energy markets**. It incorporates **GARCH models**, **copula theory**, and **deep learning (LSTM)** for crash analysis, forecasting, and strategic decision-making.

---

## 📘 Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Setup Instructions](#setup-instructions)
4. [Data Format](#data-format)
5. [Code Structure & Explanation](#code-structure--explanation)
6. [Visual Output Samples](#visual-output-samples)
7. [Tech Stack](#tech-stack)
8. [License](#license)

---

## 📌 Project Overview

With increasing volatility in global energy markets and rapid adoption of cryptocurrencies, this dashboard answers critical questions like:

- Can crypto assets serve as **hedges** or **safe havens** during energy market crashes?
- How **dependent** are their returns in volatile regimes?
- Can we **predict short-term returns** with machine learning?

---

## 🚀 Features

| Section | Description |
|--------|-------------|
| 📈 Return Distributions | Plots and compares the return behaviors of cryptos vs selected energy markets. |
| 📉 Crash Detection | Detects energy market crashes using drawdown logic. |
| 📊 GARCH Volatility | Uses ARCH/GARCH modeling to assess volatility regimes. |
| 🔗 Copula Analysis | Measures tail dependence between cryptos and energy markets. |
| 🧠 LSTM Forecasting | Deep learning-based return direction predictor. |
| 🧪 Strategy Backtesting | Backtests a basic long-short strategy using predicted returns. |

---

## ⚙️ Setup Instructions

```bash
git clone https://github.com/YOUR_USERNAME/crypto-energy-dashboard.git
cd crypto-energy-dashboard
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Data Format

Upload an Excel file with:
- **Sheet Name**: `Sheet4_LogReturns`
- **Required Columns**:
    - `date`
    - Cryptos: `BTC`, `ETH`, `ADA`, `LTC`, `DOT`, `SOL`
    - Energies: e.g., `NG`, `CL`, `ELC`, etc.

---

## 🔍 Code Structure & Explanation

### 1. GARCH Volatility

```python
def fit_garch(series):
    model = arch_model(series, vol='Garch', p=1, q=1)
    res = model.fit(disp='off')
    return res.conditional_volatility
```
📌 **Why?** GARCH captures volatility clustering, a common property in financial returns.

---

### 2. Crash Detection

```python
rolling_max = cumulative_energy.cummax()
drawdown = cumulative_energy - rolling_max
drawdown_pct = drawdown / rolling_max
```
📌 Identifies drawdowns > 20% as market crashes.

---

### 3. Copula Estimation

```python
cop.select(data, controls=controls)
lambda_L = np.mean((u < 0.05) & (v < 0.05))
```
📌 Tail dependence metrics are crucial to identifying **"safe haven"** potential.

---

### 4. LSTM Prediction Logic

```python
model = Sequential()
model.add(LSTM(units, return_sequences=True, input_shape=(X_train.shape[1], 1)))
```
📌 Deep learning is used for predicting return directions.

---

## 🖼️ Visual Output Samples

### 🔸 Return Distribution
![Return Distribution](assets/sample_return_distribution.png)

### 🔸 Crash Highlighting
![Crash Highlight](assets/sample_crash_highlight.png)

### 🔸 Copula Dependence Classification
![Copula Classification](assets/sample_copula_result.png)

### 🔸 LSTM Performance Summary
![LSTM Strategy](assets/sample_lstm_summary.png)

> 📷 Add your screenshots to `/assets/` and replace the image file names above.

---

## 🧰 Tech Stack

- Python, Streamlit
- arch, pyvinecopulib
- scikit-learn, tensorflow.keras
- matplotlib, seaborn, plotly

---

## 📄 License

MIT License — free to use and adapt with attribution.

---
