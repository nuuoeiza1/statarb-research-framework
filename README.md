# Intraday Cointegration Model for Low-Capital Statistical Arbitrage

[![MSc Thesis](https://img.shields.io/badge/KIT_M.Sc._Thesis-Financial_Engineering-blue.svg)](#)
[![Python](https://img.shields.io/badge/Python-brightgreen.svg)](#)
[![Domain](https://img.shields.io/badge/Domain-Statistical_Arbitrage-orange.svg)](#)
[![Sharpe OOS](https://img.shields.io/badge/OOS_Sharpe-9.85-brightgreen.svg)](#)

> **M.Sc. Financial Engineering Thesis**  
> **Institution:** Karlsruhe Institute of Technology (KIT), HECTOR School of Engineering & Management, Germany  
> **Author:** Ratchatapong Lukmuang  

---

## 📌 Notice on Code Availability & IP Protection

To protect proprietary research and prevent **alpha decay**, live execution scripts, signal parameters, raw tick data, and fitted model weights are **intentionally omitted** from this public repository. 

This repository serves as a high-level technical overview of the quantitative framework, methodology, and system architecture developed for the thesis.

👉 **Read the full thesis write-up, econometric analyses, and empirical results on LinkedIn:**  
🔗 [Inside My M.Sc. Thesis: Building Algo-Trading at 9.85 Sharpe via Statistical Arbitrage for $100–$1,000 Retail Accounts](https://www.linkedin.com/pulse/inside-my-msc-thesis-building-algo-trading-985-sharpe-lukmuang-mgfif/)

---

## 📊 Empirical Performance Summary

All backtest metrics are reported **net of real-world frictions**, including a $0.04 round-trip commission per share/lot, variable bid-ask spreads, overnight swap avoidance, and market-close protections.

### In-Sample (IS) vs. Out-of-Sample (OOS) Performance

| Metric | Champion (IS) | SPY B&H (IS) | Champion (OOS) | SPY B&H (OOS) |
| :--- | :---: | :---: | :---: | :---: |
| **Testing Window** | Nov 1, 2023 – Oct 21, 2025 | Nov 1, 2023 – Oct 21, 2025 | Oct 21, 2025 – Mar 31, 2026 | Oct 21, 2025 – Mar 31, 2026 |
| **Net PnL (%)** | **+16.15%** | +58.88% | **+6.06%** | -5.02% |
| **Sharpe Ratio** | **4.87** | 1.61 | **9.85** | -0.84 |
| **Sortino Ratio** | **37.98** | 2.14 | **65.45** | -1.32 |
| **Calmar Ratio** | **293.27** | 2.13 | **126.03** | -0.53 |
| **Max Drawdown** | **-0.06%** | -27.67% | **-0.05%** | -9.40% |

### Additional OOS Execution & Risk Metrics
* **OOS Win Rate:** 90.12% (165 trades)
* **Risk of Ruin:** 0% (Validated via 10,000-path Monte Carlo simulation)
* **Account Size Target:** $100 – $1,000 retail base capital

---

## 🛠 Framework & Research Overview

### 1. Market Pair & Broker Constraints
* **Traded Pair:** `SPY.US CFD` vs. `US500 CFD` (Pepperstone MetaTrader 5 API).
* **Capital Base:** Designed for $100 to $1,000 retail accounts (unaffected by US PDT rules via European CFD access).
* **Friction Model:** $0.04 round-trip commission per lot on `SPY.US` (breakeven threshold ~0.40 points).

### 2. Data Engineering & Cleansing
* **Dataset:** ~2.5 years of tick-level data stored in Parquet format.
* **Spike Filter:** Purges tick anomalies exceeding $5 \times$ the 99th percentile rolling bid-ask spread.
* **Resampling Grid:** 5-second sampling grid chosen to balance temporal resolution against "Market Silence" (achieving 92–94% fidelity without introducing phantom signals).
* **Dataset Split:** Chronological 80/20 split (In-Sample: Nov 2023 – Oct 2025, Out-of-Sample: Oct 2025 – Mar 31, 2026).

### 3. Model Selection & Cointegration Matrix
Evaluated 7 econometric candidates across linear (EG-ECM), dynamic-linear (Kalman Filter + OU), and non-linear (Gaussian Copula) families. Model selection was driven by a novel evaluation metric:

$$\text{CTES} = \text{Retained Rate} \times \frac{ \text{Zero-Crossing Rate}}{\text{Half-Life} \times \text{Hurst Exponent}}$$

* **Champion Model:** `Copula_OU (AR1)` at 1-minute resolution (CTES: 761.43, mean half-life: ~30 seconds).
* **Rejected Models:** GARCH (rejected by ARCH-LM test) and SETAR (rejected by Chow test) due to lack of empirical support for added complexity.

### 4. Asynchronous Multi-Timeframe Architecture
To capture mean-reversion windows of ~30 seconds without bottlenecking statistical calculations, execution is split into two asynchronous modules:
* **The Brain (1-minute):** Calculates heavy statistics (cointegration parameters, copula marginals, OU constants).
* **The Action (5-second):** Monitors price spreads at microstructure resolution (92–94% fidelity) and executes trades authorized by *The Brain*.
<p align="center">
  <a href="https://github.com/user-attachments/assets/398933db-65c1-4cda-ba44-fe4c34c3fb5f" target="_blank">
    <img src="https://github.com/user-attachments/assets/398933db-65c1-4cda-ba44-fe4c34c3fb5f" alt="System Architecture" width="380" />
  </a>
  <br />
  <sub>🔍 <i>Click image to enlarge</i></sub>
</p>

  
                                          +---------------------------+
                                          |    Raw Tick Data Feed     |
                                          +---------------------------+
                                                        |
                                                        v
                                          +---------------------------+
                                          |Data Preprocessing Pipeline|
                                          +---------------------------+
                                                        |
                                                        v
                     +----------------------------------+----------------------------------+
                     | Brain Module (1-minute TF)       |                                  |
                     |                                  v                                  |
                     |                         /-----------------\                         |
                     |                        /                   \                        |
                     |                       < Sampling Frequency  >                       |
                     |                        \                   /                        |
                     |                         \-----------------/                         |
                     |                           /             \                           |
                     |                          /               \                          |
                     |                         v                 \                         |
                     |       +-------------------+                \                        |
                     |       | 1-min OHLC        |                 \                       |
                     |       | Resampling        |                  \                      |
                     |       +-------------------+                   \                     |
                     |                 |                              \                    |
                     |                 v                               \                   |
                     |       +-------------------+                      \                  |
                     |       | Champion Model:   |                       |                 |
                     |       | Copula_OU AR1     |                       |                 |
                     |       +-------------------+                       |                 |
                     |                 |                                 |                 |
                     |                 v                                 |                 |
                     |       +-------------------+                       |                 |
                     |       | Sequential        |                       |                 |
                     |       | Statistical Gating|                       |                 |
                     |       +-------------------+                       |                 |
                     |                 |                                 |                 |
                     |                 v                                 |                 |
                     |       +-------------------+                       |                 |
                     |       | Parameter Calib:  |                       |                 |
                     |       | Hedge Ratio/MI/OU |                       |                 |
                     |       +-------------------+                       |                 |
                     |                 |                                 |                 |
                     |                 v                                 |                 |
                     |       +-------------------+                       |                 |
                     |       | Generate Trade    |                       |                 |
                     |       | Signal Bundle     |                       |                 |
                     |       +-------------------+                       |                 |
                     +-----------------|---------------------------------|-----------------+
                                       |                                 |
                                       |                                 v
                                       |               +-----------------------------------+
                                       |               | Action Module (5-second TF)       |
                                       |               |                                   |
                                       |               |       +-------------------+       |
                                       |               |       | 5-sec OHLC        |       |
                                       |               |       | Resampling        |       |
                                       |               |       +-------------------+       |
                                       |               |                 |                 |
                                       |               |                 v                 |
                                       |               |       +-------------------+       |
                                       |               |       | Monitor Micro-    |       |
                                       |               |       | Spread Convergence|       |
                                       |               |       +-------------------+       |
                                       |               |                 |                 |
                                       |               |                 v                 |
                                       |---------------------->+-------------------+       |
                                        (Signal Bundle)|       | Listen for Entry/ |       |
                                         Transmission  |       | Exit Authorization|       |
                                                       |       +-------------------+       |
                                                       |                 |                 |
                                                       |                 v                 |
                                                       |       +-------------------+       |
                                                       |       | Order Execution   |       |
                                                       |       +-------------------+       |
                                                       +-----------------|-----------------+
                                                                         |
                                                                         v
                                                           +---------------------------+
                                                           |    Action Log & PnL       |
                                                           |       Monitoring          |
                                                           +---------------------------+


### 5. Risk Management & Proprietary Session Guard
* **Session Guard:** Blocks entry signals during the first 90 minutes of the New York trading session to prevent losses caused by zero tail dependence violations, momentum jumps, and bid-ask spread noise at the open.
* **Position Sizing:** Continuous Kelly position sizing capped at a 70% maximum margin ceiling.
* **Circuit Breakers:** 10% equity drawdown halts new entries; 20% equity drawdown triggers hard liquidation.
* **Swap & Close Protection:** Halts entries 10 minutes prior to daily swap cutoff, force-closes positions 5 minutes prior, and resumes 3 minutes after.

---

## 📧 Contact

**Ratchatapong Lukmuang**  
M.Sc. Financial Engineering | Karlsruhe Institute of Technology (KIT)  
*Open to Quantitative Research, Quantitative Trading, and Quantitative Risk roles in Germany and internationally.*

* **LinkedIn Article:** [Inside My M.Sc. Thesis](https://www.linkedin.com/pulse/inside-my-msc-thesis-building-algo-trading-985-sharpe-lukmuang-mgfif/)
* **Contact Details:** [Available via LinkedIn profile](https://www.linkedin.com/in/ratchatapong-lukmuang/)
