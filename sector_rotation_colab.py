# sector_rotation_colab.py

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from config import CONFIG, sector_etfs, benchmark_etf
from holdings import fetch_top_holdings
from analytics import compute_statistics, plot_equity_lines
from utils import display_current_sector_selection

# === Imposta periodo di analisi ===
start_date = "2010-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

print(f"Configurazione: {CONFIG}")
print(f"Periodo: {start_date} - {end_date}")

# === Scarica dati ===
tickers = sorted(list(set(sector_etfs + [benchmark_etf])))
data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close'].ffill().dropna()

# === Calcolo indicatori ===
lookback = CONFIG['lookback_days']
daily_returns = data.pct_change()
momentum = data[sector_etfs].pct_change(lookback).shift(1)
volatility = daily_returns[sector_etfs].rolling(lookback).std().shift(1)
ma200 = data[sector_etfs].rolling(200).mean().shift(1)

monthly_prices = data.resample(CONFIG['rebalance_frequency']).last()
monthly_returns = monthly_prices.pct_change()
momentum_monthly = momentum.resample(CONFIG['rebalance_frequency']).last()
benchmark_returns = data[benchmark_etf].resample(CONFIG['rebalance_frequency']).last().pct_change()

common_index = momentum_monthly.index.intersection(monthly_returns.index)
portfolio_returns = []
dates_used = []

for date in common_index:
    top = momentum_monthly.loc[date].dropna().nlargest(CONFIG['top_n_sectors']).index.tolist()
    if CONFIG['use_ma200_filter']:
        top = [t for t in top if monthly_prices.loc[date, t] > ma200.loc[date, t]]
    if CONFIG['use_volatility_filter']:
        vol_slice = volatility.loc[date, top].dropna()
        top = vol_slice.nsmallest(CONFIG['top_n_sectors']).index.tolist()

    ret = monthly_returns.loc[date, top].mean() if top else 0.0
    if ret < CONFIG['stop_loss']:
        ret = CONFIG['stop_loss']
    portfolio_returns.append(ret)
    dates_used.append(date)

portfolio_returns_ts = pd.Series(portfolio_returns, index=pd.to_datetime(dates_used))
benchmark_returns_ts = benchmark_returns.reindex(portfolio_returns_ts.index).fillna(0)

# === Analisi e output ===
equity_strategy, equity_benchmark = compute_statistics(portfolio_returns_ts, benchmark_returns_ts)
plot_equity_lines(equity_strategy, equity_benchmark, benchmark_etf)
display_current_sector_selection(momentum_monthly, ma200, volatility, monthly_prices)

# === Download automatico Top Holdings ===
print("\n--- Estrazione titoli per i settori selezionati ---")
for sector in CONFIG['selected_sectors']:
    top_holdings = fetch_top_holdings(sector)
    print(f"\nTop 10 holdings per {sector}:")
    for rank, row in top_holdings.iterrows():
        print(f"{rank+1}. {row['symbol']} - {row['name']} ({row['weight']}%)")

print("\nEsecuzione completata.")
