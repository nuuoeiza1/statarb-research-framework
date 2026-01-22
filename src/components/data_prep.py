##################################################################
## NEED TO OPEN MT5 (PROGRAM) !! ##
## If error: MT5 initialize() failed: (-6, 'Terminal: Authorization failed') -> Create a new account in the broker.
##################################################################

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import os

def download_ticks(symbol: str, start_date: str, end_date: str, output_folder: str):
    """
    Downloads tick data from MetaTrader 5 for the given symbol
    and saves it as daily .parquet files inside a subfolder named after the symbol.

    Parameters:
        symbol (str): The trading symbol in MT5 (e.g., "US500.cash")
        start_date (str): Start date in format "YYYY-MM-DD"
        end_date (str): End date in format "YYYY-MM-DD"
        output_folder (str): Parent folder where symbol subfolder and .parquet files will be saved
    """

    # Initialize connection to MetaTrader 5
    if not mt5.initialize():
        print("❌ MT5 initialize() failed:", mt5.last_error())
        return
    
    # Print account info
    account_info = mt5.account_info()
    if account_info is not None:
        print("--- Current Account Details ---")
        print(f"Login ID: {account_info.login}")
        print(f"Trade Server: {account_info.server}")
        print(f"Name: {account_info.name}")
        print(f"Currency: {account_info.currency}")
        print(f"Balance: {account_info.balance}")
        print("-------------------------------")

    # Ensure the symbol is available for trading
    if not mt5.symbol_select(symbol, True):
        print(f"❌ Failed to select symbol: {symbol}")
        mt5.shutdown()
        return

    # Create output path: output_folder/symbol/
    symbol_folder = os.path.join(output_folder, symbol)
    os.makedirs(symbol_folder, exist_ok=True)

    # Convert string dates to datetime objects
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # Loop through each day in the range
    current = start
    while current <= end:
        day_str = current.strftime("%Y-%m-%d")
        file_path = os.path.join(symbol_folder, symbol+"_"+f"{day_str}.parquet")

        # Skip if the file for this date already exists
        if os.path.exists(file_path):
            print(f"⏩ Skipping {day_str} (already exists)")
            current += timedelta(days=1)
            continue

        # Define start and end of the day
        day_start = datetime(current.year, current.month, current.day)
        day_end = day_start + timedelta(days=1)

        print(f"📥 Downloading tick data for {symbol} | {day_str} ...")

        # Retrieve tick data from MT5
        ticks = mt5.copy_ticks_range(symbol, day_start, day_end, mt5.COPY_TICKS_ALL)
        df = pd.DataFrame(ticks)

        if df.empty:
            print(f"⚠️ No tick data available for {day_str}")
        else:
            # Convert Unix timestamp to datetime
            df['time'] = pd.to_datetime(df['time'], unit='s')

            # Save DataFrame to .parquet
            df.to_parquet(file_path, index=False)
            print(f"✅ Saved: {file_path} | {len(df)} rows")

        # Proceed to next day
        current += timedelta(days=1)

    # Shutdown MT5 connection
    mt5.shutdown()


"""
# Example usage
download_ticks(
    symbol="SPY.US",
    #symbol="US500",                            # Symbol must exactly match your MT5
    #symbol="US500-F",                          # Symbol must exactly match your MT5
    start_date="2021-01-01",
    end_date="2025-12-21",
    output_folder="data/pepperstone_tick-data"       # You can use an absolute or relative path
)
"""

##############################################################################################
import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from pathlib import Path
from typing import List, Optional
from pandas.tseries.holiday import USFederalHolidayCalendar

ProgressBar().register()

def preprocess_tick_file(file_path: Path, symbol: str) -> pd.DataFrame:
    """
    Step 1: Convert raw data into a standardized UTC format and select columns.
    """
    df = pd.read_parquet(file_path)
    # Convert Unix MS to UTC to maintain data integrity
    df["datetime_utc"] = pd.to_datetime(df["time_msc"], unit="ms", utc=True)
    df["symbol"] = symbol
    return df[["datetime_utc", "bid", "ask", "symbol"]]

def load_ticks_to_dask(symbol: str, path: Path, start_date: Optional[str] = None, end_date: Optional[str] = None) -> dd.DataFrame:
    """
    Step 2: Filter daily parquet files by date and load them into a Dask DataFrame.
    """
    target_path = Path(path) / symbol
    if not target_path.exists():
        raise FileNotFoundError(f"❌ Path not found: {target_path}")

    all_files = sorted(target_path.glob(f"{symbol}_*.parquet"))
    filtered_files = []
    for file in all_files:
        try:
            date_part = file.stem.split("_")[1]
            if (start_date is None or date_part >= start_date) and \
               (end_date is None or date_part <= end_date):
                filtered_files.append(file)
        except IndexError:
            continue

    if not filtered_files:
        print(f"⚠️ No files found for {symbol} in the given date range.")
        return None

    df_list = [preprocess_tick_file(f, symbol) for f in filtered_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    ddf = dd.from_pandas(combined_df, npartitions=48)
    return ddf

def clean_tick_data(ddf: dd.DataFrame, symbol: str) -> dd.DataFrame:
    """
    Step 3: Clean data and remove price outliers.
    Keep raw data without time filtering to see full broker behavior.
    """
    print(f"🧹 Cleaning {symbol} data...")
    ddf = ddf.dropna(subset=["bid", "ask"])
    ddf = ddf[(ddf["bid"] > 0) & (ddf["ask"] > 0) & (ddf["ask"] >= ddf["bid"])]
    ddf = ddf.sort_values("datetime_utc")
    ddf = ddf.drop_duplicates(subset="datetime_utc", keep="last")

    ddf["spread"] = ddf["ask"] - ddf["bid"]
    print(f"   > Calculating Spread Threshold for {symbol}:")
    with ProgressBar():
        # Using quantile to identify and remove price spikes
        threshold = ddf["spread"].quantile(0.99).compute() * 5
    
    ddf = ddf[ddf["spread"] < threshold]
    return ddf

def create_raw_baseline(ddf_list: List[dd.DataFrame], tickers: List[str], freq: str = '1s') -> pd.DataFrame:
    """
    NEW STEP: Create a 'Ruler' table. 
    > Reference Grid: Creates a precise time-aligned table using raw, unmodified tick data.
    > Gap Detection: Identifies where "Real Ticks" exist versus where "Data Gaps" occur before any filling.
    > Establish Baseline: Serves as a ground-truth benchmark to measure how much the data changes during processing.
    """
    print(f"📏 Creating Raw Baseline (The Ruler) at {freq}...")
    raw_resampled_list = []
    
    for ddf, ticker in zip(ddf_list, tickers):
        df = ddf.compute().set_index("datetime_utc")
        # Just resample the last known mid price per second, no filling
        mid = (df["bid"] + df["ask"]) / 2
        res = mid.resample(freq).last().to_frame(f"{ticker}_raw")
        raw_resampled_list.append(res)
    
    # Outer join keeps everything; gaps will be NaNs
    raw_baseline = pd.concat(raw_resampled_list, axis=1, join="outer")
    return raw_baseline

def calculate_data_fidelity(df_wide: pd.DataFrame, raw_baseline: pd.DataFrame, tickers: List[str]):
    """
    NEW STEP: Compare Final Table vs Raw Baseline.
    > Measure Freshness: Calculates the percentage of "Real Data" versus "Synthetic Data" (forward-filled) in the final table.
    > Quality Assessment: Determines if the model is learning from actual price movement or stagnant, filled prices.
    > Sync Validation: Uses Correlation checks to ensure multiple assets are correctly aligned in the time domain.
    """
    print("\n" + "="*60)
    print("📊 DATA FIDELITY & ACCURACY REPORT")
    print("="*60)
    
    # We only care about rows that exist in your final trading table
    # This automatically ignores market-closed periods where both are NaN
    comp = df_wide.join(raw_baseline, how='left')
    
    results = []
    for ticker in tickers:
        total_trading_seconds = len(comp)
        # Real data is where the raw baseline actually had a tick in that specific second
        real_data_mask = comp[f"{ticker}_raw"].notna()
        real_points = real_data_mask.sum()
        
        fidelity_pct = (real_points / total_trading_seconds) * 100
        
        results.append({
            "Symbol": ticker,
            "Total Secs": total_trading_seconds,
            "Real Ticks": real_points,
            "Fidelity %": f"{fidelity_pct:.2f}%"
        })
    
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))

    # Correlation based on new naming: SYMBOL_close
    col1, col2 = f"{tickers[0]}_close", f"{tickers[1]}_close"
    if col1 in df_wide.columns and col2 in df_wide.columns:
        correlation = df_wide[col1].corr(df_wide[col2])
        print("-" * 60)
        print(f"📈 Sync Correlation (Close): {correlation:.6f}")
    print("="*60 + "\n")

def resample_and_sync_only_dup(ddf_list: List[dd.DataFrame], tickers: List[str], freq: str = '1s') -> pd.DataFrame:
    """
    Step 4: Synchronize multiple assets into a Wide-Format DataFrame.
    Maintains UTC time to ensure data goes until 23:00.
    """
    # Create the Raw Baseline first for comparison later
    raw_baseline = create_raw_baseline(ddf_list, tickers, freq)

    print(f"🔗 Synchronizing assets to {freq} frequency...")
    resampled_dfs = []

    for ddf, ticker in zip(ddf_list, tickers):
        print(f"   > Processing {ticker}:")
        df = ddf.compute().set_index("datetime_utc")
        df["mid_price"] = (df["bid"] + df["ask"]) / 2
        
        # Resample Mid Price to OHLC
        res = df["mid_price"].resample(freq).ohlc().add_prefix(f"{ticker}_")
        
        # Core columns for Arbitrage
        res[f"{ticker}_bid"] = df["bid"].resample(freq).last()
        res[f"{ticker}_ask"] = df["ask"].resample(freq).last()
        res[f"{ticker}_spread"] = df["spread"].resample(freq).max()
        
        resampled_dfs.append(res)

    df_wide = pd.concat(resampled_dfs, axis=1, join="outer")
    df_wide = df_wide.dropna(subset=[f"{tickers[0]}_close", f"{tickers[1]}_close"])
    
    # RUN VALIDATION
    calculate_data_fidelity(df_wide, raw_baseline, tickers)
    
    return df_wide

def resample_and_sync(ddf_list: List[dd.DataFrame], tickers: List[str], freq: str = '1min') -> pd.DataFrame:
    """
    Step 4: Synchronize multiple assets into a Wide-Format DataFrame.
    Optimized for ML Prediction using Fixed UTC Schedule (16:31 - 22:59) 
    with dynamic Stock Market Holiday filtering.
    """
    # Create the Raw Baseline first for comparison later
    raw_baseline = create_raw_baseline(ddf_list, tickers, freq)

    print(f"🔗 Synchronizing assets to {freq} frequency...")
    resampled_dfs = []

    for ddf, ticker in zip(ddf_list, tickers):
        print(f"   > Processing {ticker}:")
        with ProgressBar():
            # Trigger Dask computation
            df = ddf.compute()
        
        df = df.set_index("datetime_utc")
        df.index = pd.to_datetime(df.index).tz_convert('UTC')
        df["mid_price"] = (df["bid"] + df["ask"]) / 2
        
        # Resample Mid Price to OHLC
        res = df["mid_price"].resample(freq).ohlc().add_prefix(f"{ticker}_")

        res[f"{ticker}_bid"] = df["bid"].resample(freq).last()
        res[f"{ticker}_ask"] = df["ask"].resample(freq).last()
        res[f"{ticker}_spread"] = df["spread"].resample(freq).max()
        
        resampled_dfs.append(res)

    # Outer Join preserves data even if one symbol is missing
    df_wide = pd.concat(resampled_dfs, axis=1, join="outer")

    # --- 1. DYNAMIC STOCK MARKET HOLIDAYS ---
    # Generate federal holidays for the range of data
    cal = USFederalHolidayCalendar()
    fed_holidays = cal.holidays(start=df_wide.index.min(), end=df_wide.index.max(), return_name=True)
    
    # Exclude holidays where the Stock Market is actually OPEN (Columbus Day & Veterans Day)
    market_open_holidays = ["Columbus Day", "Veterans Day", "Indigenous Peoples' Day"]
    stock_market_holidays = fed_holidays[~fed_holidays.isin(market_open_holidays)].index.normalize()

    # --- 2. TIME GRID & REINDEX (16:31 - 22:59) ---
    unique_dates = df_wide.index.normalize().unique()
    full_grid = []
    for d in unique_dates:
        # Skip days that are in our dynamic stock market holiday list
        if d in stock_market_holidays:
            continue
            
        full_grid.extend(pd.date_range(
            start=d + pd.Timedelta(hours=16, minutes=31), 
            end=d + pd.Timedelta(hours=22, minutes=59), 
            freq=freq, tz='UTC'
        ))
    
    df_wide = df_wide.reindex(full_grid)

    # --- 3. APPLY FFILL & CLEAN ---
    # Forward fill limit set to 5 minutes for continuity
    df_wide = df_wide.ffill(limit=5)

    # --- 4. Drop rows where either leg is missing (Strict sync for Arbitrage/Prediction) ---
    # df_wide = df_wide.dropna(subset=[f"{tickers[0]}_mid", f"{tickers[1]}_mid"])
    df_wide = df_wide.dropna(subset=[f"{tickers[0]}_close", f"{tickers[1]}_close"])
    
    # RUN VALIDATION
    calculate_data_fidelity(df_wide, raw_baseline, tickers)
    
    return df_wide