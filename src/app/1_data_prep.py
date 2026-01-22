##############################################################################################
## (START) >> DATA PREP
# ** Reqiure src/components/data_prep.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from components.data_prep import *

def run_data_preparation(SYMBOLS, START_DATE, END_DATE, OUTPUT_FOLDER, OUTPUT_FILENAME):
    ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    OUTPUT_FOLDER = os.path.join(ROOT_PATH, "data", OUTPUT_FOLDER)

    ## > Load_tick data
    for symbol in SYMBOLS:
        download_ticks(symbol=symbol, start_date=START_DATE, end_date=END_DATE, output_folder=OUTPUT_FOLDER)

    ## > Load data to dask dataframe
    ddf_spy = load_ticks_to_dask(SYMBOLS[0], path=OUTPUT_FOLDER, start_date=START_DATE, end_date=END_DATE)
    ddf_us500 = load_ticks_to_dask(SYMBOLS[1], path=OUTPUT_FOLDER, start_date=START_DATE, end_date=END_DATE)

    ## > Clean data
    ddf_spy = clean_tick_data(ddf_spy, SYMBOLS[0])
    ddf_us500 = clean_tick_data(ddf_us500, SYMBOLS[1])

    ## > Resample (1min) + Data Fidelity
    # The 'Fidelity %' printed in the console tells you how much of this data is fresh vs ffilled.
    final_df_only_dup = resample_and_sync_only_dup([ddf_spy, ddf_us500], SYMBOLS, freq='1min')
    final_df = resample_and_sync([ddf_spy, ddf_us500], SYMBOLS, freq='1min')

    # Display the final synchronized table
    print("\n--- Final Synchronized DataFrame (UTC) ---")
    print(final_df.head())
    print(final_df.tail())

    # Check for rows until 23:00 UTC as expected
    print(f"\nData covers from {final_df.index.min()} to {final_df.index.max()}")

    print(final_df)

    #del START_DATE, END_DATE, OUTPUT_FOLDER

    ########################################################################
    # 1. Validate and prepare Index (Ensure datetime format)
    final_df_only_dup.index = pd.to_datetime(final_df_only_dup.index)
    final_df.index = pd.to_datetime(final_df.index)

    # 2. Identify Timestamps present in only-dup but missing from spare
    # (Explains why only-dup may have more rows)
    only_in_final_dup = final_df_only_dup.index.difference(final_df.index)

    # 3. Identify Timestamps present in spare but missing from only-rep
    # (Gaps filled using ffill where original source lacked price data)
    only_in_final_real = final_df.index.difference(final_df_only_dup.index)

    # 4. Check for duplicates in only-dup (can cause abnormal row counts)
    duplicates_in_final_dup = final_df_only_dup.index.duplicated().sum()

    # --- Summary Results ---
    print(f"--- Total rows ---")
    print(f"Rows in final_df_only_dup (Original Records):          {len(final_df_only_dup):,}")
    print(f"Rows in final_df (Final Records):                      {len(final_df):,}")
    print(f"Different rows:                                        {len(final_df_only_dup) - len(final_df):,}")

    print(f"\n--- Analysis (Diff) ---")
    print(f"1. Timestamps unique to only-dup (Excess)(Data loss):  {len(only_in_final_dup):,}")
    print(f"2. Timestamps unique to spare (ffill additions):       {len(only_in_final_real):,}")
    print(f"3. Count of duplicated Timestamps:                     {duplicates_in_final_dup:,}")

    if len(only_in_final_dup) > 0:
        print("\n--- Sample: First 10 timestamps unique to only-rep ---")
        print(only_in_final_dup[:10])

    ## (END) >> DATA PREP
    ##############################################################################################
    print("")

    # Only_dup (recheck only)
    final_df_only_dup.to_csv(os.path.join(ROOT_PATH, "data", f"{OUTPUT_FILENAME}_only_dup.csv"), index=True)
    print(f"🔍 Saved Inspection Data: \"../../data/{OUTPUT_FILENAME}_only_dup.csv\"")
    
    final_df.to_csv(os.path.join(ROOT_PATH, "data", f"{OUTPUT_FILENAME}.csv"), index=True)
    print(f"🔍 Saved Inspection Data: \"../../data/{OUTPUT_FILENAME}.csv\"")

    final_df.to_parquet(os.path.join(ROOT_PATH, "data", f"{OUTPUT_FILENAME}.parquet"), compression="snappy")
    print(f"✅ Saved Master Data: \"../../data/{OUTPUT_FILENAME}.parquet\"")
    print("File has been saved successfully!")


## Broker Data Characteristics (Demo account)
# > On 2023-04-03: Data from 11:46 to 23:58 (First day!)
# > On 2023-04-04: Data from 00:01 to 23:58
# > On 2023-04-05: Data from 00:01 to 02:59, 11:46 to 23:58
# > On 2023-04-06: Data from 00:01 to 23:58
# > From 2023-04-10 to 2023-10-18: Data from 16:46 to 22:54-22:58 => ROBUSTNESS CHECK!! 
# > From 2023-10-19 till Now (2025-12-30): Data from 16:31 to 22:58 => MAIN DATA

## Broker Data Characteristics (Real account)
# > On 2021-10-01: Data from 00:01 to 23:54 (First day!)
# > On 2021-10-04: Data from 11:30 to 23:58
# > On 2021-10-05: Data from 00:01 to 23:58
# > From 2021-10-06 to 2021-10-08: Data from 00:01 to 02:59, 11:30 to 23:58
# > On 2021-10-11: Data from 11:30 to 23:58
# > On 2021-10-12: Data from 00:01 to 02:59, 11:30 to 23:58
# > ... Strange data till 2021-10-29
# > NO DATA IN 2022-2024!! 
# > From 2025-01-02 till Now (2025-12-30): Data from 16:31 to 22:54-22:58

## >> Real Account <<
"""
SYMBOLS = ["SPY.US.a", "US500.a"] 
OUTPUT_FOLDER = "pepperstone_tick-data_real"
"""

## >> Demo Account << 
SYMBOLS = ["SPY.US", "US500"]
OUTPUT_FOLDER = "pepperstone_tick-data"

START_DATE = "2021-01-01" 
#START_DATE = "2025-01-01"
END_DATE = "2025-12-31"
OUTPUT_FILENAME = "Final_data"

run_data_preparation(SYMBOLS, START_DATE, END_DATE, OUTPUT_FOLDER, OUTPUT_FILENAME)
