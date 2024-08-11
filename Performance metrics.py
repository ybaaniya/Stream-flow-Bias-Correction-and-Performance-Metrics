import os
import numpy as np
import pandas as pd
import hydrostats as hs
import hydrostats.data as hd
import HydroErr as he
from joblib import Parallel, delayed

# Load the main CSV file
main_df = pd.read_csv('path to master csv file that has corresponding simulated and obserevd id')

def parse_dates(date_series):
    dt = pd.to_datetime(date_series, format="%Y-%m-%d", errors='coerce')
    dt = dt.fillna(pd.to_datetime(date_series, format="%m/%d/%y", errors='coerce'))
    dt = dt.apply(lambda x: x if pd.isnull(x) or x.year <= 2025 else x - pd.DateOffset(years=100))
    return dt

def process_row(row, missing_files):
    matching_column = row['matching_column']   # change the column that represent the observed gauged column in master csv
    validation = row['linkno']    # change the column that represent either GEOGLOWS V1 or V2 column  in master csv

    matching_file_path = f'path to observed data/{matching_column}.csv'  # path to folder where you have observed gauge
    validation_file_path = f'path to simulated data/{validation}.csv'  # path to individual simulated value

    # Check if both files exist
    if not os.path.exists(matching_file_path):
        missing_files.append(matching_file_path)
        return None
    if not os.path.exists(validation_file_path):
        missing_files.append(validation_file_path)
        return None

    # Load the corresponding CSV files using pandas and rename columns
    df_o = pd.read_csv(matching_file_path, dtype={'Date': 'str'}, names=['Date', 'Observed'], skiprows=1)
    df_s = pd.read_csv(validation_file_path, dtype={'Date': 'str'}, names=['Date', 'Simulated'], skiprows=1)

    # Parse dates
    df_o['Date'] = parse_dates(df_o['Date'])
    df_s['Date'] = parse_dates(df_s['Date'])

    # Merge data and set the parsed date as the index
    merged_df = pd.merge(df_o, df_s, on='Date', how='inner').set_index('Date')

    # Filter data by date range
    merged_df = merged_df['1940-01-01':'2024-08-08']

    # Convert columns to numeric
    merged_df = merged_df.apply(pd.to_numeric, errors='coerce')
    #merged_df.to_csv(f'/path to save the simulated and observed timeseries for the matching date /{matching_column}.csv')
    # Print statement to indicate which merged_df is being saved
    print(f"Processed and merged data for {matching_column}")

    # Filter out negative values
    merged_df_filtered = merged_df[(merged_df['Simulated'] >= 0) & (merged_df['Observed'] >= 0)]

    # Calculate metrics
    metrics = {
        'me': he.me(merged_df['Simulated'], merged_df['Observed'], remove_neg=True),
        'rmse': he.rmse(merged_df['Simulated'], merged_df['Observed'], remove_neg=True),
        'mae': he.mae(merged_df['Simulated'], merged_df['Observed'], remove_neg=True),
        'nse': he.nse(merged_df['Simulated'], merged_df['Observed'], remove_neg=True),
        'pearson_r': he.pearson_r(merged_df['Simulated'], merged_df['Observed'], remove_neg=True),
        'r_squared': he.r_squared(merged_df['Simulated'], merged_df['Observed'], remove_neg=True),
        'kge_2012': he.kge_2012(merged_df['Simulated'], merged_df['Observed'], remove_neg=True),
        'nrmse_mean': he.nrmse_mean(merged_df['Simulated'], merged_df['Observed'], remove_neg=True),
        'std_obs': merged_df_filtered['Observed'].std(),
        'std_sim': merged_df_filtered['Simulated'].std(),
        'mean_obs': merged_df_filtered['Observed'].mean(),
        'mean_sim': merged_df_filtered['Simulated'].mean(),
        'name': f"{matching_column}"
    }

    return metrics

if __name__ == "__main__":
    num_jobs = os.cpu_count()  # Number of cores to use

    missing_files = []

    # Use joblib for parallel processing
    results = Parallel(n_jobs=num_jobs)(
        delayed(process_row)(row, missing_files) for index, row in main_df.iterrows()
    )

    # Filter out None results
    filtered_results = [result for result in results if result is not None]

    # Convert to DataFrame
    results_df = pd.DataFrame(filtered_results)

    # Save the DataFrame to a CSV file
    results_df.to_csv('path to save the performance metrics', index=False)

    # Save the missing files report
    with open('path to save the report of missing or skipped rows.txt', 'w') as f:
        for file in missing_files:
            f.write(f"{file}\n")

    print("Results saved to metrics.csv")
    print("Missing files report saved to missing_files.txt")
