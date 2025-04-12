import pandas as pd
import numpy as np
import sys
from itertools import combinations

def read_csv_files(*file_names):
    """Read CSV file and return a dictionary of DataFrames."""
    data = {}
    for file_name in file_names:
        try:
            df = pd.read_csv(file_name)
            if 'id' not in df.columns or 'prediction' not in df.columns:
                raise ValueError(f"File {file_name} must have 'id' and 'prediction' columns")
            data[file_name] = df.sort_values('id')
        except Exception as e:
            print(f"An error occurs while reading file {file_name}: {e}")
            sys.exit(1)
    return data

def calculate_error_metrics(df1, df2, file1_name, file2_name):
    """Calculate error metrics: mean error, variance, and standard deviation."""
    # Check whether the IDs match
    if not df1['id'].equals(df2['id']):
        print(f"Các ID trong {file1_name} và {file2_name} không khớp")
        return None, None, None
    
    # Calculate error metrics
    errors = np.abs(df1['prediction'] - df2['prediction'])
    mean_error = errors.mean()
    
    return mean_error

def calculate_large_diff_percentage(df1, df2, file1_name, file2_name, threshold=120):
    """Calculate the percentage of samples with large differences."""
    if not df1['id'].equals(df2['id']):
        print(f"The IDs in {file1_name} and {file2_name} do not match")
        return None
    
    errors = np.abs(df1['prediction'] - df2['prediction'])
    large_diff_count = (errors > threshold).sum()
    total_count = len(errors)
    percentage = (large_diff_count / total_count) * 100 if total_count > 0 else 0
    
    return percentage

def compare_files(data):
    """Comparing all pairs of files in data."""
    results = []
    file_pairs = list(combinations(data.keys(), 2))
    
    for file1_name, file2_name in file_pairs:
        df1 = data[file1_name]
        df2 = data[file2_name]
        
        # Calculate error metrics
        mean_error = calculate_error_metrics(df1, df2, file1_name, file2_name)
        
        # Calculate percentage of large differences
        percentage_120 = calculate_large_diff_percentage(df1, df2, file1_name, file2_name)
        percentage_180 = calculate_large_diff_percentage(df1, df2, file1_name, file2_name, threshold=180)
        
        if mean_error is not None:
            results.append({
                'file_pair': f"{file1_name} vs {file2_name}",
                'mean_error': mean_error,
                'percentage_120': percentage_120,
                'percentage_180': percentage_180
            })
    
    return results

def print_results(results):
    """Print the results of the comparison."""
    for result in results:
        print(f"\nComparison: {result['file_pair']}")
        print(f"Average mean error: {result['mean_error']:.4f}")
        print(f"The percentage of examples that greater than 2 minutes: {result['percentage_120']:.2f}%")
        print(f"The percentage of examples that greater than 3 minutes: {result['percentage_120']:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Please give at least 2 CSV files")
        print("How to use: python script.py file1.csv file2.csv [file3.csv ...]")
        sys.exit(1)
    
    file_names = sys.argv[1:]
    data = read_csv_files(*file_names)
    results = compare_files(data)
    print_results(results)