import pandas as pd
import os

def analyze_empty_competitor_columns():
    """
    Analyzes the master_post_data.csv file to find which columns are
    consistently empty for competitor entries.
    """
    # Define file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_processed_dir = os.path.join(script_dir, '..', 'data', 'data_processed')
    master_data_path = os.path.join(data_processed_dir, 'master_post_data.csv')

    if not os.path.exists(master_data_path):
        print(f"Error: Master data file not found at {master_data_path}")
        return

    # Load data
    df_master = pd.read_csv(master_data_path)

    # Filter for competitor data
    df_competitors = df_master[df_master['source'] == 'competitor'].copy()

    if df_competitors.empty:
        print("No competitor data found in the master file.")
        return

    # Find columns that are entirely null for competitors
    empty_columns = [col for col in df_competitors.columns if df_competitors[col].isnull().all()]

    print("Columns that are consistently empty for 'competitor' data:")
    if not empty_columns:
        print(" - None")
    else:
        for col in empty_columns:
            print(f" - {col}")

if __name__ == '__main__':
    analyze_empty_competitor_columns() 