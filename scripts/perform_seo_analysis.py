import pandas as pd
import os

def perform_seo_analysis():
    """
    Integrates multiple data sources to create a master analysis dataset.
    1. Loads N-side data, post metrics, query-level inflow, and keyword search volumes.
    2. Calculates SEO efficiency (C = Inflow / Total Search Volume) for each query per post per month.
    3. Aggregates metrics per post (e.g., average SEO efficiency).
    4. Merges all data into a single master CSV file for further analysis.
    """
    # --- 1. Define Paths ---
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    nside_path = os.path.join(base_dir, 'data', 'data_processed', 'post_analysis_results.csv')
    metrics_path = os.path.join(base_dir, 'data', 'data_input', 'post-metrics.csv')
    query_inflow_path = os.path.join(base_dir, 'data', 'data_input', 'post-searchQuery.csv')
    search_volume_path = os.path.join(base_dir, 'data', 'data_processed', 'keyword_search_volume_transform.csv')
    output_path = os.path.join(base_dir, 'data', 'data_processed', 'master_analysis_dataset.csv')
    
    # --- 2. Load DataFrames ---
    try:
        df_nside = pd.read_csv(nside_path)
        df_metrics = pd.read_csv(metrics_path)
        df_query_inflow = pd.read_csv(query_inflow_path)
        df_search_volume = pd.read_csv(search_volume_path)
        print("All source files loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: Could not find a source file. {e}")
        return

    # --- 3. Data Preparation & SEO Efficiency (C) Calculation ---
    
    # Prepare date columns for merging
    df_query_inflow['date'] = pd.to_datetime(df_query_inflow['startDate']).dt.strftime('%Y-%m')
    
    # Calculate total search volume (pc + mobile)
    df_search_volume['total_volume'] = df_search_volume['pc'] + df_search_volume['mobile']
    
    # Merge query inflow data with search volume data
    df_merged_query = pd.merge(
        df_query_inflow,
        df_search_volume[['searchQuery', 'date', 'total_volume']],
        on=['searchQuery', 'date'],
        how='left'
    )
    
    # Calculate SEO efficiency 'C'
    # Avoid division by zero, fill resulting NaN/inf with 0
    df_merged_query['seo_efficiency_C'] = (df_merged_query['searchInflow'] / df_merged_query['total_volume']).fillna(0)
    
    # Replace infinite values (if any) with 0
    df_merged_query.replace([float('inf'), float('-inf')], 0, inplace=True)

    # --- 4. Aggregate by PostID ---
    
    # Calculate average 'C' for each post
    df_avg_c = df_merged_query.groupby('postId')['seo_efficiency_C'].mean().reset_index()
    df_avg_c.rename(columns={'seo_efficiency_C': 'average_C'}, inplace=True)
    
    # --- 5. Build Master Analysis Dataset ---
    
    # Merge N-side data with post metrics
    df_master = pd.merge(df_nside, df_metrics, on='postId', how='left')
    
    # Merge with the new average_C metric
    df_master = pd.merge(df_master, df_avg_c, on='postId', how='left')
    
    # Fill NaN values in the new metrics columns with 0 for consistency
    df_master['average_C'].fillna(0, inplace=True)
    
    # --- 6. Save the Master Dataset ---
    df_master.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Master analysis dataset created successfully at: {output_path}")

    return df_master

if __name__ == "__main__":
    master_df = perform_seo_analysis()
    if master_df is not None:
        print("\n--- Master Dataset Preview ---")
        print(master_df.head())
        print("\n--- Master Dataset Info ---")
        master_df.info() 