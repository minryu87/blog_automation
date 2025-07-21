import pandas as pd
import os
import numpy as np

def calculate_feature_differences():
    """
    Calculates the difference between each of our posts and the average of its
    top competitors within the same topic cluster. This creates a set of
    'difference features' for modeling.
    """
    # --- 1. Define File Paths ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    blog_automation_dir = os.path.join(script_dir, '..')
    
    master_data_path = os.path.join(blog_automation_dir, 'data', 'data_processed', 'master_post_data.csv')
    clusters_path = os.path.join(blog_automation_dir, 'data', 'data_processed', 'topic_clusters.csv')
    output_dir = os.path.join(blog_automation_dir, 'data', 'data_processed')
    output_path = os.path.join(output_dir, 'feature_differences.csv')

    os.makedirs(output_dir, exist_ok=True)

    # --- 2. Load Datasets ---
    print("Loading master data and topic clusters...")
    df_master = pd.read_csv(master_data_path)
    df_clusters = pd.read_csv(clusters_path)

    # --- 3. Ensure Consistent Data Types for Merging ---
    df_master['post_identifier'] = df_master['post_identifier'].astype(str)
    df_clusters['our_postId'] = df_clusters['our_postId'].astype(str)
    df_clusters['competitor_post_url'] = df_clusters['competitor_post_url'].astype(str)

    # --- 4. Prepare DataFrames ---
    df_ours = df_master[df_master['source'] == 'ours'].copy()
    df_competitors = df_master[df_master['source'] == 'competitor'].copy()

    # Identify numeric columns for comparison
    numeric_cols = df_ours.select_dtypes(include=np.number).columns.tolist()
    # Remove identifiers or irrelevant columns
    cols_to_exclude = ['our_postId', 'non_brand_inflow', 'non_brand_average_ctr', 'total_views', 'average']
    numeric_cols_to_compare = [col for col in numeric_cols if col not in cols_to_exclude]
    
    print(f"Identified {len(numeric_cols_to_compare)} numeric features for comparison.")

    # --- 5. Calculate Competitor Averages per Topic ---
    print("Calculating competitor feature averages for each topic...")
    # Map competitor URLs to their topics (representative_query)
    competitor_topics = df_clusters[['competitor_post_url', 'representative_query']].drop_duplicates()
    df_competitors_with_topic = pd.merge(
        df_competitors,
        competitor_topics,
        left_on='post_identifier',
        right_on='competitor_post_url',
        how='inner'
    )
    
    # Calculate the mean of numeric features for each topic
    competitor_topic_averages = df_competitors_with_topic.groupby('representative_query')[numeric_cols_to_compare].mean().reset_index()
    # Rename columns to avoid clashes
    competitor_topic_averages.columns = ['representative_query'] + [f"{col}_competitor_avg" for col in numeric_cols_to_compare]

    print("\n[DEBUG] Competitor topic averages sample:")
    print(competitor_topic_averages.head())
    print(f"[DEBUG] Found averages for {len(competitor_topic_averages)} topics.")


    # --- 6. Map Our Posts to Topics and Competitor Averages ---
    print("\nMapping our posts to competitor averages...")
    # Get the topic for each of our posts
    our_post_topics = df_clusters[['our_postId', 'representative_query']].drop_duplicates()
    
    # Merge our posts with their topics
    df_ours_with_topic = pd.merge(
        df_ours,
        our_post_topics,
        left_on='post_identifier',
        right_on='our_postId',
        how='inner'
    )
    
    print(f"\n[DEBUG] Our posts after merging with topics: {len(df_ours_with_topic)} rows")
    print(df_ours_with_topic[['post_identifier', 'representative_query']].head())

    # Merge our posts with the competitor averages for their topic
    df_analysis = pd.merge(
        df_ours_with_topic,
        competitor_topic_averages,
        on='representative_query',
        how='left'
    )

    print(f"\n[DEBUG] Our posts after merging with competitor averages: {len(df_analysis)} rows")
    print(df_analysis[['post_identifier', 'representative_query', f"{numeric_cols_to_compare[0]}_competitor_avg"]].head())

    # Drop topics where no competitor average could be computed
    df_analysis.dropna(subset=[f"{numeric_cols_to_compare[0]}_competitor_avg"], inplace=True)
    
    print(f"\n[DEBUG] Final analysis rows after dropping NA averages: {len(df_analysis)} rows")


    # --- 7. Calculate Difference Features ---
    print("\nCalculating difference features (our_value - competitor_avg)...")
    for col in numeric_cols_to_compare:
        our_col = col
        competitor_avg_col = f"{col}_competitor_avg"
        diff_col = f"{col}_diff"
        # Ensure columns are numeric before subtraction
        df_analysis[our_col] = pd.to_numeric(df_analysis[our_col], errors='coerce')
        df_analysis[competitor_avg_col] = pd.to_numeric(df_analysis[competitor_avg_col], errors='coerce')
        df_analysis[diff_col] = df_analysis[our_col] - df_analysis[competitor_avg_col]

    # --- 8. Create Final DataFrame ---
    final_columns = ['post_identifier', 'representative_query', 'non_brand_inflow', 'non_brand_average_ctr'] + [f"{col}_diff" for col in numeric_cols_to_compare]
    df_final = df_analysis[final_columns].copy()

    df_final.to_csv(output_path, index=False, encoding='utf-8-sig')

    print("\n--- Success! ---")
    print(f"Calculated feature differences for {len(df_final)} posts.")
    print(f"Final analysis-ready data saved to: {output_path}")
    print("\n--- Sample Output ---")
    print(df_final.head())
    print("---------------------\n")

if __name__ == '__main__':
    calculate_feature_differences() 