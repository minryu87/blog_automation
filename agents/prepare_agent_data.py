import pandas as pd
import numpy as np
import os

def prepare_data_for_agents():
    """
    Prepares and cleans the master dataset for the Agno agent system.
    - Filters out brand keywords to create non-brand performance metrics.
    - Creates a stable, cleaned dataset for agents to work with.
    """
    print("--- Preparing data for Agno agents ---")
    
    # --- 1. Setup Paths ---
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    nside_path = os.path.join(base_dir, 'data', 'data_processed', 'post_analysis_results.csv')
    query_inflow_path = os.path.join(base_dir, 'data', 'data_input', 'post-searchQuery.csv')
    search_volume_path = os.path.join(base_dir, 'data', 'data_processed', 'keyword_search_volume_transform.csv')
    output_path = os.path.join(base_dir, 'agents', 'agent_base_dataset.csv')

    # --- 2. Load Data ---
    try:
        df_nside = pd.read_csv(nside_path)
        df_query_inflow = pd.read_csv(query_inflow_path)
        df_search_volume = pd.read_csv(search_volume_path)
    except FileNotFoundError as e:
        print(f"Error: A source file was not found. {e}")
        return

    # --- 3. Create Non-Brand Performance Metrics ---
    brand_keyword = '내이튼'
    df_non_brand = df_query_inflow[~df_query_inflow['searchQuery'].str.contains(brand_keyword, na=False)].copy()
    
    # Target 1: Non-brand inflow
    df_target_inflow = df_non_brand.groupby('postId')['searchInflow'].sum().reset_index()
    df_target_inflow.rename(columns={'searchInflow': 'non_brand_inflow'}, inplace=True)
    
    # Target 2: Non-brand average C
    df_non_brand['date'] = pd.to_datetime(df_non_brand['startDate']).dt.strftime('%Y-%m')
    df_search_volume['total_volume'] = df_search_volume['pc'] + df_search_volume['mobile']
    df_detailed = pd.merge(df_non_brand, df_search_volume, on=['searchQuery', 'date'], how='left')
    df_detailed['C'] = (df_detailed['searchInflow'] / df_detailed['total_volume']).fillna(0)
    df_detailed.replace([np.inf, -np.inf], 0, inplace=True)
    
    df_target_c = df_detailed.groupby('postId')['C'].mean().reset_index()
    df_target_c.rename(columns={'C': 'non_brand_average_C'}, inplace=True)

    # --- 4. Create and Clean the Base Dataset ---
    df_base = pd.merge(df_nside, df_target_inflow, on='postId', how='inner')
    df_base = pd.merge(df_base, df_target_c, on='postId', how='left').fillna(0)

    # Convert posting_score to a numeric feature
    def score_to_numeric(score):
        score = str(score)
        if '준최' in score: return int(score.split('-')[0].replace('준최', ''))
        if '저품질' in score: return -1 * int(score.replace('저품질', ''))
        return 0
    df_base['posting_score_numeric'] = df_base['posting_score'].apply(score_to_numeric)
    
    # Select and rename columns for clarity
    features_to_keep = [
        'postId', 'post_title', 'posting_score_numeric', 'exposure_score', 'score_가독성', 
        'score_주제', 'score_콘텐츠', 'score_키워드', 'score_형태소', 'score_후기/정보',
        'content_with_space', 'word_count', 'valid_images', 'invalid_images',
        'non_brand_inflow', 'non_brand_average_C'
    ]
    df_base = df_base[features_to_keep]
    df_base.fillna(0, inplace=True)

    # --- 5. Save the Dataset ---
    df_base.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Agent base dataset saved to: {output_path}")
    print("Base dataset preview:")
    print(df_base.head())

if __name__ == "__main__":
    prepare_data_for_agents() 