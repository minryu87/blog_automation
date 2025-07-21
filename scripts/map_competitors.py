import pandas as pd
import os
import re

def map_competitors_by_topic():
    """
    Maps our blog posts to competitor posts based on representative search queries.
    This creates 'topic clusters' for competitive analysis.
    """
    # --- 1. Define File Paths ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    blog_automation_dir = os.path.join(script_dir, '..')
    
    rep_queries_path = os.path.join(blog_automation_dir, 'data', 'data_processed', 'representative_queries.csv')
    popular_contents_path = os.path.join(blog_automation_dir, 'data', 'data_input', 'popular-contents.csv')
    output_dir = os.path.join(blog_automation_dir, 'data', 'data_processed')
    output_path = os.path.join(output_dir, 'topic_clusters.csv')

    os.makedirs(output_dir, exist_ok=True)

    # --- 2. Load Datasets ---
    print("Loading datasets...")
    df_ours = pd.read_csv(rep_queries_path)
    df_popular = pd.read_csv(popular_contents_path)

    print(f"Loaded {len(df_ours)} of our posts with representative queries.")
    print(f"Loaded {len(df_popular)} popular content entries.")

    # --- 3. Identify Competitor Posts ---
    # A competitor post is one where 'myContent' is NaN (or not 'checked').
    print("Identifying competitor posts...")
    df_competitors_raw = df_popular[df_popular['myContent'].isnull()].copy()
    
    # Filter by rank to focus on top performers (rank <= 3)
    max_rank = 3
    print(f"Filtering competitors to include only top {max_rank} ranks...")
    df_competitors = df_competitors_raw[df_competitors_raw['rank'] <= max_rank].copy()
    
    # Clean up competitor data
    df_competitors = df_competitors.dropna(subset=['contentId'])
    df_competitors = df_competitors[df_competitors['contentId'].str.contains('blog.naver.com', na=False)]
    # Rename for clarity and future merging
    df_competitors.rename(columns={'contentId': 'competitor_post_url'}, inplace=True)

    print(f"Found {len(df_competitors)} competitor posts.")

    # --- 4. Merge Our Posts with Competitors on Query ---
    # We use a left merge to keep all of our posts, even if no competitors were found.
    print("Merging our posts with competitors based on search query...")
    # Rename columns to avoid confusion after merge
    df_ours.rename(columns={'postId': 'our_postId'}, inplace=True)
    df_competitors.rename(columns={'searchQuery': 'representative_query'}, inplace=True)

    # Merge the dataframes
    df_merged = pd.merge(
        df_ours,
        df_competitors[['representative_query', 'competitor_post_url', 'rank']],
        on='representative_query',
        how='left'
    )
    
    # Drop rows where no competitor was found
    df_final = df_merged.dropna(subset=['competitor_post_url'])

    # --- 5. Clean and Save Results ---
    # Select and reorder columns for clarity
    final_columns = [
        'our_postId', 
        'representative_query', 
        'competitor_post_url',
        'rank'
    ]
    df_final = df_final[final_columns]
    
    df_final.to_csv(output_path, index=False, encoding='utf-8-sig')

    print("\n--- Success! ---")
    print(f"Created {len(df_final)} mappings between our posts and competitor posts.")
    print(f"Topic cluster data saved to: {output_path}")
    print("\n--- Sample Output ---")
    print(df_final.head())
    print("---------------------\n")


if __name__ == '__main__':
    map_competitors_by_topic() 