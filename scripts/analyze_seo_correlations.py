import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def analyze_seo_correlations():
    """
    Analyzes the master dataset to validate hypotheses about SEO effectiveness.
    1. Identifies top-performing keywords based on average SEO efficiency.
    2. For each top keyword, analyzes the correlation between post features (N-side data)
       and post-performance (search inflow).
    3. Visualizes these correlations as heatmaps and saves them.
    """
    # --- 1. Setup ---
    # Set up paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    master_dataset_path = os.path.join(base_dir, 'data', 'data_processed', 'master_analysis_dataset.csv')
    query_inflow_path = os.path.join(base_dir, 'data', 'data_input', 'post-searchQuery.csv')
    search_volume_path = os.path.join(base_dir, 'data', 'data_processed', 'keyword_search_volume_transform.csv')
    output_dir = os.path.join(base_dir, 'data', 'data_analysis')
    os.makedirs(output_dir, exist_ok=True)

    # Set up Korean font for matplotlib
    try:
        if os.name == 'posix': # For macOS
            font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
            if os.path.exists(font_path):
                plt.rc('font', family='AppleGothic')
            else:
                 print("AppleGothic font not found. Please set a path to a Korean font.")
        elif os.name == 'nt': # For Windows
            font_name = fm.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
            plt.rc('font', family=font_name)
        plt.rcParams['axes.unicode_minus'] = False # Fix for minus sign not showing
        print("Font setup for plots complete.")
    except Exception as e:
        print(f"Could not set up Korean font, plots may not render text correctly. Error: {e}")


    # --- 2. Load Data ---
    try:
        df_master = pd.read_csv(master_dataset_path)
        df_query_inflow = pd.read_csv(query_inflow_path)
        df_search_volume = pd.read_csv(search_volume_path)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # --- 3. (Hypothesis 1) Identify High-Competitiveness Keywords ---
    
    # Recalculate SEO efficiency C for detailed analysis
    df_query_inflow['date'] = pd.to_datetime(df_query_inflow['startDate']).dt.strftime('%Y-%m')
    df_search_volume['total_volume'] = df_search_volume['pc'] + df_search_volume['mobile']
    df_detailed = pd.merge(
        df_query_inflow,
        df_search_volume[['searchQuery', 'date', 'total_volume']],
        on=['searchQuery', 'date'],
        how='left'
    )
    df_detailed['seo_efficiency_C'] = (df_detailed['searchInflow'] / df_detailed['total_volume']).fillna(0)
    df_detailed.replace([float('inf'), float('-inf')], 0, inplace=True)

    # Calculate keyword competitiveness (avg C and total inflow)
    keyword_agg = df_detailed.groupby('searchQuery').agg(
        average_C=('seo_efficiency_C', 'mean'),
        total_inflow=('searchInflow', 'sum'),
        post_count=('postId', 'nunique')
    ).reset_index()

    # Filter for keywords with significant data (e.g., inflow > 100, linked to >2 posts)
    competitive_keywords = keyword_agg[
        (keyword_agg['total_inflow'] > 100) & (keyword_agg['post_count'] > 2)
    ].sort_values(by='average_C', ascending=False)
    
    top_10_keywords = competitive_keywords.head(10)
    print("\n--- (Hypothesis 1) Top 10 High-Competitiveness Keywords ---")
    print(top_10_keywords)
    top_10_keywords.to_csv(os.path.join(output_dir, 'top_10_competitive_keywords.csv'), index=False, encoding='utf-8-sig')


    # --- 4. (Hypothesis 2) Analyze SEO Factors for Top Keywords ---
    print("\n--- (Hypothesis 2) Generating Correlation Heatmaps for Top Keywords ---")
    
    analysis_features = [
        'searchInflow', 'posting_score_numeric', 'exposure_score', 'score_가독성', 'score_주제', 
        'score_콘텐츠', 'score_키워드', 'score_형태소', 'score_후기/정보',
        'content_with_space', 'word_count', 'valid_images'
    ]

    # Convert posting_score to a numeric value for correlation
    def score_to_numeric(score):
        if pd.isna(score):
            return 0
        score = str(score)
        if '준최' in score:
            return int(score.split('-')[0].replace('준최', ''))
        elif '저품질' in score:
            return -1 * int(score.replace('저품질', ''))
        elif '일반' in score:
            return 0
        return 0
    
    df_master['posting_score_numeric'] = df_master['posting_score'].apply(score_to_numeric)


    for keyword in top_10_keywords['searchQuery']:
        # Get all post data for the specific keyword
        keyword_post_ids = df_detailed[df_detailed['searchQuery'] == keyword]['postId'].unique()
        df_keyword_posts = df_master[df_master['postId'].isin(keyword_post_ids)].copy()
        
        # Merge with inflow data to get specific inflow per post for that keyword
        df_inflow_per_post = df_detailed[df_detailed['searchQuery'] == keyword].groupby('postId')['searchInflow'].sum().reset_index()
        
        df_analysis = pd.merge(df_keyword_posts, df_inflow_per_post, on='postId')

        # Select only relevant features for correlation analysis
        df_corr = df_analysis[analysis_features].corr()

        # Generate and save heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(df_corr[['searchInflow']].sort_values(by='searchInflow', ascending=False), 
                    annot=True, cmap='viridis', fmt='.2f', linewidths=.5)
        
        plt.title(f"'{keyword}' 유입량과 포스트 요소 간의 상관관계", fontsize=16)
        plt.tight_layout()
        
        # Sanitize filename
        safe_keyword = "".join([c for c in keyword if c.isalpha() or c.isdigit() or c.isspace()]).rstrip()
        heatmap_path = os.path.join(output_dir, f'correlation_heatmap_{safe_keyword}.png')
        plt.savefig(heatmap_path)
        plt.close()
        print(f"Saved heatmap for '{keyword}' to {heatmap_path}")

if __name__ == "__main__":
    analyze_seo_correlations() 