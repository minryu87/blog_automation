import pandas as pd
import os
import json
import time
from tqdm import tqdm
import re # Added for _safe_convert

# To allow importing from the parent 'blog_automation' directory
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from blog_automation.app.services import data_collector, data_parser

def _safe_convert(value, type_func, default=None):
    """Safely converts a value to a specified type, returning a default on failure."""
    if value is None:
        return default
    try:
        # Remove non-numeric characters like '%' or ',' before converting
        cleaned_value = re.sub(r'[^\d.]', '', str(value))
        return type_func(cleaned_value)
    except (ValueError, TypeError):
        return default

def _process_parsed_json(parsed_data: dict, url: str) -> dict:
    """
    Flattens the complex parsed JSON data into a single-level dictionary
    that matches the schema of 'agent_base_dataset.csv'.
    """
    flat_data = {}

    def get_score(details_list, category_name):
        if not isinstance(details_list, list): return None
        for item in details_list:
            if item.get('category') == category_name:
                return _safe_convert(item.get('score'), float)
        return None

    flat_data['post_identifier'] = url
    flat_data['source'] = 'competitor'
    flat_data['post_id'] = None
    flat_data['post_title'] = parsed_data.get('post_title')
    flat_data['post_body'] = parsed_data.get('post_content')
    flat_data['posting_level'] = parsed_data.get('posting_score')
    flat_data['exposure_score'] = _safe_convert(parsed_data.get('exposure_score'), float)


    score_details = parsed_data.get('posting_score_details')
    flat_data['readability_score'] = get_score(score_details, '가독성')
    flat_data['topic_focus_score'] = get_score(score_details, '주제')
    flat_data['content_quality_score'] = get_score(score_details, '콘텐츠')
    flat_data['keyword_usage_score'] = get_score(score_details, '키워드')
    flat_data['morpheme_score'] = get_score(score_details, '형태소')
    flat_data['review_info_score'] = get_score(score_details, '후기/정보')

    text_analysis = parsed_data.get('text_analysis', {}) or {}
    flat_data['char_count_with_space'] = _safe_convert(text_analysis.get('content_with_space'), int, 0)
    flat_data['char_count_without_space'] = _safe_convert(text_analysis.get('content_without_space'), int, 0)
    
    morpheme_counts = parsed_data.get('morpheme_counts', {}) or {}
    flat_data['morpheme_count'] = _safe_convert(morpheme_counts.get('morpheme_count'), int, 0)
    flat_data['word_count'] = _safe_convert(morpheme_counts.get('word_count'), int, 0)


    image_analysis = parsed_data.get('image_analysis', {}) or {}
    valid_images = _safe_convert(image_analysis.get('valid_images'), int, 0)
    invalid_images = _safe_convert(image_analysis.get('invalid_images'), int, 0)
    material_images = _safe_convert(image_analysis.get('material_images'), int, 0)
    uploaded_images = _safe_convert(image_analysis.get('uploaded_images'), int, 0)
    content_images = _safe_convert(image_analysis.get('content_images'), int, 0)

    flat_data['valid_image_count'] = valid_images
    flat_data['invalid_image_count'] = invalid_images
    flat_data['material_image_count'] = material_images
    flat_data['uploaded_image_count'] = uploaded_images
    flat_data['content_image_count'] = content_images
    flat_data['total_image_count'] = valid_images + invalid_images + material_images + uploaded_images + content_images
    
    category_analysis = parsed_data.get('category_analysis')
    if category_analysis and len(category_analysis) > 0:
        top_category = category_analysis[0]
        flat_data['top_category_name'] = top_category.get('category_name')
        flat_data['top_category_percentage'] = top_category.get('percentage')
        flat_data['category_keywords'] = top_category.get('keywords')
    else:
        flat_data['top_category_name'] = None
        flat_data['top_category_percentage'] = None
        flat_data['category_keywords'] = None
        
    morpheme_analysis = parsed_data.get('morpheme_analysis_details')
    if morpheme_analysis and len(morpheme_analysis) > 0:
        top_morpheme = morpheme_analysis[0]
        flat_data['top_morpheme_type'] = top_morpheme.get('morpheme_type')
        flat_data['top_morpheme_percentage'] = top_morpheme.get('percentage')
        flat_data['morpheme_words'] = top_morpheme.get('words')
    else:
        flat_data['top_morpheme_type'] = None
        flat_data['top_morpheme_percentage'] = None
        flat_data['morpheme_words'] = None
        
    flat_data['non_brand_inflow'] = None
    flat_data['non_brand_average_ctr'] = None
    
    return flat_data

def find_local_json_file(url: str) -> str | None:
    """Finds the path to a pre-fetched AND pre-parsed JSON file based on its URL."""
    try:
        # 파일명 생성 로직을 data_collector.py에서 직접 가져와 사용합니다.
        safe_filename_base = f"{url.replace('/', '_').replace(':', '_')}.html"
        json_filename = os.path.splitext(safe_filename_base)[0] + '.json'
        
        # 프로젝트 루트를 기준으로 경로를 재설정하여 안정성을 높입니다.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
        competitor_data_dir = os.path.join(project_root, 'blog_automation', 'data', 'data_processed', 'NSIDE-postAnalysis-competitor')

        json_path = os.path.join(competitor_data_dir, json_filename)
        
        if os.path.exists(json_path):
            return json_path
        else:
            return None
    except Exception as e:
        print(f"\n[ERROR] Error finding local json for {url}: {e}")
        return None

def fetch_nside_data_for_url(url: str, use_local_files: bool = True) -> dict | None:
    """
    Now optimized to read pre-parsed local JSON files directly.
    """
    json_path = None
    try:
        if use_local_files:
            json_path = find_local_json_file(url)
            if not json_path:
                # 로컬 JSON 파일이 없으면 건너뜁니다.
                return None
        else:
            # 실시간 수집 로직은 현재 사용되지 않지만, 만약을 위해 남겨둡니다.
            html_path = data_collector.fetch_html(url)
            if not html_path:
                return None
            json_path = data_parser.parse_html_file(html_path)
            if not json_path:
                return None
        
        # 1. Load the pre-parsed JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            parsed_data = json.load(f)
        
        # 2. Flatten the data to match the master dataset schema
        flat_data = _process_parsed_json(parsed_data, url)
        
        return flat_data

    except Exception as e:
        print(f"\nAn unexpected error occurred during processing for {url}: {e}")
        return None

def enrich_topic_cluster_data():
    """
    Enriches the topic_clusters.csv by fetching detailed data for all posts
    (our posts and competitors) and creating a master dataset for analysis.
    """
    # --- 1. Define File Paths ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    blog_automation_dir = os.path.join(script_dir, '..')
    
    clusters_path = os.path.join(blog_automation_dir, 'data', 'data_processed', 'topic_clusters.csv')
    our_data_path = os.path.join(blog_automation_dir, 'agents', 'agent_base_dataset.csv')
    output_dir = os.path.join(blog_automation_dir, 'data', 'data_processed')
    output_path = os.path.join(output_dir, 'master_post_data.csv')

    os.makedirs(output_dir, exist_ok=True)

    # --- 2. Load Datasets ---
    print("Loading datasets...")
    df_clusters = pd.read_csv(clusters_path)
    df_ours_base = pd.read_csv(our_data_path)
    
    # --- 3. Identify Unique Posts to Fetch ---
    unique_our_post_ids = df_clusters['our_postId'].unique()
    unique_competitor_urls = df_clusters['competitor_post_url'].unique()

    print(f"Found {len(unique_our_post_ids)} unique 'our' posts.")
    print(f"Found {len(unique_competitor_urls)} unique competitor posts to fetch.")

    # --- 4. Prepare Our Post Data ---
    print("\nPreparing data for our posts...")
    df_ours_enriched = df_ours_base[df_ours_base['post_id'].isin(unique_our_post_ids)].copy()
    
    # Add missing image columns to align schemas with competitors
    image_cols_to_add = {
        'material_image_count': 0,
        'uploaded_image_count': 0, # This might be available in some datasets, but we default to 0 for consistency
        'content_image_count': 0
    }
    for col, default_val in image_cols_to_add.items():
        if col not in df_ours_enriched.columns:
            df_ours_enriched[col] = default_val

    # Fill NA in existing image counts just in case and ensure correct type
    df_ours_enriched['valid_image_count'] = pd.to_numeric(df_ours_enriched['valid_image_count'], errors='coerce').fillna(0)
    df_ours_enriched['invalid_image_count'] = pd.to_numeric(df_ours_enriched['invalid_image_count'], errors='coerce').fillna(0)

    # Calculate total_image_count consistently across all data sources
    df_ours_enriched['total_image_count'] = (
        df_ours_enriched['valid_image_count'] +
        df_ours_enriched['invalid_image_count'] +
        df_ours_enriched['material_image_count'] +
        df_ours_enriched['uploaded_image_count'] +
        df_ours_enriched['content_image_count']
    ).astype(int)

    df_ours_enriched['source'] = 'ours'
    df_ours_enriched['post_id'] = df_ours_enriched['post_id'].astype(str)
    df_ours_enriched.rename(columns={'post_id': 'post_identifier'}, inplace=True)
    print(f"Prepared {len(df_ours_enriched)} of our posts from existing data.")

    # --- 5. Fetch Competitor Data ---
    print("\nProcessing pre-parsed data for competitor posts (super-fast operation)...")
    competitor_data_list = []
    
    for url in tqdm(unique_competitor_urls, desc="Processing competitor JSON"):
        # use_local_files=True를 명시적으로 전달합니다.
        data = fetch_nside_data_for_url(url, use_local_files=True)
        if data:
            competitor_data_list.append(data)
            
    df_competitors_enriched = pd.DataFrame(competitor_data_list)
    print(f"Successfully processed data for {len(df_competitors_enriched)} competitor posts.")

    # --- 6. Combine and Save Master Dataset ---
    print("\nCombining all data into a master dataset...")
    df_master = pd.concat([df_ours_enriched, df_competitors_enriched], ignore_index=True, sort=False)
    
    df_master.to_csv(output_path, index=False, encoding='utf-8-sig')

    print("\n--- Success! ---")
    print(f"Master dataset with {len(df_master)} total entries created.")
    print(f"Saved to: {output_path}")
    print("\n--- Sample Output ---")
    print("Columns in our data:", df_ours_enriched.columns.tolist())
    print("Columns in competitor data:", df_competitors_enriched.columns.tolist())
    print(df_master.head())
    print(df_master.tail())
    print("---------------------\n")

if __name__ == '__main__':
    enrich_topic_cluster_data() 