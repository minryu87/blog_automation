import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import os
import re
from sklearn.cluster import KMeans
import warnings

# Suppress KMeans warning about memory leaks on Windows with MKL, which can be noisy.
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.cluster._kmeans')


# --- Helper Functions & Models ---

_model = None

def get_model():
    """Initializes and returns the sentence transformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return _model

# --- Feature Calculation Functions ---

def calculate_relative_query_fulfillment_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    (STABLE) Calculates how well a post fulfills the semantic intent of its representative query,
    relative to its competitors for the same query.
    Source: P3_Benchmark_CTR_Query_Relation_code_log.json
    """
    feature_name = 'relative_query_fulfillment_score'
    if df.empty:
        df[feature_name] = np.nan
        return df

    model = get_model()
    
    df['post_body'] = df['post_body'].fillna('').astype(str)
    df['representative_query'] = df['representative_query'].fillna('').astype(str)

    unique_queries = df['representative_query'].unique().tolist()
    post_bodies = df['post_body'].tolist()

    query_embeddings = model.encode(unique_queries, convert_to_tensor=True, show_progress_bar=False)
    body_embeddings = model.encode(post_bodies, convert_to_tensor=True, show_progress_bar=False)

    query_embedding_map = {query: emb for query, emb in zip(unique_queries, query_embeddings)}
    
    ordered_query_embeddings = torch.stack([query_embedding_map.get(q, torch.zeros(model.get_sentence_embedding_dimension()).to(body_embeddings.device)) for q in df['representative_query']])
    
    fulfillment_scores = util.cos_sim(body_embeddings, ordered_query_embeddings).diag()
    df['temp_fulfillment_score'] = fulfillment_scores.cpu().numpy()

    avg_scores = df.groupby(['representative_query', 'source'])['temp_fulfillment_score'].mean().unstack()

    if 'ours' not in avg_scores.columns:
        avg_scores['ours'] = np.nan
    if 'competitor' not in avg_scores.columns:
        avg_scores['competitor'] = np.nan

    relative_scores = (avg_scores['ours'] / avg_scores['competitor']).fillna(1.0)
    relative_scores.replace([np.inf, -np.inf], 1.0, inplace=True)

    df[feature_name] = df['representative_query'].map(relative_scores)
    df[feature_name].fillna(1.0, inplace=True)

    df.drop(columns=['temp_fulfillment_score'], inplace=True)

    print(f"  - Feature calculated: {feature_name}")
    return df

def calculate_relative_semantic_actionability(df: pd.DataFrame) -> pd.DataFrame:
    """
    (STABLE) Calculates the 'relative_semantic_actionability' feature.
    Source: P2_Benchmark_CTR_Semantic_code_log.json
    """
    feature_name = 'relative_semantic_actionability'

    if not df.empty:
        df[feature_name] = np.nan
    else:
        return df

    action_keywords = [
        '방법', '해결', '가이드', '팁', '전략', '단계', '하는 법', '솔루션',
        'how to', 'solution', 'guide', 'tip', 'strategy', 'steps', 'tutorial'
    ]

    try:
        model = get_model()

        action_embeddings = model.encode(action_keywords, convert_to_tensor=True, show_progress_bar=False)
        actionability_vector = action_embeddings.mean(dim=0)

        df_work = df.copy()
        bodies = df_work['post_body'].fillna('').astype(str).tolist()
        body_embeddings = model.encode(bodies, convert_to_tensor=True, show_progress_bar=False)

        actionability_vector_device = actionability_vector.to(body_embeddings.device)

        actionability_scores = util.cos_sim(body_embeddings, actionability_vector_device)
        df_work['temp_action_score'] = actionability_scores.cpu().numpy().flatten()

        for query, group in df_work.groupby('representative_query'):
            ours_posts = group[group['source'] == 'ours']
            competitor_posts = group[group['source'] == 'competitor']

            if ours_posts.empty or competitor_posts.empty:
                continue

            avg_competitor_score = competitor_posts['temp_action_score'].mean()

            our_scores = ours_posts['temp_action_score']
            our_indices = ours_posts.index

            if avg_competitor_score > 1e-6:
                ratio = our_scores / avg_competitor_score
                df.loc[our_indices, feature_name] = ratio
            else:
                df.loc[our_indices, feature_name] = 1.0

    except Exception as e:
        print(f"Error generating {feature_name}: {e}")
        if feature_name not in df.columns:
            df[feature_name] = np.nan
    
    # 최종적으로 NaN 값을 채워준다.
    df[feature_name] = df[feature_name].fillna(1.0)
    
    print(f"  - Feature calculated: {feature_name}")
    return df

def calculate_title_body_semantic_cohesion(df: pd.DataFrame) -> pd.DataFrame:
    """
    (STABLE) Calculates the semantic cohesion between each post's title and body.
    Source: P3_Benchmark_CTR_TitleBody_Relation_code_log.json
    """
    feature_name = 'title_body_semantic_cohesion'
    if df.empty:
        df[feature_name] = pd.Series(dtype=float)
        return df

    try:
        model = get_model()
        titles = df['post_title'].fillna('').astype(str).tolist()
        bodies = df['post_body'].fillna('').astype(str).tolist()

        title_embeddings = model.encode(titles, convert_to_tensor=True, show_progress_bar=False)
        body_embeddings = model.encode(bodies, convert_to_tensor=True, show_progress_bar=False)

        cosine_scores = util.cos_sim(title_embeddings, body_embeddings).diag()
        df[feature_name] = cosine_scores.cpu().tolist()

    except Exception as e:
        print(f"Error calculating {feature_name}: {e}")
        df[feature_name] = np.nan
    
    print(f"  - Feature calculated: {feature_name}")
    return df

def calculate_title_hook_pattern_presence(df: pd.DataFrame) -> pd.DataFrame:
    """
    (STABLE) Analyzes post titles for the presence of 'hook' patterns.
    Source: P1_Internal_CTR_All_Intrinsic_code_log.json
    """
    feature_name = 'title_hook_pattern_presence'
    if not df.empty:
        hook_keywords = [
            '방법', '후기', '비용', '해결', '정리', '추천', '꿀팁', '비교', '총정리', '솔직'
        ]
        
        combined_pattern = re.compile(
            r'(\d+)|' + 
            r'(\?|까\s*?$)|' + 
            f'({"|".join(hook_keywords)})'
        )
        titles = df['post_title'].fillna('')
        df[feature_name] = titles.str.contains(combined_pattern, regex=True, na=False).astype(int)
    else:
        df[feature_name] = pd.Series(dtype=int)

    print(f"  - Feature calculated: {feature_name}")
    return df

def calculate_archetypal_contrast_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    (STABLE) Calculates the Archetypal Contrast Score for each post.
    Source: B그룹_본문Semantic분석_code_log.json
    """
    feature_name = 'archetypal_contrast_score'
    
    if df.empty:
        df[feature_name] = pd.Series(dtype=np.float64)
        return df

    model = get_model()
    df[feature_name] = np.nan

    df_copy = df.copy()
    df_copy['combined_text'] = (
        df_copy['post_title'].fillna('') + ' ' +
        df_copy['post_body'].fillna('') + ' ' +
        df_copy['morpheme_words'].fillna('')
    ).str.strip()

    df[feature_name] = np.nan

    texts_to_encode = df_copy['combined_text'].tolist()
    all_embeddings = model.encode(texts_to_encode, convert_to_tensor=True, show_progress_bar=False)

    grouped = df_copy.groupby('top_category_name')

    for _, group in grouped:
        ours_posts = group[group['source'] == 'ours']
        competitor_posts = group[group['source'] == 'competitor']

        low_perf_ours = pd.DataFrame(columns=df_copy.columns)
        high_perf_ours = pd.DataFrame(columns=df_copy.columns)
        if not ours_posts.empty and ours_posts['non_brand_inflow'].notna().any():
            inflow_data = ours_posts['non_brand_inflow'].dropna()
            if len(inflow_data) > 0:
                low_quantile = inflow_data.quantile(0.3)
                high_quantile = inflow_data.quantile(0.7)
                low_perf_ours = ours_posts[ours_posts['non_brand_inflow'] <= low_quantile]
                high_perf_ours = ours_posts[ours_posts['non_brand_inflow'] >= high_quantile]

        high_perf_group = pd.concat([competitor_posts, high_perf_ours])
        low_perf_group = low_perf_ours

        group_locs = df.index.get_indexer_for(group.index)
        high_perf_locs = df.index.get_indexer_for(high_perf_group.index)
        low_perf_locs = df.index.get_indexer_for(low_perf_group.index)

        valid_group_locs = group_locs[group_locs != -1]
        valid_high_perf_locs = high_perf_locs[high_perf_locs != -1]
        valid_low_perf_locs = low_perf_locs[low_perf_locs != -1]
        
        high_perf_centroids = torch.Tensor().to(all_embeddings.device)
        low_perf_centroids = torch.Tensor().to(all_embeddings.device)

        if len(valid_high_perf_locs) > 0:
            high_perf_embeddings = all_embeddings[valid_high_perf_locs]
            n_clusters_high = min(3, len(valid_high_perf_locs))
            if n_clusters_high > 0:
                kmeans_high = KMeans(n_clusters=n_clusters_high, random_state=42, n_init='auto')
                kmeans_high.fit(high_perf_embeddings.cpu().numpy())
                high_perf_centroids = torch.tensor(kmeans_high.cluster_centers_, device=all_embeddings.device)

        if len(valid_low_perf_locs) > 0:
            low_perf_embeddings = all_embeddings[valid_low_perf_locs]
            n_clusters_low = min(3, len(valid_low_perf_locs))
            if n_clusters_low > 0:
                kmeans_low = KMeans(n_clusters=n_clusters_low, random_state=42, n_init='auto')
                kmeans_low.fit(low_perf_embeddings.cpu().numpy())
                low_perf_centroids = torch.tensor(kmeans_low.cluster_centers_, device=all_embeddings.device)
        
        if len(valid_group_locs) > 0:
            group_embeddings = all_embeddings[valid_group_locs]
            
            if high_perf_centroids.numel() > 0:
                sim_to_high = util.cos_sim(group_embeddings, high_perf_centroids)
                max_sim_high, _ = torch.max(sim_to_high, dim=1)
            else:
                max_sim_high = torch.zeros(len(valid_group_locs), device=all_embeddings.device)

            if low_perf_centroids.numel() > 0:
                sim_to_low = util.cos_sim(group_embeddings, low_perf_centroids)
                max_sim_low, _ = torch.max(sim_to_low, dim=1)
            else:
                max_sim_low = torch.zeros(len(valid_group_locs), device=all_embeddings.device)
            
            contrast_scores = (max_sim_high - max_sim_low).cpu().tolist()
            
            df.loc[group.index, feature_name] = contrast_scores

    df[feature_name].fillna(0, inplace=True)
    print(f"  - Feature calculated: {feature_name}")
    return df

def calculate_semantic_contrast_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    (STABLE) Calculates the semantic contrast score for each post.
    Source: B그룹_본문Semantic분석_code_log.json
    """
    feature_name = 'semantic_contrast_score'
    if df.empty:
        df[feature_name] = pd.Series(dtype=np.float64)
        return df

    model = get_model()

    df['temp_combined_text'] = df['post_title'].fillna('') + ' ' + \
                              df['post_body'].fillna('') + ' ' + \
                              df['morpheme_words'].fillna('')

    original_index = df.index
    texts = df['temp_combined_text'].tolist()
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)

    index_to_position = {idx: pos for pos, idx in enumerate(original_index)}
    df[feature_name] = np.nan

    for category, group in df.groupby('top_category_name'):
        if len(group) < 3:
            continue

        ours_in_group = group[group['source'] == 'ours']
        competitors_in_group = group[group['source'] == 'competitor']

        if ours_in_group.empty:
            continue

        inflow_data = ours_in_group['non_brand_inflow'].dropna()
        if len(inflow_data) < 2:
            continue

        high_perf_quantile = inflow_data.quantile(0.7)
        low_perf_quantile = inflow_data.quantile(0.3)

        high_perf_ours_indices = ours_in_group[ours_in_group['non_brand_inflow'] >= high_perf_quantile].index
        low_perf_indices = ours_in_group[ours_in_group['non_brand_inflow'] <= low_perf_quantile].index
        competitor_indices = competitors_in_group.index

        high_perf_total_indices = competitor_indices.union(high_perf_ours_indices)
        
        if high_perf_total_indices.empty or low_perf_indices.empty:
            continue

        high_perf_positions = [index_to_position[idx] for idx in high_perf_total_indices]
        low_perf_positions = [index_to_position[idx] for idx in low_perf_indices]
        group_positions = [index_to_position[idx] for idx in group.index]

        high_perf_embeddings = embeddings[high_perf_positions]
        low_perf_embeddings = embeddings[low_perf_positions]
        group_embeddings = embeddings[group_positions]
        
        high_centroid = high_perf_embeddings.mean(axis=0, keepdim=True)
        low_centroid = low_perf_embeddings.mean(axis=0, keepdim=True)

        sim_to_high = util.cos_sim(group_embeddings, high_centroid)
        sim_to_low = util.cos_sim(group_embeddings, low_centroid)

        contrast_scores = (sim_to_high - sim_to_low).flatten().cpu().numpy()

        df.loc[group.index, feature_name] = contrast_scores

    df = df.drop(columns=['temp_combined_text'])
    print(f"  - Feature calculated: {feature_name}")
    return df

def calculate_competitive_positioning_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    (STABLE) Calculates the competitive positioning score for each post.
    Source: B그룹_본문Semantic분석_code_log.json
    """
    feature_name = 'competitive_positioning_score'
    if df.empty:
        df[feature_name] = pd.Series(dtype=float)
        return df

    model = get_model()
    df[feature_name] = np.nan

    texts_to_embed = (df['post_title'].fillna('') + ' ' + df['post_body'].fillna('')).tolist()
    all_embeddings = model.encode(texts_to_embed, convert_to_tensor=True, show_progress_bar=False)

    for category, group_df in df.groupby('top_category_name'):
        group_indices = group_df.index
        ours_df = group_df[group_df['source'] == 'ours']

        if len(ours_df) < 10:
            continue

        low_perf_threshold = ours_df['non_brand_inflow'].quantile(0.3, interpolation='lower')
        high_perf_threshold = ours_df['non_brand_inflow'].quantile(0.7, interpolation='higher')

        high_perf_indices = group_df[
            (group_df['source'] == 'competitor') |
            ((group_df['source'] == 'ours') & (group_df['non_brand_inflow'] >= high_perf_threshold))
        ].index

        low_perf_indices = ours_df[ours_df['non_brand_inflow'] <= low_perf_threshold].index

        if high_perf_indices.empty or low_perf_indices.empty:
            continue

        high_perf_tensor_indices = df.index.get_indexer(high_perf_indices)
        low_perf_tensor_indices = df.index.get_indexer(low_perf_indices)
        group_tensor_indices = df.index.get_indexer(group_indices)

        high_embeddings = all_embeddings[high_perf_tensor_indices]
        low_embeddings = all_embeddings[low_perf_tensor_indices]
        group_embeddings = all_embeddings[group_tensor_indices]

        k_high = min(3, len(high_embeddings))
        k_low = min(3, len(low_embeddings))

        if k_high == 0 or k_low == 0:
            continue

        kmeans_high = KMeans(n_clusters=k_high, random_state=42, n_init='auto').fit(high_embeddings.cpu().numpy())
        high_archetypes = torch.tensor(kmeans_high.cluster_centers_).to(model.device)

        kmeans_low = KMeans(n_clusters=k_low, random_state=42, n_init='auto').fit(low_embeddings.cpu().numpy())
        low_archetypes = torch.tensor(kmeans_low.cluster_centers_).to(model.device)

        sim_to_high_archetypes = util.cos_sim(group_embeddings, high_archetypes)
        sim_to_low_archetypes = util.cos_sim(group_embeddings, low_archetypes)

        max_sim_high = torch.max(sim_to_high_archetypes, dim=1).values
        max_sim_low = torch.max(sim_to_low_archetypes, dim=1).values
        
        min_dist_high = 1 - max_sim_high
        min_dist_low = 1 - max_sim_low

        score = min_dist_low / (min_dist_high + min_dist_low + 1e-9)

        df.loc[group_indices, feature_name] = score.cpu().numpy()

    print(f"  - Feature calculated: {feature_name}")
    return df


# --- Main Execution Logic ---

def main():
    """
    Main function to run the feature calculation test.
    """
    print("피처 데이터셋 생성 스크립트를 시작합니다.")

    # --- 1. Load Data ---
    master_df_path = "/Users/min/codes/medilawyer_sales/blog_automation/data/data_processed/master_post_data.csv"
    output_path = "/Users/min/codes/medilawyer_sales/blog_automation/data/modeling/feature_calculate"
    
    print(f"마스터 데이터 로딩: {master_df_path}")
    try:
        master_df = pd.read_csv(master_df_path)
    except FileNotFoundError:
        print(f"오류: 마스터 파일을 찾을 수 없습니다 - {master_df_path}")
        return

    # --- 2. Define Feature Lists ---
    CTR_FEATURE_FUNCS = [
        calculate_relative_query_fulfillment_score,
        calculate_relative_semantic_actionability,
        calculate_title_body_semantic_cohesion,
        calculate_title_hook_pattern_presence,
    ]
    INFLOW_FEATURE_FUNCS = [
        calculate_archetypal_contrast_score,
        calculate_semantic_contrast_score,
        calculate_competitive_positioning_score,
    ]

    # --- 3. Process CTR Features ---
    print("\n--- CTR 피처 계산 시작 ---")
    # Start with identifiers
    identifier_cols = [col for col in ['post_identifier', 'source', 'top_category_name', 'representative_query'] if col in master_df.columns]
    ctr_results_df = master_df[identifier_cols].copy()
    
    # Run each feature function and add the result column
    temp_df = master_df.copy()
    for func in CTR_FEATURE_FUNCS:
        temp_df = func(temp_df)
        feature_name = [col for col in temp_df.columns if col not in ctr_results_df.columns and col in temp_df.columns and 'temp' not in col][-1]
        ctr_results_df[feature_name] = temp_df[feature_name]

    # --- 4. Process Inflow Features ---
    print("\n--- Inflow 피처 계산 시작 ---")
    inflow_results_df = master_df[identifier_cols].copy()
    
    temp_df = master_df.copy()
    for func in INFLOW_FEATURE_FUNCS:
        temp_df = func(temp_df)
        feature_name = [col for col in temp_df.columns if col not in inflow_results_df.columns and col in temp_df.columns and 'temp' not in col][-1]
        inflow_results_df[feature_name] = temp_df[feature_name]

    # --- 5. Save Results ---
    os.makedirs(output_path, exist_ok=True)
    ctr_output_file = os.path.join(output_path, "ctr_feature_value.csv")
    inflow_output_file = os.path.join(output_path, "inflow_feature_value.csv")

    print(f"\nCTR 피처 결과 저장: {ctr_output_file}")
    ctr_results_df.to_csv(ctr_output_file, index=False, encoding='utf-8-sig')
    
    print(f"Inflow 피처 결과 저장: {inflow_output_file}")
    inflow_results_df.to_csv(inflow_output_file, index=False, encoding='utf-8-sig')

    print("\n모든 작업이 완료되었습니다.")


if __name__ == "__main__":
    main() 