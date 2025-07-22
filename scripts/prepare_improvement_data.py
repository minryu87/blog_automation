import pandas as pd
import joblib
import os
from datetime import datetime

def main():
    """
    1. ctr_feature_value.csv와 master_post_data.csv 로드
    2. 두 데이터를 병합하여 title, body 추가
    3. 훈련된 CTR 모델 로드 및 예측
    4. 결과를 CSV로 저장
    """
    print("포스트 개선을 위한 데이터 준비 스크립트를 시작합니다 (단순 병합 방식).")

    # --- 경로 설정 ---
    base_path = "/Users/min/codes/medilawyer_sales/blog_automation"
    feature_data_path = os.path.join(base_path, "data/modeling/feature_calculate/ctr_feature_value.csv")
    master_data_path = os.path.join(base_path, "data/data_processed/master_post_data.csv")
    model_path = os.path.join(base_path, "data/modeling/trained_models/ctr_champion_model.joblib")
    output_dir = os.path.join(base_path, "data/post_edit")
    os.makedirs(output_dir, exist_ok=True)

    # --- 데이터 로딩 ---
    try:
        print(f"\n[1/3] 데이터 로딩...")
        print(f"  - 피처 데이터: {feature_data_path}")
        feature_df = pd.read_csv(feature_data_path)
        print(f"  - 마스터 데이터: {master_data_path}")
        master_df = pd.read_csv(master_data_path)
        print("데이터 로딩 완료.")
    except FileNotFoundError as e:
        print(f"오류: 데이터 파일을 찾을 수 없습니다. {e}")
        return

    # --- 데이터 전처리 및 병합 ---
    print("\n[2/3] 데이터 전처리 및 병합...")
    
    # 컬럼 이름 통일 (post_identifier -> post_id)
    feature_df.rename(columns={'post_identifier': 'post_id'}, inplace=True, errors='ignore')
    master_df.rename(columns={'post_identifier': 'post_id'}, inplace=True, errors='ignore')
        
    # 타입 통일
    feature_df['post_id'] = feature_df['post_id'].astype(str)
    master_df['post_id'] = master_df['post_id'].astype(str)
    
    # 양쪽 모두 중복 제거
    master_df.drop_duplicates(subset='post_id', keep='last', inplace=True)
    feature_df.drop_duplicates(subset='post_id', keep='last', inplace=True)

    # 'ours'인 데이터만 사용
    feature_df_ours = feature_df[feature_df['source'] == 'ours'].copy()

    # 필요한 컬럼만 선택하여 병합
    merged_df = pd.merge(
        feature_df_ours,
        master_df[['post_id', 'post_title', 'post_body']],
        on='post_id',
        how='inner' # 피처와 본문이 모두 있는 포스트만 선택
    )
    print("데이터 병합 완료.")

    # --- 모델 로드 및 CTR 예측 ---
    try:
        print(f"\n[3/3] CTR 예측 및 결과 저장...")
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"오류: 훈련된 모델 파일을 찾을 수 없습니다. 경로: {model_path}")
        return

    feature_names_for_model = [
        'relative_query_fulfillment_score',
        'title_body_semantic_cohesion'
    ]
    print(f"  - 모델 예측에 사용할 피처: {feature_names_for_model}")
    
    X_pred = merged_df[feature_names_for_model].copy()
    
    # 예측 전 NaN 값 처리
    for col in feature_names_for_model:
        if X_pred[col].isnull().any():
            median_val = X_pred[col].median()
            if pd.isna(median_val):
                median_val = 0
            X_pred[col].fillna(median_val, inplace=True)
            print(f"  - '{col}' 피처의 NaN 값을 중앙값({median_val:.4f})으로 대체했습니다.")

    merged_df['predicted_ctr'] = model.predict(X_pred)
    
    # --- 결과 저장 ---
    output_columns = [
        'post_id', 'post_title', 'post_body', 
        'relative_query_fulfillment_score', 'relative_semantic_actionability',
        'title_body_semantic_cohesion', 'title_hook_pattern_presence',
        'predicted_ctr'
    ]
    # 결과 DF에 없는 컬럼이 있을 경우를 대비
    for col in output_columns:
        if col not in merged_df.columns:
            merged_df[col] = pd.NA

    results_df = merged_df[output_columns].sort_values(by='predicted_ctr', ascending=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"{timestamp}_improvement_candidates.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"결과가 성공적으로 저장되었습니다: {output_path}")
    print("\n모든 작업이 완료되었습니다.")


if __name__ == "__main__":
    main() 