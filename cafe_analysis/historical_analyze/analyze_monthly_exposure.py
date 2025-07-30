import os
import json
import pandas as pd
from tqdm import tqdm

def load_all_processed_data(processed_dir):
    """지정된 폴더의 모든 _analyzed.json 파일을 읽어 DataFrame으로 반환합니다."""
    all_posts = []
    if not os.path.exists(processed_dir):
        print(f"오류: '{processed_dir}' 폴더를 찾을 수 없습니다.")
        return pd.DataFrame()

    target_files = [f for f in os.listdir(processed_dir) if f.endswith('_analyzed.json')]
    print(f"총 {len(target_files)}개의 파일에서 메타데이터를 로드합니다...")

    for filename in tqdm(target_files, desc="메타데이터 로드 중"):
        file_path = os.path.join(processed_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_posts.extend(data)
            
    print(f"총 {len(all_posts)}개의 게시글 메타데이터를 로드했습니다.")
    
    # 중복된 article_id 제거 (마지막에 로드된 것을 유지)
    df = pd.DataFrame(all_posts)
    df = df.drop_duplicates(subset=['article_id'], keep='last')
    print(f"중복 제거 후 {len(df)}개의 고유한 게시글입니다.")
    
    return df

def analyze_clinic_exposure(merged_df, output_dir):
    """병원명 노출 조회수를 긍정/중립과 부정으로 나누어 분석합니다."""
    print("1. 월별/카페별/병원별 노출 조회수 분석 중...")
    
    # 분석에 필요한 데이터만 추출하고, 'analysis' 컬럼을 펼침
    df = merged_df[['article_id', 'cafe_name', 'analysis']].copy()
    df = pd.concat([df.drop(['analysis'], axis=1), df['analysis'].apply(pd.Series)], axis=1)

    # clinic_sentiments가 없는 행은 제외
    df.dropna(subset=['clinic_sentiments'], inplace=True)
    
    # 각 sentiment 항목을 별도의 행으로 펼침 (Explode)
    exploded_df = df.explode('clinic_sentiments')
    
    # 펼쳐진 dict에서 clinic_name과 sentiment를 새 컬럼으로 추출
    sentiments_df = exploded_df['clinic_sentiments'].apply(pd.Series)
    final_df = pd.concat([exploded_df.drop(['clinic_sentiments'], axis=1), sentiments_df], axis=1).dropna(subset=['clinic_name', 'sentiment'])
    
    # 월별 조회수 데이터와 다시 병합
    view_cols = [col for col in merged_df.columns if col.startswith('202')]
    final_df = pd.merge(final_df[['article_id', 'cafe_name', 'clinic_name', 'sentiment']], merged_df[['article_id'] + view_cols], on='article_id')

    # 1-1. 긍정/중립 분석
    pos_neu_df = final_df[final_df['sentiment'].isin(['긍정', '중립'])]
    pos_neu_agg = pos_neu_df.groupby(['cafe_name', 'clinic_name'])[view_cols].sum().reset_index()
    pos_neu_long = pos_neu_agg.melt(id_vars=['cafe_name', 'clinic_name'], var_name='month', value_name='views').sort_values(by=['cafe_name', 'clinic_name', 'month'])
    
    output_path_pos = os.path.join(output_dir, 'monthly_clinic_views_positive_neutral.csv')
    pos_neu_long.to_csv(output_path_pos, index=False, encoding='utf-8-sig')
    print(f" -> 긍정/중립 노출 조회수 분석 완료: {output_path_pos}")

    # 1-2. 부정 분석
    neg_df = final_df[final_df['sentiment'] == '부정']
    neg_agg = neg_df.groupby(['cafe_name', 'clinic_name'])[view_cols].sum().reset_index()
    neg_long = neg_agg.melt(id_vars=['cafe_name', 'clinic_name'], var_name='month', value_name='views').sort_values(by=['cafe_name', 'clinic_name', 'month'])

    output_path_neg = os.path.join(output_dir, 'monthly_clinic_views_negative.csv')
    neg_long.to_csv(output_path_neg, index=False, encoding='utf-8-sig')
    print(f" -> 부정 노출 조회수 분석 완료: {output_path_neg}")


def analyze_treatment_exposure(merged_df, output_dir):
    """진료 분야별 노출 조회수를 분석합니다."""
    print("2. 월별/카페별/진료 분야별 노출 조회수 분석 중...")

    df = merged_df[['article_id', 'cafe_name', 'analysis']].copy()
    df = pd.concat([df.drop(['analysis'], axis=1), df['analysis'].apply(pd.Series)], axis=1)
    df.dropna(subset=['related_treatments'], inplace=True)
    
    exploded_df = df.explode('related_treatments')
    
    view_cols = [col for col in merged_df.columns if col.startswith('202')]
    final_df = pd.merge(exploded_df[['article_id', 'cafe_name', 'related_treatments']], merged_df[['article_id'] + view_cols], on='article_id')

    agg_df = final_df.groupby(['cafe_name', 'related_treatments'])[view_cols].sum().reset_index()
    long_df = agg_df.melt(id_vars=['cafe_name', 'related_treatments'], var_name='month', value_name='views').sort_values(by=['cafe_name', 'related_treatments', 'month'])

    output_path = os.path.join(output_dir, 'monthly_treatment_views.csv')
    long_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f" -> 진료 분야별 노출 조회수 분석 완료: {output_path}")


def main():
    """메인 실행 함수: 데이터 로드, 병합, 분석 수행."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cafe_analysis_dir = os.path.dirname(script_dir)
    processed_dir = os.path.join(cafe_analysis_dir, 'data', 'historical_processed')
    output_dir = script_dir

    # 1. 데이터 로드
    views_df = pd.read_csv(os.path.join(script_dir, 'monthly_view_estimates.csv'))
    meta_df = load_all_processed_data(processed_dir)
    
    if meta_df.empty or views_df.empty:
        print("분석에 필요한 데이터가 부족하여 종료합니다.")
        return

    # 데이터 병합 전, 'article_id' 컬럼의 데이터 타입을 str으로 통일하여 오류 방지
    meta_df['article_id'] = meta_df['article_id'].astype(str)
    views_df['article_id'] = views_df['article_id'].astype(str)

    # 2. 데이터 병합
    merged_df = pd.merge(meta_df, views_df, on='article_id', how='inner', suffixes=('', '_views'))
    print(f"메타데이터와 조회수 데이터 병합 완료. 총 {len(merged_df)}개의 게시글.")
    
    # 3. 분석 함수 호출
    analyze_clinic_exposure(merged_df, output_dir)
    analyze_treatment_exposure(merged_df, output_dir)

    print("\n===== 모든 월별 노출 조회수 분석이 완료되었습니다. =====")


if __name__ == "__main__":
    main() 