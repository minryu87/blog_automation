import os
import json
import pandas as pd
from collections import defaultdict

def load_all_processed_data(processed_dir):
    """지정된 폴더의 모든 _analyzed.json 파일을 읽어 하나의 리스트로 통합합니다."""
    all_posts = []
    if not os.path.exists(processed_dir):
        print(f"오류: '{processed_dir}' 폴더를 찾을 수 없습니다.")
        return all_posts

    target_files = [f for f in os.listdir(processed_dir) if f.endswith('_analyzed.json')]
    print(f"총 {len(target_files)}개의 파일에서 데이터를 로드합니다...")

    for filename in target_files:
        file_path = os.path.join(processed_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_posts.extend(data)
            
    print(f"총 {len(all_posts)}개의 게시글 데이터를 로드했습니다.")
    return all_posts

def analyze_cafe_distribution(posts, output_dir):
    """club_id 기준으로 cafe_name의 개수 및 비율을 분석합니다."""
    print("1. 카페 분포 분석 중...")
    df = pd.DataFrame(posts)
    
    # club_id와 cafe_name이 모두 있는 데이터만 필터링
    df_filtered = df.dropna(subset=['club_id', 'cafe_name'])

    # club_id와 cafe_name으로 그룹화하여 개수 계산
    cafe_counts = df_filtered.groupby(['club_id', 'cafe_name']).size().reset_index(name='count')
    
    # club_id별 전체 개수 계산
    total_counts = cafe_counts.groupby('club_id')['count'].sum().reset_index(name='total_count')
    
    # 원본 데이터와 병합하여 비율 계산
    result = pd.merge(cafe_counts, total_counts, on='club_id')
    result['percentage'] = (result['count'] / result['total_count'] * 100).round(2)
    
    output_path = os.path.join(output_dir, 'cafe_distribution.csv')
    result.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f" -> 분석 완료. 결과 저장: {output_path}")

def analyze_clinic_sentiments(posts, output_dir):
    """카페별로 언급된 모든 병원의 긍정/중립/부정 개수를 분석합니다."""
    print("2. 병원 평판 분석 중...")
    sentiment_data = []
    for post in posts:
        # 'analysis' 키와 그 안의 'clinic_sentiments'가 유효한 리스트인지 확인
        sentiments = post.get('analysis', {}).get('clinic_sentiments')
        if isinstance(sentiments, list):
            for sentiment in sentiments:
                # sentiment가 딕셔너리 형태인지, 필요한 키가 있는지 확인
                if isinstance(sentiment, dict) and 'clinic_name' in sentiment and 'sentiment' in sentiment:
                    sentiment_data.append({
                        'cafe_name': post.get('cafe_name'),
                        'clinic_name': sentiment['clinic_name'],
                        'sentiment': sentiment['sentiment']
                    })

    if not sentiment_data:
        print(" -> 분석할 병원 평판 데이터가 없습니다.")
        return

    df = pd.DataFrame(sentiment_data)
    
    # cafe_name, clinic_name, sentiment로 그룹화하여 개수 계산 후 피벗
    sentiment_counts = df.groupby(['cafe_name', 'clinic_name', 'sentiment']).size().unstack(fill_value=0)
    
    # 컬럼 이름 재정의 (긍정, 중립, 부정 순서로 정렬)
    sentiment_counts = sentiment_counts.reindex(columns=['긍정', '중립', '부정'], fill_value=0)
    sentiment_counts.rename(columns={'긍정': 'positive', '중립': 'neutral', '부정': 'negative'}, inplace=True)
    
    output_path = os.path.join(output_dir, 'clinic_sentiments.csv')
    sentiment_counts.to_csv(output_path, encoding='utf-8-sig')
    print(f" -> 분석 완료. 결과 저장: {output_path}")

def analyze_related_treatments(posts, output_dir):
    """카페별 관련 진료 분야 언급 횟수를 분석합니다."""
    print("3. 관련 진료 분야 분석 중...")
    treatment_data = []
    for post in posts:
        treatments = post.get('analysis', {}).get('related_treatments')
        if isinstance(treatments, list):
            for treatment in treatments:
                treatment_data.append({
                    'cafe_name': post.get('cafe_name'),
                    'treatment': treatment
                })

    if not treatment_data:
        print(" -> 분석할 진료 분야 데이터가 없습니다.")
        return
        
    df = pd.DataFrame(treatment_data)
    treatment_counts = df.groupby(['cafe_name', 'treatment']).size().reset_index(name='count')
    
    output_path = os.path.join(output_dir, 'related_treatments.csv')
    treatment_counts.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f" -> 분석 완료. 결과 저장: {output_path}")


def main():
    """메인 실행 함수: 데이터 로드 및 모든 분석 수행."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cafe_analysis_dir = os.path.dirname(script_dir)
    
    processed_dir = os.path.join(cafe_analysis_dir, 'data', 'historical_processed')
    output_dir = script_dir  # 결과를 현재 스크립트 위치에 저장

    # 1. 모든 데이터 로드
    all_posts = load_all_processed_data(processed_dir)
    if not all_posts:
        return

    # 2. 각 통계 분석 실행
    analyze_cafe_distribution(all_posts, output_dir)
    analyze_clinic_sentiments(all_posts, output_dir)
    analyze_related_treatments(all_posts, output_dir)

    print("\n===== 모든 통계 분석이 완료되었습니다. =====")


if __name__ == "__main__":
    main() 