import os
import json
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from tqdm import tqdm

def load_all_processed_data(processed_dir):
    """지정된 폴더의 모든 _analyzed.json 파일을 읽어 하나의 리스트로 통합합니다."""
    all_posts = []
    if not os.path.exists(processed_dir):
        print(f"오류: '{processed_dir}' 폴더를 찾을 수 없습니다.")
        return all_posts

    target_files = [f for f in os.listdir(processed_dir) if f.endswith('_analyzed.json')]
    print(f"총 {len(target_files)}개의 파일에서 데이터를 로드합니다...")

    for filename in tqdm(target_files, desc="파일 로드 중"):
        file_path = os.path.join(processed_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_posts.extend(data)
            
    print(f"총 {len(all_posts)}개의 게시글 데이터를 로드했습니다.")
    return all_posts

def calculate_monthly_views_for_post(post, latest_month_str):
    """단일 게시글에 대해 월별 조회수를 추정하는 함수."""
    try:
        post_date_str = post.get('date')
        views = int(post.get('views', 0))
        post_date = datetime.strptime(post_date_str, '%Y.%m.%d')
    except (ValueError, TypeError):
        # 날짜나 조회수 형식이 잘못된 경우 건너뜀
        return {}

    post_month_str = post_date.strftime('%Y-%m')

    # 최신 월에 게시된 글은 모든 조회수를 해당 월에 할당
    if post_month_str == latest_month_str:
        return {post_month_str: views}

    # 3주 배분 모델 적용
    views_w1 = round(views * 0.5)
    views_w2 = round((views - views_w1) * 0.5)
    views_w3 = views - views_w1 - views_w2

    daily_views_w1 = views_w1 / 7 if views_w1 > 0 else 0
    daily_views_w2 = views_w2 / 7 if views_w2 > 0 else 0
    daily_views_w3 = views_w3 / 7 if views_w3 > 0 else 0
    
    monthly_distribution = defaultdict(float)

    for i in range(21): # 3주 = 21일
        current_day = post_date + timedelta(days=i)
        current_month_str = current_day.strftime('%Y-%m')
        
        if 0 <= i < 7:
            daily_views = daily_views_w1
        elif 7 <= i < 14:
            daily_views = daily_views_w2
        else:
            daily_views = daily_views_w3
        
        monthly_distribution[current_month_str] += daily_views

    # 최종 결과를 정수로 반올림
    return {month: round(count) for month, count in monthly_distribution.items()}

def main():
    """메인 실행 함수: 데이터 로드 및 월별 조회수 추정."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cafe_analysis_dir = os.path.dirname(script_dir)
    processed_dir = os.path.join(cafe_analysis_dir, 'data', 'historical_processed')
    output_dir = script_dir

    all_posts = load_all_processed_data(processed_dir)
    if not all_posts:
        return

    # 데이터에서 가장 최신 월 찾기
    latest_date = max(datetime.strptime(p['date'], '%Y.%m.%d') for p in all_posts if p.get('date'))
    latest_month_str = latest_date.strftime('%Y-%m')
    print(f"데이터의 최신 월: {latest_month_str}")

    final_results = []
    for post in tqdm(all_posts, desc="월별 조회수 추정 중"):
        monthly_views = calculate_monthly_views_for_post(post, latest_month_str)
        
        # 기본 정보와 월별 조회수 정보를 합침
        result_row = {
            'club_id': post.get('club_id'),
            'article_id': post.get('article_id')
        }
        result_row.update(monthly_views)
        final_results.append(result_row)
        
    # Pandas DataFrame으로 변환하여 CSV로 저장
    df = pd.DataFrame(final_results)
    
    # club_id, article_id를 제외한 월 컬럼들을 정렬
    month_columns = sorted([col for col in df.columns if col not in ['club_id', 'article_id']])
    final_columns = ['club_id', 'article_id'] + month_columns
    df = df[final_columns]

    # NaN 값을 0으로 채우고 정수형으로 변환
    for col in month_columns:
        df[col] = df[col].fillna(0).astype(int)

    output_path = os.path.join(output_dir, 'monthly_view_estimates.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n===== 월별 조회수 추정 완료. 결과 저장: {output_path} =====")


if __name__ == "__main__":
    main() 