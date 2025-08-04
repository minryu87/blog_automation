import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from datetime import datetime, timedelta
import matplotlib.dates as mdates

def analyze_negative_clinics_cumulative():
    """
    부정적 감정을 가진 병원들의 월별 누적 조회수를 분석합니다.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(script_dir, '..', 'data', 'historical_processed')
    output_dir = os.path.join(script_dir, 'negative_clinics_analysis')
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== 부정적 감정 병원 누적 조회수 분석 ===")
    
    if not os.path.exists(processed_dir):
        print("processed 폴더를 찾을 수 없습니다.")
        return
    
    # 데이터 수집
    negative_clinics = defaultdict(list)  # 병원별 부정 게시글
    all_negative_posts = []              # 모든 부정 게시글
    
    for filename in os.listdir(processed_dir):
        if filename.endswith('_analyzed.json'):
            file_path = os.path.join(processed_dir, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    for post in data:
                        analysis = post.get('analysis', {})
                        clinic_sentiments = analysis.get('clinic_sentiments', [])
                        
                        # 부정적 감정이 있는 게시글 찾기
                        negative_sentiments = []
                        for sentiment in clinic_sentiments:
                            sentiment_type = sentiment.get('sentiment', '').lower()
                            if sentiment_type in ['negative', '부정', '나쁨', '안좋음']:
                                clinic_name = sentiment.get('clinic_name')
                                if clinic_name:
                                    negative_sentiments.append(clinic_name)
                        
                        if negative_sentiments:
                            views = post.get('views', 0) if post.get('views') else 0
                            date_str = post.get('date', '')
                            
                            post_info = {
                                'article_id': post.get('article_id'),
                                'title': post.get('title'),
                                'date': date_str,
                                'views': views,
                                'club_id': post.get('club_id'),
                                'cafe_name': post.get('cafe_name'),
                                'filename': filename,
                                'negative_clinics': negative_sentiments
                            }
                            
                            all_negative_posts.append(post_info)
                            
                            # 각 부정적 병원별로 데이터 수집
                            for clinic in negative_sentiments:
                                negative_clinics[clinic].append(post_info)
                                
            except Exception as e:
                print(f"파일 읽기 오류 ({filename}): {e}")
    
    if not all_negative_posts:
        print("부정적 감정이 있는 게시글을 찾을 수 없습니다.")
        return
    
    print(f"\n총 {len(all_negative_posts)}개 부정 게시글 발견")
    print(f"총 {len(negative_clinics)}개 병원에서 부정적 감정 발견")
    
    # 부정 게시글 수로 정렬
    clinic_negative_counts = {clinic: len(posts) for clinic, posts in negative_clinics.items()}
    sorted_clinics = sorted(clinic_negative_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n=== 부정 게시글 수 TOP 10 병원 ===")
    for clinic, count in sorted_clinics[:10]:
        total_views = sum(post['views'] for post in negative_clinics[clinic])
        print(f"{clinic}: {count}개 게시글, 총 {total_views:,}회 조회")
    
    # 누적 그래프 생성
    create_cumulative_charts(negative_clinics, output_dir)
    
    # CSV 저장
    save_negative_analysis(negative_clinics, output_dir)
    
    return negative_clinics

def create_cumulative_charts(negative_clinics, output_dir):
    """
    누적 조회수 차트를 생성합니다.
    """
    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    # TOP 10 병원만 선택
    top_clinics = sorted(negative_clinics.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    
    # 전체 기간 설정 (2024-06 ~ 2025-07)
    start_date = datetime(2024, 6, 1)
    end_date = datetime(2025, 7, 31)
    
    # 월별 날짜 리스트 생성
    date_range = []
    current_date = start_date
    while current_date <= end_date:
        date_range.append(current_date)
        current_date = (current_date.replace(day=1) + timedelta(days=32)).replace(day=1)
    
    # 1. 개별 병원별 누적 그래프
    fig, axes = plt.subplots(2, 5, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, (clinic, posts) in enumerate(top_clinics):
        if i >= 10:
            break
            
        ax = axes[i]
        
        # 월별 누적 조회수 계산
        cumulative_views = []
        current_cumulative = 0
        
        for date in date_range:
            month_key = date.strftime('%Y-%m')
            
            # 해당 월의 게시글들 찾기
            month_posts = [post for post in posts if post['date'].startswith(month_key)]
            month_views = sum(post['views'] for post in month_posts)
            
            current_cumulative += month_views
            cumulative_views.append(current_cumulative)
        
        # 그래프 그리기
        ax.plot(date_range, cumulative_views, marker='o', linewidth=2, markersize=4)
        ax.set_title(f'{clinic}\n(총 {len(posts)}개 게시글)', fontsize=10, fontweight='bold')
        ax.set_ylabel('누적 조회수', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # x축 포맷 설정
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 최종 누적값 표시
        if cumulative_views:
            ax.text(0.02, 0.98, f'총 {cumulative_views[-1]:,}회', 
                   transform=ax.transAxes, fontsize=8, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    output_chart1 = os.path.join(output_dir, 'negative_clinics_individual_cumulative.png')
    plt.savefig(output_chart1, dpi=300, bbox_inches='tight')
    print(f"개별 병원 누적 차트 저장: {output_chart1}")
    
    # 2. 전체 병원 누적 비교 그래프
    plt.figure(figsize=(15, 10))
    
    for clinic, posts in top_clinics[:5]:  # TOP 5만 표시
        cumulative_views = []
        current_cumulative = 0
        
        for date in date_range:
            month_key = date.strftime('%Y-%m')
            month_posts = [post for post in posts if post['date'].startswith(month_key)]
            month_views = sum(post['views'] for post in month_posts)
            current_cumulative += month_views
            cumulative_views.append(current_cumulative)
        
        plt.plot(date_range, cumulative_views, marker='o', linewidth=2, label=clinic)
    
    plt.title('부정적 감정 병원별 누적 조회수 비교 (TOP 5)', fontsize=16, fontweight='bold')
    plt.xlabel('날짜', fontsize=12)
    plt.ylabel('누적 조회수', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # x축 포맷 설정
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    output_chart2 = os.path.join(output_dir, 'negative_clinics_comparison_cumulative.png')
    plt.savefig(output_chart2, dpi=300, bbox_inches='tight')
    print(f"병원 비교 누적 차트 저장: {output_chart2}")
    
    plt.show()

def save_negative_analysis(negative_clinics, output_dir):
    """
    부정적 감정 분석 결과를 CSV로 저장합니다.
    """
    # 병원별 부정 게시글 분석
    clinic_analysis = []
    
    for clinic, posts in negative_clinics.items():
        total_posts = len(posts)
        total_views = sum(post['views'] for post in posts)
        avg_views = total_views / total_posts if total_posts > 0 else 0
        
        # 월별 분포
        monthly_posts = defaultdict(int)
        monthly_views = defaultdict(int)
        
        for post in posts:
            date_str = post['date']
            if date_str and '.' in date_str:
                try:
                    parts = date_str.split('.')
                    if len(parts) >= 2:
                        year = parts[0]
                        month = parts[1].zfill(2)
                        month_key = f"{year}-{month}"
                        monthly_posts[month_key] += 1
                        monthly_views[month_key] += post['views']
                except:
                    pass
        
        clinic_analysis.append({
            'clinic_name': clinic,
            'negative_post_count': total_posts,
            'total_negative_views': total_views,
            'avg_negative_views': avg_views,
            'monthly_distribution': dict(monthly_posts),
            'monthly_views': dict(monthly_views)
        })
    
    # 정렬
    clinic_analysis.sort(key=lambda x: x['negative_post_count'], reverse=True)
    
    # CSV 저장
    df = pd.DataFrame(clinic_analysis)
    output_csv = os.path.join(output_dir, 'negative_clinics_analysis.csv')
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"부정적 감정 분석 저장: {output_csv}")
    
    # 상세 게시글 정보 저장
    detailed_posts = []
    for clinic, posts in negative_clinics.items():
        for post in posts:
            detailed_posts.append({
                'clinic_name': clinic,
                'article_id': post['article_id'],
                'title': post['title'],
                'date': post['date'],
                'views': post['views'],
                'club_id': post['club_id'],
                'cafe_name': post['cafe_name']
            })
    
    detailed_df = pd.DataFrame(detailed_posts)
    detailed_csv = os.path.join(output_dir, 'negative_clinics_detailed_posts.csv')
    detailed_df.to_csv(detailed_csv, index=False, encoding='utf-8-sig')
    print(f"부정 게시글 상세 정보 저장: {detailed_csv}")

if __name__ == "__main__":
    analyze_negative_clinics_cumulative() 