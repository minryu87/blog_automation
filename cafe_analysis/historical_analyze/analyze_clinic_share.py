import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import defaultdict, Counter
import numpy as np

def analyze_clinic_share():
    """
    "신경치료"가 언급된 게시글에서 병원별 점유율을 분석합니다.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(script_dir, '..', 'data', 'historical_processed')
    output_dir = os.path.join(script_dir, 'clinic_analysis')
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    TARGET_TREATMENT = "신경치료"
    
    print(f"=== {TARGET_TREATMENT} 관련 병원별 점유율 분석 ===")
    
    if not os.path.exists(processed_dir):
        print("processed 폴더를 찾을 수 없습니다.")
        return
    
    # 데이터 수집
    clinic_posts = defaultdict(list)  # 병원별 게시글
    clinic_views = defaultdict(int)   # 병원별 총 조회수
    all_posts = []                   # 모든 관련 게시글
    
    for filename in os.listdir(processed_dir):
        if filename.endswith('_analyzed.json'):
            file_path = os.path.join(processed_dir, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    for post in data:
                        analysis = post.get('analysis', {})
                        related_treatments = analysis.get('related_treatments', [])
                        
                        # 신경치료가 언급된 게시글 찾기
                        if TARGET_TREATMENT in related_treatments:
                            mentioned_clinics = analysis.get('mentioned_clinics', [])
                            clinic_sentiments = analysis.get('clinic_sentiments', [])
                            
                            clinic_names = []
                            for sentiment in clinic_sentiments:
                                clinic_name = sentiment.get('clinic_name')
                                if clinic_name:
                                    clinic_names.append(clinic_name)
                            
                            all_clinics = mentioned_clinics + clinic_names
                            views = post.get('views', 0) if post.get('views') else 0
                            
                            post_info = {
                                'article_id': post.get('article_id'),
                                'title': post.get('title'),
                                'date': post.get('date', ''),
                                'views': views,
                                'club_id': post.get('club_id'),
                                'cafe_name': post.get('cafe_name'),
                                'filename': filename,
                                'clinics': all_clinics
                            }
                            
                            all_posts.append(post_info)
                            
                            # 각 병원별로 데이터 집계
                            for clinic in all_clinics:
                                clinic_posts[clinic].append(post_info)
                                clinic_views[clinic] += views
                                
            except Exception as e:
                print(f"파일 읽기 오류 ({filename}): {e}")
    
    if not all_posts:
        print(f"'{TARGET_TREATMENT}'이 언급된 게시글을 찾을 수 없습니다.")
        return
    
    print(f"\n총 {len(all_posts)}개 게시글에서 '{TARGET_TREATMENT}' 언급")
    print(f"총 {len(clinic_posts)}개 병원이 언급됨")
    
    # 데이터프레임 생성
    clinic_data = []
    for clinic, posts in clinic_posts.items():
        total_views = clinic_views[clinic]
        clinic_data.append({
            'clinic_name': clinic,
            'post_count': len(posts),
            'total_views': total_views,
            'avg_views': total_views / len(posts) if posts else 0
        })
    
    df = pd.DataFrame(clinic_data)
    df = df.sort_values('post_count', ascending=False)
    
    # 비율 계산
    total_posts = len(all_posts)
    total_views = sum(clinic_views.values())
    
    df['post_share'] = (df['post_count'] / total_posts * 100).round(2)
    df['views_share'] = (df['total_views'] / total_views * 100).round(2)
    
    print(f"\n=== 병원별 점유율 분석 결과 ===")
    print(f"총 게시글 수: {total_posts}")
    print(f"총 조회수: {total_views:,}")
    
    print(f"\n=== TOP 10 병원 (게시글 수 기준) ===")
    top_clinics = df.head(10)
    for idx, row in top_clinics.iterrows():
        print(f"{row['clinic_name']}: {row['post_count']}개 게시글 ({row['post_share']}%), {row['total_views']:,}회 조회 ({row['views_share']}%)")
    
    # CSV 저장
    output_csv = os.path.join(output_dir, f'{TARGET_TREATMENT}_clinic_analysis.csv')
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n분석 결과 저장: {output_csv}")
    
    # 시각화
    create_visualizations(df, output_dir, TARGET_TREATMENT)
    
    return df

def create_visualizations(df, output_dir, treatment):
    """
    시각화 그래프를 생성합니다.
    """
    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 게시글 수 기준 TOP 15 병원
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    top_15 = df.head(15)
    
    # 게시글 수 차트
    bars1 = ax1.bar(range(len(top_15)), top_15['post_count'], color='skyblue', alpha=0.7)
    ax1.set_title(f'{treatment} 관련 게시글 수 TOP 15 병원', fontsize=16, fontweight='bold')
    ax1.set_ylabel('게시글 수', fontsize=12)
    ax1.set_xlabel('병원명', fontsize=12)
    
    # x축 라벨 설정
    ax1.set_xticks(range(len(top_15)))
    ax1.set_xticklabels(top_15['clinic_name'], rotation=45, ha='right', fontsize=10)
    
    # 값 표시
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}개\n({top_15.iloc[i]["post_share"]}%)',
                ha='center', va='bottom', fontsize=9)
    
    # 조회수 차트
    bars2 = ax2.bar(range(len(top_15)), top_15['total_views'], color='lightcoral', alpha=0.7)
    ax2.set_title(f'{treatment} 관련 조회수 TOP 15 병원', fontsize=16, fontweight='bold')
    ax2.set_ylabel('총 조회수', fontsize=12)
    ax2.set_xlabel('병원명', fontsize=12)
    
    # x축 라벨 설정
    ax2.set_xticks(range(len(top_15)))
    ax2.set_xticklabels(top_15['clinic_name'], rotation=45, ha='right', fontsize=10)
    
    # 값 표시
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{int(height):,}회\n({top_15.iloc[i]["views_share"]}%)',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_chart = os.path.join(output_dir, f'{treatment}_clinic_analysis_chart.png')
    plt.savefig(output_chart, dpi=300, bbox_inches='tight')
    print(f"차트 저장: {output_chart}")
    
    # 2. 점유율 파이 차트
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 게시글 수 점유율 파이 차트
    top_10_posts = df.head(10)
    others_posts = df.iloc[10:]['post_count'].sum()
    post_data = list(top_10_posts['post_count']) + [others_posts]
    post_labels = list(top_10_posts['clinic_name']) + ['기타']
    
    ax1.pie(post_data, labels=post_labels, autopct='%1.1f%%', startangle=90)
    ax1.set_title(f'{treatment} 관련 게시글 수 점유율', fontsize=14, fontweight='bold')
    
    # 조회수 점유율 파이 차트
    top_10_views = df.head(10)
    others_views = df.iloc[10:]['total_views'].sum()
    views_data = list(top_10_views['total_views']) + [others_views]
    views_labels = list(top_10_views['clinic_name']) + ['기타']
    
    ax2.pie(views_data, labels=views_labels, autopct='%1.1f%%', startangle=90)
    ax2.set_title(f'{treatment} 관련 조회수 점유율', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_pie = os.path.join(output_dir, f'{treatment}_clinic_share_pie.png')
    plt.savefig(output_pie, dpi=300, bbox_inches='tight')
    print(f"파이 차트 저장: {output_pie}")
    
    plt.show()

if __name__ == "__main__":
    analyze_clinic_share() 