import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
from wordcloud import WordCloud
import numpy as np
from konlpy.tag import Okt
import matplotlib.font_manager as fm

def analyze_keywords_wordcloud():
    """
    "신경치료" 관련 게시글에서 main keywords를 추출하고 wordcloud를 생성합니다.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(script_dir, '..', 'data', 'historical_processed')
    output_dir = os.path.join(script_dir, 'keyword_analysis')
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    TARGET_TREATMENT = "신경치료"
    
    print(f"=== {TARGET_TREATMENT} 관련 키워드 분석 ===")
    
    if not os.path.exists(processed_dir):
        print("processed 폴더를 찾을 수 없습니다.")
        return
    
    # 데이터 수집
    all_keywords = []
    keyword_posts = []
    
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
                            main_keywords = analysis.get('main_keywords', [])
                            
                            if main_keywords:
                                post_info = {
                                    'article_id': post.get('article_id'),
                                    'title': post.get('title'),
                                    'date': post.get('date', ''),
                                    'views': post.get('views', 0) if post.get('views') else 0,
                                    'club_id': post.get('club_id'),
                                    'cafe_name': post.get('cafe_name'),
                                    'filename': filename,
                                    'keywords': main_keywords
                                }
                                keyword_posts.append(post_info)
                                all_keywords.extend(main_keywords)
                                
            except Exception as e:
                print(f"파일 읽기 오류 ({filename}): {e}")
    
    if not keyword_posts:
        print(f"'{TARGET_TREATMENT}'이 언급된 게시글을 찾을 수 없습니다.")
        return
    
    print(f"\n총 {len(keyword_posts)}개 게시글에서 키워드 추출")
    print(f"총 {len(all_keywords)}개 키워드 수집")
    
    # 키워드 전처리 및 NLP 처리
    processed_keywords = preprocess_keywords(all_keywords)
    
    # 키워드 빈도 분석
    keyword_freq = Counter(processed_keywords)
    
    print(f"\n=== TOP 20 키워드 ===")
    for keyword, count in keyword_freq.most_common(20):
        print(f"{keyword}: {count}회")
    
    # CSV 저장
    keyword_df = pd.DataFrame(keyword_freq.most_common(), columns=['keyword', 'frequency'])
    output_csv = os.path.join(output_dir, f'{TARGET_TREATMENT}_keywords_analysis.csv')
    keyword_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n키워드 분석 결과 저장: {output_csv}")
    
    # WordCloud 생성
    create_wordcloud(keyword_freq, output_dir, TARGET_TREATMENT)
    
    # 추가 분석: 키워드별 게시글 수
    analyze_keyword_posts(keyword_posts, output_dir, TARGET_TREATMENT)
    
    return keyword_freq

def preprocess_keywords(keywords):
    """
    키워드 전처리 및 NLP 처리
    """
    processed = []
    
    # 불용어 정의
    stopwords = {
        '치과', '병원', '의원', '치료', '진료', '상담', '예약', '방문', '진찰',
        '검사', '진단', '수술', '시술', '치료법', '방법', '과정', '단계',
        '결과', '효과', '개선', '완치', '치유', '회복', '증상', '통증',
        '아픔', '불편', '문제', '이상', '증상', '상태', '경우', '때',
        '이유', '원인', '결과', '후', '전', '중', '간', '때문', '때',
        '것', '수', '것', '등', '등등', '그', '이', '저', '우리', '저희',
        '너희', '그들', '이들', '저들', '우리들', '저희들', '너희들',
        '그것', '이것', '저것', '무엇', '어떤', '어떤', '어떻게', '언제',
        '어디', '누가', '무엇을', '어떤', '어떻게', '언제', '어디', '누가',
        '무엇을', '어떤', '어떻게', '언제', '어디', '누가', '무엇을'
    }
    
    for keyword in keywords:
        if not keyword or len(keyword.strip()) < 2:
            continue
            
        # 기본 전처리
        keyword = keyword.strip().lower()
        
        # 특수문자 제거 (하지만 한글은 유지)
        keyword = re.sub(r'[^\w\s가-힣]', '', keyword)
        
        # 숫자 제거
        keyword = re.sub(r'\d+', '', keyword)
        
        # 공백 정리
        keyword = keyword.strip()
        
        # 길이 체크
        if len(keyword) < 2:
            continue
            
        # 불용어 체크
        if keyword in stopwords:
            continue
            
        # 의미있는 키워드만 추가
        if is_meaningful_keyword(keyword):
            processed.append(keyword)
    
    return processed

def is_meaningful_keyword(keyword):
    """
    의미있는 키워드인지 판단
    """
    # 너무 짧거나 긴 키워드 제외
    if len(keyword) < 2 or len(keyword) > 20:
        return False
    
    # 특정 패턴 제외
    patterns_to_exclude = [
        r'^[0-9]+$',  # 숫자만
        r'^[a-zA-Z]+$',  # 영문만
        r'^[가-힣]{1}$',  # 한글 한 글자
        r'^[가-힣]{2}$',  # 한글 두 글자 (의미없는 것들)
    ]
    
    for pattern in patterns_to_exclude:
        if re.match(pattern, keyword):
            return False
    
    return True

def create_wordcloud(keyword_freq, output_dir, treatment):
    """
    WordCloud 생성
    """
    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    # WordCloud 생성
    wordcloud = WordCloud(
        font_path='/System/Library/Fonts/AppleGothic.ttf',  # macOS 한글 폰트
        width=1200,
        height=800,
        background_color='white',
        max_words=100,
        max_font_size=100,
        random_state=42,
        colormap='viridis'
    )
    
    # 빈도수 딕셔너리 생성
    word_freq_dict = dict(keyword_freq)
    
    # WordCloud 생성
    wordcloud.generate_from_frequencies(word_freq_dict)
    
    # 시각화
    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'{treatment} 관련 키워드 WordCloud', fontsize=20, fontweight='bold', pad=20)
    
    # 저장
    output_wordcloud = os.path.join(output_dir, f'{treatment}_keywords_wordcloud.png')
    plt.savefig(output_wordcloud, dpi=300, bbox_inches='tight')
    print(f"WordCloud 저장: {output_wordcloud}")
    
    plt.show()

def analyze_keyword_posts(keyword_posts, output_dir, treatment):
    """
    키워드별 게시글 분석
    """
    # 키워드별 게시글 수 집계
    keyword_post_count = Counter()
    keyword_view_count = Counter()
    
    for post in keyword_posts:
        keywords = post['keywords']
        views = post['views']
        
        for keyword in keywords:
            keyword_post_count[keyword] += 1
            keyword_view_count[keyword] += views
    
    # 데이터프레임 생성
    keyword_analysis = []
    for keyword in keyword_post_count:
        keyword_analysis.append({
            'keyword': keyword,
            'post_count': keyword_post_count[keyword],
            'total_views': keyword_view_count[keyword],
            'avg_views': keyword_view_count[keyword] / keyword_post_count[keyword] if keyword_post_count[keyword] > 0 else 0
        })
    
    df = pd.DataFrame(keyword_analysis)
    df = df.sort_values('post_count', ascending=False)
    
    # CSV 저장
    output_csv = os.path.join(output_dir, f'{treatment}_keyword_post_analysis.csv')
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"키워드별 게시글 분석 저장: {output_csv}")
    
    # TOP 20 키워드 차트
    create_keyword_chart(df.head(20), output_dir, treatment)

def create_keyword_chart(df, output_dir, treatment):
    """
    키워드별 게시글 수 차트 생성
    """
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # 게시글 수 차트
    bars1 = ax1.bar(range(len(df)), df['post_count'], color='lightgreen', alpha=0.7)
    ax1.set_title(f'{treatment} 관련 키워드별 게시글 수 TOP 20', fontsize=16, fontweight='bold')
    ax1.set_ylabel('게시글 수', fontsize=12)
    
    # x축 라벨 설정
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df['keyword'], rotation=45, ha='right', fontsize=10)
    
    # 값 표시
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}개',
                ha='center', va='bottom', fontsize=9)
    
    # 평균 조회수 차트
    bars2 = ax2.bar(range(len(df)), df['avg_views'], color='lightblue', alpha=0.7)
    ax2.set_title(f'{treatment} 관련 키워드별 평균 조회수 TOP 20', fontsize=16, fontweight='bold')
    ax2.set_ylabel('평균 조회수', fontsize=12)
    
    # x축 라벨 설정
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels(df['keyword'], rotation=45, ha='right', fontsize=10)
    
    # 값 표시
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(height):,}회',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_chart = os.path.join(output_dir, f'{treatment}_keyword_analysis_chart.png')
    plt.savefig(output_chart, dpi=300, bbox_inches='tight')
    print(f"키워드 분석 차트 저장: {output_chart}")
    
    plt.show()

if __name__ == "__main__":
    analyze_keywords_wordcloud() 