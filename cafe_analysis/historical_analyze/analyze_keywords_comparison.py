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

def analyze_keywords_comparison():
    """
    전체 게시글과 신경치료 특화 키워드를 비교하여 두 가지 wordcloud를 생성합니다.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(script_dir, '..', 'data', 'historical_processed')
    output_dir = os.path.join(script_dir, 'keyword_comparison')
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    TARGET_TREATMENT = "신경치료"
    
    print(f"=== 키워드 비교 분석 ===")
    
    if not os.path.exists(processed_dir):
        print("processed 폴더를 찾을 수 없습니다.")
        return
    
    # 데이터 수집
    all_keywords = []           # 전체 키워드
    nerve_keywords = []         # 신경치료 관련 키워드
    all_posts = []             # 전체 게시글
    nerve_posts = []           # 신경치료 관련 게시글
    
    for filename in os.listdir(processed_dir):
        if filename.endswith('_analyzed.json'):
            file_path = os.path.join(processed_dir, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    for post in data:
                        analysis = post.get('analysis', {})
                        main_keywords = analysis.get('main_keywords', [])
                        related_treatments = analysis.get('related_treatments', [])
                        
                        if main_keywords:
                            post_info = {
                                'article_id': post.get('article_id'),
                                'title': post.get('title'),
                                'date': post.get('date', ''),
                                'views': post.get('views', 0) if post.get('views') else 0,
                                'club_id': post.get('club_id'),
                                'cafe_name': post.get('cafe_name'),
                                'filename': filename,
                                'keywords': main_keywords,
                                'related_treatments': related_treatments
                            }
                            
                            all_posts.append(post_info)
                            all_keywords.extend(main_keywords)
                            
                            # 신경치료 관련 게시글 분리
                            if TARGET_TREATMENT in related_treatments:
                                nerve_posts.append(post_info)
                                nerve_keywords.extend(main_keywords)
                                
            except Exception as e:
                print(f"파일 읽기 오류 ({filename}): {e}")
    
    if not all_posts:
        print("게시글을 찾을 수 없습니다.")
        return
    
    print(f"\n총 {len(all_posts)}개 게시글에서 키워드 추출")
    print(f"총 {len(all_keywords)}개 키워드 수집")
    print(f"신경치료 관련 {len(nerve_posts)}개 게시글")
    print(f"신경치료 관련 {len(nerve_keywords)}개 키워드")
    
    # 키워드 전처리
    all_processed = preprocess_keywords(all_keywords)
    nerve_processed = preprocess_keywords(nerve_keywords)
    
    # 빈도 분석
    all_freq = Counter(all_processed)
    nerve_freq = Counter(nerve_processed)
    
    print(f"\n=== 전체 키워드 TOP 10 ===")
    for keyword, count in all_freq.most_common(10):
        print(f"{keyword}: {count}회")
    
    print(f"\n=== 신경치료 키워드 TOP 10 ===")
    for keyword, count in nerve_freq.most_common(10):
        print(f"{keyword}: {count}회")
    
    # 신경치료 특화 키워드 찾기
    specialized_keywords = find_specialized_keywords(all_freq, nerve_freq, len(all_posts), len(nerve_posts))
    
    print(f"\n=== 신경치료 특화 키워드 TOP 20 ===")
    for keyword, score in specialized_keywords[:20]:
        print(f"{keyword}: 특화점수 {score:.2f}")
    
    # WordCloud 생성
    create_comparison_wordclouds(all_freq, nerve_freq, specialized_keywords, output_dir, TARGET_TREATMENT)
    
    # CSV 저장
    save_analysis_results(all_freq, nerve_freq, specialized_keywords, output_dir, TARGET_TREATMENT)
    
    return all_freq, nerve_freq, specialized_keywords

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

def find_specialized_keywords(all_freq, nerve_freq, total_posts, nerve_posts):
    """
    신경치료 특화 키워드를 찾습니다.
    """
    specialized = []
    
    for keyword in nerve_freq:
        if keyword in all_freq:
            # 전체에서의 비율
            all_ratio = all_freq[keyword] / total_posts
            # 신경치료에서의 비율
            nerve_ratio = nerve_freq[keyword] / nerve_posts
            
            # 특화 점수 계산 (비율의 차이)
            specialization_score = nerve_ratio - all_ratio
            
            # 최소 빈도 조건 (신경치료에서 2회 이상)
            if nerve_freq[keyword] >= 2 and specialization_score > 0:
                specialized.append((keyword, specialization_score))
    
    # 특화 점수 순으로 정렬
    specialized.sort(key=lambda x: x[1], reverse=True)
    
    return specialized

def create_comparison_wordclouds(all_freq, nerve_freq, specialized_keywords, output_dir, treatment):
    """
    비교 wordcloud들을 생성합니다.
    """
    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 전체 키워드 WordCloud
    create_wordcloud(all_freq, output_dir, f"{treatment}_all_keywords", "전체 게시글 키워드")
    
    # 2. 신경치료 키워드 WordCloud
    create_wordcloud(nerve_freq, output_dir, f"{treatment}_nerve_keywords", "신경치료 관련 키워드")
    
    # 3. 신경치료 특화 키워드 WordCloud
    specialized_dict = dict(specialized_keywords[:50])  # TOP 50만 사용
    create_wordcloud(specialized_dict, output_dir, f"{treatment}_specialized_keywords", "신경치료 특화 키워드")

def create_wordcloud(keyword_freq, output_dir, filename, title):
    """
    WordCloud 생성
    """
    if not keyword_freq:
        print(f"키워드가 없어서 {filename} WordCloud를 생성할 수 없습니다.")
        return
    
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
    plt.title(f'{title} WordCloud', fontsize=20, fontweight='bold', pad=20)
    
    # 저장
    output_wordcloud = os.path.join(output_dir, f'{filename}_wordcloud.png')
    plt.savefig(output_wordcloud, dpi=300, bbox_inches='tight')
    print(f"WordCloud 저장: {output_wordcloud}")
    
    plt.show()

def save_analysis_results(all_freq, nerve_freq, specialized_keywords, output_dir, treatment):
    """
    분석 결과를 CSV로 저장합니다.
    """
    # 전체 키워드 분석
    all_df = pd.DataFrame(all_freq.most_common(), columns=['keyword', 'frequency'])
    all_csv = os.path.join(output_dir, f'{treatment}_all_keywords_analysis.csv')
    all_df.to_csv(all_csv, index=False, encoding='utf-8-sig')
    print(f"전체 키워드 분석 저장: {all_csv}")
    
    # 신경치료 키워드 분석
    nerve_df = pd.DataFrame(nerve_freq.most_common(), columns=['keyword', 'frequency'])
    nerve_csv = os.path.join(output_dir, f'{treatment}_nerve_keywords_analysis.csv')
    nerve_df.to_csv(nerve_csv, index=False, encoding='utf-8-sig')
    print(f"신경치료 키워드 분석 저장: {nerve_csv}")
    
    # 특화 키워드 분석
    specialized_df = pd.DataFrame(specialized_keywords, columns=['keyword', 'specialization_score'])
    specialized_csv = os.path.join(output_dir, f'{treatment}_specialized_keywords_analysis.csv')
    specialized_df.to_csv(specialized_csv, index=False, encoding='utf-8-sig')
    print(f"특화 키워드 분석 저장: {specialized_csv}")

if __name__ == "__main__":
    analyze_keywords_comparison() 