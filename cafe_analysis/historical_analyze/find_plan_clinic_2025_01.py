import pandas as pd
import os
import json
from collections import defaultdict

def find_plan_clinic_2025_01():
    """
    플란치과의원 경기 동탄점이 언급된 2025년 1월 게시글 중 조회수 상위 5개를 찾습니다.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(script_dir, '..', 'data', 'historical_processed')
    
    TARGET_CLINIC = "플란치과의원 경기 동탄점"
    TARGET_MONTH = "2025.01"
    
    print(f"=== '{TARGET_CLINIC}' 언급 2025년 1월 게시글 조회수 TOP 5 ===")
    
    # processed 데이터에서 해당 병원이 언급된 게시글 찾기
    if os.path.exists(processed_dir):
        found_posts = []
        
        for filename in os.listdir(processed_dir):
            if filename.endswith('_analyzed.json'):
                file_path = os.path.join(processed_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    for post in data:
                        analysis = post.get('analysis', {})
                        
                        # mentioned_clinics에서 확인
                        mentioned_clinics = analysis.get('mentioned_clinics', [])
                        clinic_sentiments = analysis.get('clinic_sentiments', [])
                        
                        # clinic_sentiments에서도 확인
                        clinic_names = []
                        for sentiment in clinic_sentiments:
                            clinic_name = sentiment.get('clinic_name')
                            if clinic_name:
                                clinic_names.append(clinic_name)
                        
                        # 모든 병원명을 합쳐서 검색
                        all_clinics = mentioned_clinics + clinic_names
                        
                        if TARGET_CLINIC in all_clinics:
                            views = post.get('views')
                            date = post.get('date', '')
                            
                            # 2025년 1월 게시글만 필터링
                            if date.startswith(TARGET_MONTH) and views is not None:
                                found_posts.append({
                                    'article_id': post.get('article_id'),
                                    'title': post.get('title'),
                                    'date': date,
                                    'views': views,
                                    'club_id': post.get('club_id'),
                                    'cafe_name': post.get('cafe_name'),
                                    'filename': filename
                                })
        
        if found_posts:
            # 조회수 기준으로 정렬 (내림차순)
            found_posts.sort(key=lambda x: x['views'], reverse=True)
            
            print(f"\n총 {len(found_posts)}개 게시글에서 '{TARGET_CLINIC}' 언급 발견")
            print("-" * 80)
            
            print("카페명|제목|게시일자|조회수")
            print("-" * 80)
            
            # 상위 5개 출력
            for i, post in enumerate(found_posts[:5], 1):
                cafe_name = post['cafe_name']
                title = post['title']
                date = post['date']
                views = post['views']
                
                print(f"{cafe_name}|{title}|{date}|{views:,}")
            
            # 추가 정보
            print(f"\n=== 추가 정보 ===")
            print(f"전체 게시글 수: {len(found_posts)}개")
            if found_posts:
                print(f"조회수 범위: {min(p['views'] for p in found_posts)} ~ {max(p['views'] for p in found_posts)}")
                print(f"평균 조회수: {sum(p['views'] for p in found_posts) / len(found_posts):.1f}")
                
        else:
            print(f"'{TARGET_CLINIC}'이 언급된 2025년 1월 게시글을 찾을 수 없습니다.")
            
            # 비슷한 이름 검색
            print(f"\n=== 비슷한 이름 검색 ===")
            similar_clinics = []
            for filename in os.listdir(processed_dir):
                if filename.endswith('_analyzed.json'):
                    file_path = os.path.join(processed_dir, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        for post in data:
                            analysis = post.get('analysis', {})
                            mentioned_clinics = analysis.get('mentioned_clinics', [])
                            clinic_sentiments = analysis.get('clinic_sentiments', [])
                            
                            clinic_names = []
                            for sentiment in clinic_sentiments:
                                clinic_name = sentiment.get('clinic_name')
                                if clinic_name:
                                    clinic_names.append(clinic_name)
                            
                            all_clinics = mentioned_clinics + clinic_names
                            
                            for clinic in all_clinics:
                                if "플란" in clinic or "플란치과" in clinic:
                                    if clinic not in similar_clinics:
                                        similar_clinics.append(clinic)
            
            if similar_clinics:
                print("플란 관련 병원명:")
                for clinic in similar_clinics:
                    print(f"  - {clinic}")
            else:
                print("플란 관련 병원명을 찾을 수 없습니다.")
    else:
        print("processed 폴더를 찾을 수 없습니다.")

if __name__ == "__main__":
    find_plan_clinic_2025_01() 