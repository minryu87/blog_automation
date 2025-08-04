import pandas as pd
import os
import json
from collections import defaultdict

def check_plan_clinic_all_months():
    """
    플란치과의원 경기 동탄점이 언급된 모든 게시글을 월별로 확인합니다.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(script_dir, '..', 'data', 'historical_processed')
    
    TARGET_CLINIC = "플란치과의원 경기 동탄점"
    
    print(f"=== '{TARGET_CLINIC}' 언급 게시글 전체 검색 ===")
    
    # processed 데이터에서 해당 병원이 언급된 게시글 찾기
    if os.path.exists(processed_dir):
        found_posts = []
        monthly_posts = defaultdict(list)
        
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
                            
                            if views is not None:
                                found_posts.append({
                                    'article_id': post.get('article_id'),
                                    'title': post.get('title'),
                                    'date': date,
                                    'views': views,
                                    'club_id': post.get('club_id'),
                                    'cafe_name': post.get('cafe_name'),
                                    'filename': filename
                                })
                                
                                # 월별 분류
                                if date and '.' in date:
                                    try:
                                        parts = date.split('.')
                                        if len(parts) >= 2:
                                            year = parts[0]
                                            month = parts[1].zfill(2)
                                            month_key = f"{year}-{month}"
                                            monthly_posts[month_key].append({
                                                'title': post.get('title'),
                                                'date': date,
                                                'views': views,
                                                'cafe_name': post.get('cafe_name')
                                            })
                                    except:
                                        pass
        
        if found_posts:
            print(f"\n총 {len(found_posts)}개 게시글에서 '{TARGET_CLINIC}' 언급 발견")
            print("-" * 80)
            
            # 월별 요약
            print("=== 월별 분포 ===")
            for month in sorted(monthly_posts.keys()):
                posts = monthly_posts[month]
                total_views = sum(p['views'] for p in posts)
                print(f"{month}: {len(posts)}개 게시글, 총 {total_views:,} 조회수")
                
                # 해당 월의 게시글 목록
                for i, post in enumerate(posts, 1):
                    print(f"  {i}. {post['title']}")
                    print(f"     날짜: {post['date']}, 조회수: {post['views']:,}, 카페: {post['cafe_name']}")
                print()
            
            # 전체 게시글 목록 (조회수 순)
            print("=== 전체 게시글 목록 (조회수 순) ===")
            found_posts.sort(key=lambda x: x['views'], reverse=True)
            
            for i, post in enumerate(found_posts, 1):
                print(f"{i}. 제목: {post['title']}")
                print(f"   게시글 ID: {post['article_id']}")
                print(f"   날짜: {post['date']}")
                print(f"   조회수: {post['views']:,}")
                print(f"   카페: {post['cafe_name']}")
                print(f"   파일: {post['filename']}")
                print()
                
        else:
            print(f"'{TARGET_CLINIC}'이 언급된 게시글을 찾을 수 없습니다.")
            
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
    check_plan_clinic_all_months() 