import os
import json
from collections import defaultdict

def find_plan_clinic_2025_01_recovered():
    """
    복구된 데이터로 "플란치과의원 경기 동탄점"이 언급된 2025년 1월 게시글을 찾습니다.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(script_dir, '..', 'data', 'historical_processed')
    
    TARGET_CLINIC = "플란치과의원 경기 동탄점"
    TARGET_MONTH = "2025.01"
    
    print(f"=== {TARGET_CLINIC} 언급 게시글 검색 (2025년 1월) ===")
    print(f"대상 병원: {TARGET_CLINIC}")
    print(f"대상 월: {TARGET_MONTH}")
    
    if not os.path.exists(processed_dir):
        print("processed 폴더를 찾을 수 없습니다.")
        return
    
    found_posts = []
    monthly_views = defaultdict(int)
    
    for filename in os.listdir(processed_dir):
        if filename.endswith('_analyzed.json'):
            file_path = os.path.join(processed_dir, filename)
            
            try:
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
                        
                        if TARGET_CLINIC in all_clinics:
                            views = post.get('views')
                            date = post.get('date', '')
                            
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
                                
                                # 월별 조회수 집계
                                month_key = f"{TARGET_MONTH}"
                                monthly_views[month_key] += views
                                
            except Exception as e:
                print(f"파일 읽기 오류 ({filename}): {e}")
    
    if found_posts:
        # 조회수 순으로 정렬
        found_posts.sort(key=lambda x: x['views'], reverse=True)
        
        print(f"\n총 {len(found_posts)}개 게시글 발견!")
        print(f"2025년 1월 총 조회수: {monthly_views.get('2025.01', 0):,}")
        
        print(f"\n=== 조회수 TOP 5 ===")
        for i, post in enumerate(found_posts[:5], 1):
            print(f"{i}. 제목: {post['title']}")
            print(f"   게시글 ID: {post['article_id']}")
            print(f"   날짜: {post['date']}")
            print(f"   조회수: {post['views']:,}")
            print(f"   카페: {post['cafe_name']} (club_id: {post['club_id']})")
            print(f"   파일: {post['filename']}")
            print()
        
        if len(found_posts) > 5:
            print(f"... 외 {len(found_posts) - 5}개 게시글")
            
    else:
        print(f"\n'{TARGET_CLINIC}'이 언급된 2025년 1월 게시글을 찾을 수 없습니다.")
        
        # 유사한 병원명 검색
        print(f"\n=== 유사한 병원명 검색 ===")
        similar_clinics = []
        
        for filename in os.listdir(processed_dir):
            if filename.endswith('_analyzed.json'):
                file_path = os.path.join(processed_dir, filename)
                
                try:
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
                                if "플란" in clinic or "동탄" in clinic:
                                    if clinic not in similar_clinics:
                                        similar_clinics.append(clinic)
                                        
                except Exception as e:
                    continue
        
        if similar_clinics:
            print("유사한 병원명들:")
            for clinic in sorted(similar_clinics):
                print(f"  - {clinic}")
        else:
            print("유사한 병원명을 찾을 수 없습니다.")

if __name__ == "__main__":
    find_plan_clinic_2025_01_recovered() 