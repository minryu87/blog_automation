import os
import json
from collections import defaultdict

def find_plan_clinic_exact():
    """
    정확한 병원명 "플란치과의원 경기 동탄점"으로 검색합니다.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(script_dir, '..', 'data', 'historical_processed')
    
    TARGET_CLINIC = "플란치과의원 경기 동탄점"
    
    print(f"=== {TARGET_CLINIC} 정확한 검색 ===")
    
    if not os.path.exists(processed_dir):
        print("processed 폴더를 찾을 수 없습니다.")
        return
    
    found_posts = []
    all_clinics = set()
    
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
                        
                        all_clinics_in_post = mentioned_clinics + clinic_names
                        
                        # 모든 병원명 수집
                        for clinic in all_clinics_in_post:
                            all_clinics.add(clinic)
                        
                        # 정확한 병원명 검색
                        if TARGET_CLINIC in all_clinics_in_post:
                            views = post.get('views')
                            date = post.get('date', '')
                            
                            found_posts.append({
                                'article_id': post.get('article_id'),
                                'title': post.get('title'),
                                'date': date,
                                'views': views,
                                'club_id': post.get('club_id'),
                                'cafe_name': post.get('cafe_name'),
                                'filename': filename,
                                'mentioned_clinics': mentioned_clinics,
                                'clinic_sentiments': clinic_sentiments
                            })
                                
            except Exception as e:
                print(f"파일 읽기 오류 ({filename}): {e}")
    
    if found_posts:
        # 조회수 순으로 정렬
        found_posts.sort(key=lambda x: x['views'] if x['views'] else 0, reverse=True)
        
        print(f"\n총 {len(found_posts)}개 게시글 발견!")
        
        print(f"\n=== 모든 발견된 게시글 ===")
        for i, post in enumerate(found_posts, 1):
            print(f"{i}. 제목: {post['title']}")
            print(f"   게시글 ID: {post['article_id']}")
            print(f"   날짜: {post['date']}")
            print(f"   조회수: {post['views']:,}" if post['views'] else "   조회수: None")
            print(f"   카페: {post['cafe_name']} (club_id: {post['club_id']})")
            print(f"   파일: {post['filename']}")
            print(f"   언급된 병원: {post['mentioned_clinics']}")
            print(f"   병원 감정: {[s.get('clinic_name') for s in post['clinic_sentiments']]}")
            print()
            
    else:
        print(f"\n'{TARGET_CLINIC}'이 언급된 게시글을 찾을 수 없습니다.")
        
        # 유사한 병원명들 출력
        similar_clinics = [clinic for clinic in all_clinics if "플란" in clinic or "동탄" in clinic]
        
        if similar_clinics:
            print(f"\n=== 유사한 병원명들 ===")
            for clinic in sorted(similar_clinics):
                print(f"  - {clinic}")
        else:
            print("유사한 병원명을 찾을 수 없습니다.")

if __name__ == "__main__":
    find_plan_clinic_exact() 