import os
import json

def find_plan_clinic_any():
    """
    "플란치과의원 경기 동탄점"이 언급된 모든 게시글을 찾습니다.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(script_dir, '..', 'data', 'historical_processed')
    
    TARGET_CLINIC = "플란치과의원 경기 동탄점"
    
    print(f"=== {TARGET_CLINIC} 언급 게시글 검색 ===")
    
    if not os.path.exists(processed_dir):
        print("processed 폴더를 찾을 수 없습니다.")
        return
    
    found_posts = []
    
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
                        
                        # 부분 매칭으로 검색
                        for clinic in all_clinics:
                            if "플란치과의원" in clinic and "동탄" in clinic:
                                found_posts.append({
                                    'article_id': post.get('article_id'),
                                    'title': post.get('title'),
                                    'date': post.get('date', ''),
                                    'views': post.get('views'),
                                    'club_id': post.get('club_id'),
                                    'cafe_name': post.get('cafe_name'),
                                    'filename': filename,
                                    'matched_clinic': clinic,
                                    'all_clinics': all_clinics
                                })
                                break
                                
            except Exception as e:
                print(f"파일 읽기 오류 ({filename}): {e}")
    
    if found_posts:
        # 조회수 순으로 정렬
        found_posts.sort(key=lambda x: x['views'] if x['views'] else 0, reverse=True)
        
        print(f"\n총 {len(found_posts)}개 게시글 발견!")
        
        print(f"\n=== 발견된 게시글 ===")
        for i, post in enumerate(found_posts, 1):
            print(f"{i}. 제목: {post['title']}")
            print(f"   게시글 ID: {post['article_id']}")
            print(f"   날짜: {post['date']}")
            print(f"   조회수: {post['views']:,}" if post['views'] else "   조회수: None")
            print(f"   카페: {post['cafe_name']} (club_id: {post['club_id']})")
            print(f"   파일: {post['filename']}")
            print(f"   매칭된 병원명: '{post['matched_clinic']}'")
            print()
            
    else:
        print(f"\n'{TARGET_CLINIC}'이 언급된 게시글을 찾을 수 없습니다.")
        
        # 모든 "플란" 관련 병원명 출력
        print(f"\n=== 모든 '플란' 관련 병원명 ===")
        all_plan_clinics = set()
        
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
                            
                            for clinic in all_clinics_in_post:
                                if "플란" in clinic:
                                    all_plan_clinics.add(clinic)
                                    
                except Exception as e:
                    continue
        
        if all_plan_clinics:
            for clinic in sorted(all_plan_clinics):
                print(f"  - '{clinic}'")
        else:
            print("'플란'이 포함된 병원명을 찾을 수 없습니다.")

if __name__ == "__main__":
    find_plan_clinic_any() 