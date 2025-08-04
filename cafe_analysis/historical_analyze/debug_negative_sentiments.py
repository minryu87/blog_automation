import os
import json

def debug_negative_sentiments():
    """
    부정적 감정 데이터 구조를 디버깅합니다.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(script_dir, '..', 'data', 'historical_processed')
    
    print("=== 부정적 감정 데이터 구조 디버깅 ===")
    
    if not os.path.exists(processed_dir):
        print("processed 폴더를 찾을 수 없습니다.")
        return
    
    found_negative = False
    sample_count = 0
    
    for filename in os.listdir(processed_dir):
        if filename.endswith('_analyzed.json'):
            file_path = os.path.join(processed_dir, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    for post in data:
                        analysis = post.get('analysis', {})
                        clinic_sentiments = analysis.get('clinic_sentiments', [])
                        
                        if clinic_sentiments:
                            sample_count += 1
                            print(f"\n=== 샘플 {sample_count} ===")
                            print(f"게시글: {post.get('title', 'N/A')}")
                            print(f"clinic_sentiments: {clinic_sentiments}")
                            
                            # 각 감정 분석
                            for i, sentiment in enumerate(clinic_sentiments):
                                print(f"  감정 {i+1}: {sentiment}")
                                sentiment_type = sentiment.get('sentiment', '')
                                clinic_name = sentiment.get('clinic_name', '')
                                print(f"    - sentiment: '{sentiment_type}'")
                                print(f"    - clinic_name: '{clinic_name}'")
                                
                                # 부정적 감정 체크
                                if sentiment_type.lower() in ['negative', '부정', '나쁨', '안좋음']:
                                    found_negative = True
                                    print(f"    *** 부정적 감정 발견! ***")
                            
                            if sample_count >= 5:  # 5개 샘플만 확인
                                break
                    
                    if sample_count >= 5:
                        break
                        
            except Exception as e:
                print(f"파일 읽기 오류 ({filename}): {e}")
    
    if not found_negative:
        print("\n=== 부정적 감정을 찾을 수 없습니다 ===")
        print("가능한 원인:")
        print("1. sentiment 필드가 다른 이름으로 저장됨")
        print("2. 부정적 감정이 다른 값으로 표현됨")
        print("3. clinic_sentiments가 비어있음")
        
        # 추가 검색
        print("\n=== 추가 검색 ===")
        search_all_sentiments()
    
    return found_negative

def search_all_sentiments():
    """
    모든 sentiment 값을 찾아봅니다.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(script_dir, '..', 'data', 'historical_processed')
    
    all_sentiments = set()
    sentiment_clinics = {}
    
    for filename in os.listdir(processed_dir):
        if filename.endswith('_analyzed.json'):
            file_path = os.path.join(processed_dir, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    for post in data:
                        analysis = post.get('analysis', {})
                        clinic_sentiments = analysis.get('clinic_sentiments', [])
                        
                        for sentiment in clinic_sentiments:
                            sentiment_type = sentiment.get('sentiment', '')
                            clinic_name = sentiment.get('clinic_name', '')
                            
                            if sentiment_type:
                                all_sentiments.add(sentiment_type)
                                
                                if sentiment_type not in sentiment_clinics:
                                    sentiment_clinics[sentiment_type] = []
                                sentiment_clinics[sentiment_type].append(clinic_name)
                                
            except Exception as e:
                continue
    
    print(f"발견된 모든 sentiment 값들:")
    for sentiment in sorted(all_sentiments):
        clinics = sentiment_clinics.get(sentiment, [])
        print(f"  '{sentiment}': {len(clinics)}개 병원")
        if len(clinics) <= 5:
            print(f"    병원들: {clinics}")
        else:
            print(f"    병원들: {clinics[:5]}... 외 {len(clinics)-5}개")

if __name__ == "__main__":
    debug_negative_sentiments() 