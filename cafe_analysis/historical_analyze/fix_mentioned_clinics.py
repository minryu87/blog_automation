import os
import json
import glob
from collections import OrderedDict

def fix_mentioned_clinics():
    """
    historical_processed 폴더의 JSON 파일들에서 clinic_sentiments의 clinic_name들을 
    mentioned_clinics에 추가하여 중복 없는 리스트로 만듭니다.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(script_dir, '..', 'data', 'historical_processed')
    
    print("=== mentioned_clinics 수정 작업 시작 ===")
    
    if not os.path.exists(processed_dir):
        print("processed 폴더를 찾을 수 없습니다.")
        return
    
    # 모든 analyzed.json 파일 찾기
    json_files = glob.glob(os.path.join(processed_dir, '*_analyzed.json'))
    print(f"총 {len(json_files)}개의 JSON 파일을 처리합니다.")
    
    total_fixed = 0
    
    for file_path in json_files:
        print(f"\n처리 중: {os.path.basename(file_path)}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_fixed = 0
            
            for post in data:
                analysis = post.get('analysis', {})
                if not analysis:
                    continue
                
                mentioned_clinics = analysis.get('mentioned_clinics', [])
                clinic_sentiments = analysis.get('clinic_sentiments', [])
                
                # clinic_sentiments에서 clinic_name 추출
                sentiment_clinics = []
                for sentiment in clinic_sentiments:
                    clinic_name = sentiment.get('clinic_name')
                    if clinic_name:
                        sentiment_clinics.append(clinic_name)
                
                # 모든 clinic_name을 합쳐서 중복 제거
                all_clinics = mentioned_clinics + sentiment_clinics
                unique_clinics = list(OrderedDict.fromkeys(all_clinics))  # 순서 유지하면서 중복 제거
                
                # mentioned_clinics 업데이트
                if set(unique_clinics) != set(mentioned_clinics):
                    analysis['mentioned_clinics'] = unique_clinics
                    file_fixed += 1
                    print(f"  - 게시글 '{post.get('title', 'N/A')[:30]}...' 수정됨")
                    print(f"    기존: {mentioned_clinics}")
                    print(f"    수정: {unique_clinics}")
            
            # 파일 저장
            if file_fixed > 0:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"  -> {file_fixed}개 게시글 수정 완료")
                total_fixed += file_fixed
            else:
                print(f"  -> 수정할 게시글 없음")
                
        except Exception as e:
            print(f"  오류 발생: {e}")
    
    print(f"\n=== 작업 완료 ===")
    print(f"총 {len(json_files)}개 파일 처리")
    print(f"총 {total_fixed}개 게시글 수정됨")

if __name__ == "__main__":
    fix_mentioned_clinics() 