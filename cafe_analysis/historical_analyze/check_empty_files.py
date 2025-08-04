import os
import json
import glob

def check_empty_files():
    """
    historical_processed 폴더에서 빈 JSON 파일들을 찾습니다.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(script_dir, '..', 'data', 'historical_processed')
    
    print("=== 빈 JSON 파일 확인 ===")
    
    if not os.path.exists(processed_dir):
        print("processed 폴더를 찾을 수 없습니다.")
        return
    
    # 모든 analyzed.json 파일 찾기
    json_files = glob.glob(os.path.join(processed_dir, '*_analyzed.json'))
    print(f"총 {len(json_files)}개의 JSON 파일을 확인합니다.")
    
    empty_files = []
    total_files = 0
    
    for file_path in json_files:
        total_files += 1
        filename = os.path.basename(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
                if not content or content == '[]':
                    empty_files.append(filename)
                    print(f"  빈 파일 발견: {filename}")
                else:
                    # JSON 파싱 테스트
                    f.seek(0)
                    data = json.load(f)
                    if not data or len(data) == 0:
                        empty_files.append(filename)
                        print(f"  빈 배열 파일 발견: {filename}")
                        
        except Exception as e:
            print(f"  오류 발생 ({filename}): {e}")
    
    print(f"\n=== 결과 요약 ===")
    print(f"총 파일 수: {total_files}")
    print(f"빈 파일 수: {len(empty_files)}")
    print(f"정상 파일 수: {total_files - len(empty_files)}")
    
    if empty_files:
        print(f"\n=== 빈 파일 목록 ===")
        for filename in sorted(empty_files):
            print(f"  - {filename}")
    
    return empty_files

if __name__ == "__main__":
    check_empty_files() 