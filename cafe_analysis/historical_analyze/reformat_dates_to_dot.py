import os
import json
import re

def reformat_dates_in_files():
    """
    historical_processed 폴더의 7월 데이터 파일들을 순회하며
    'date' 필드를 'YYYY-MM-DD'에서 'YYYY.MM.DD' 형식으로 변환하고 덮어씁니다.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cafe_analysis_dir = os.path.dirname(script_dir)
    processed_dir = os.path.join(cafe_analysis_dir, 'data', 'historical_processed')

    if not os.path.exists(processed_dir):
        print(f"오류: '{processed_dir}' 폴더를 찾을 수 없습니다.")
        return
    
    print(f"--- '{processed_dir}' 폴더의 7월 데이터 날짜 형식 재변환 시작 ---")
    
    try:
        target_files = [f for f in os.listdir(processed_dir) if f.startswith('2025-07') and f.endswith('_analyzed.json')]
        
        if not target_files:
            print("변환할 7월 데이터 파일이 없습니다.")
            return

        total_files = len(target_files)
        total_converted_count = 0

        for i, filename in enumerate(target_files, 1):
            file_path = os.path.join(processed_dir, filename)
            
            print(f"[{i}/{total_files}] 처리 중: {filename}")

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_converted_count = 0
            # 각 게시물의 날짜 변환
            for post in data:
                date_str = post.get('date', '')
                # YYYY-MM-DD 형식인지 확인
                if isinstance(date_str, str) and re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
                    post['date'] = date_str.replace('-', '.')
                    file_converted_count += 1
            
            # 변경된 내용이 있을 경우에만 파일 덮어쓰기
            if file_converted_count > 0:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                print(f"    -> {file_converted_count}개의 날짜 형식을 'YYYY.MM.DD'로 변경했습니다.")
                total_converted_count += file_converted_count
        
        print(f"\n===== 총 {total_converted_count}개의 날짜 형식 변환이 완료되었습니다. =====")

    except Exception as e:
        print(f"\n[오류] 파일 처리 중 예외가 발생했습니다: {e}")

if __name__ == "__main__":
    reformat_dates_in_files() 