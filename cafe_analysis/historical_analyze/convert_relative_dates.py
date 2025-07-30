import os
import json
import re
from datetime import datetime, timedelta

def convert_date_str_to_iso(date_str: str, base_date: datetime) -> str:
    """
    'N시간 전', 'N일 전' 등 다양한 형식의 날짜 문자열을 'YYYY-MM-DD' 형식으로 변환합니다.
    """
    if not isinstance(date_str, str):
        return base_date.strftime('%Y-%m-%d')

    # 'YYYY.MM.DD.' 형식 처리
    match = re.match(r'(\d{4})\.(\d{2})\.(\d{2})\.', date_str)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

    # 'N시간 전' 형식 처리
    match = re.match(r'(\d+)시간 전', date_str)
    if match:
        hours_ago = int(match.group(1))
        # 시간 단위는 보통 같은 날짜이므로, 기준 날짜를 그대로 사용
        return base_date.strftime('%Y-%m-%d')

    # 'N일 전' 형식 처리
    match = re.match(r'(\d+)일 전', date_str)
    if match:
        days_ago = int(match.group(1))
        actual_date = base_date - timedelta(days=days_ago)
        return actual_date.strftime('%Y-%m-%d')
        
    # 'N주 전' 형식 처리
    match = re.match(r'(\d+)주 전', date_str)
    if match:
        weeks_ago = int(match.group(1))
        actual_date = base_date - timedelta(weeks=weeks_ago)
        return actual_date.strftime('%Y-%m-%d')
        
    # 처리할 수 없는 형식은 오늘 날짜로 반환
    return base_date.strftime('%Y-%m-%d')

def convert_dates_in_files():
    """
    historical_processed 폴더의 7월 데이터 파일들을 순회하며
    'date' 필드를 'YYYY-MM-DD' 형식으로 변환하고 덮어씁니다.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cafe_analysis_dir = os.path.dirname(script_dir)
    processed_dir = os.path.join(cafe_analysis_dir, 'data', 'historical_processed')

    if not os.path.exists(processed_dir):
        print(f"오류: '{processed_dir}' 폴더를 찾을 수 없습니다.")
        return

    # 날짜 계산의 기준이 될 오늘 날짜
    base_date = datetime.now()
    
    print(f"--- '{processed_dir}' 폴더의 7월 데이터 날짜 변환 시작 ---")
    
    try:
        target_files = [f for f in os.listdir(processed_dir) if f.startswith('2025-07') and f.endswith('_analyzed.json')]
        
        if not target_files:
            print("변환할 7월 데이터 파일이 없습니다.")
            return

        total_files = len(target_files)
        for i, filename in enumerate(target_files, 1):
            file_path = os.path.join(processed_dir, filename)
            
            print(f"[{i}/{total_files}] 처리 중: {filename}")

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 각 게시물의 날짜 변환
            for post in data:
                original_date = post.get('date', '')
                post['date'] = convert_date_str_to_iso(original_date, base_date)
            
            # 수정된 내용으로 파일 덮어쓰기
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        
        print("\n===== 모든 파일의 날짜 변환이 완료되었습니다. =====")

    except Exception as e:
        print(f"\n[오류] 파일 처리 중 예외가 발생했습니다: {e}")

if __name__ == "__main__":
    convert_dates_in_files() 