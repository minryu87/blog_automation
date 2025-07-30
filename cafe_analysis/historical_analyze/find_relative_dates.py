import os
import json
from datetime import datetime

def find_non_standard_dates():
    """
    historical_processed 폴더에서 7월 데이터만 대상으로,
    'YYYY.' 형식으로 시작하지 않는 날짜 값을 찾아 중복 없이 출력합니다.
    """
    
    # 이 스크립트의 위치를 기준으로 상위 cafe_analysis 폴더의 경로를 찾습니다.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # historical_analyze -> cafe_analysis
    cafe_analysis_dir = os.path.dirname(script_dir)
    processed_dir = os.path.join(cafe_analysis_dir, 'data', 'historical_processed')

    if not os.path.exists(processed_dir):
        print(f"오류: '{processed_dir}' 폴더를 찾을 수 없습니다.")
        return

    # 'YYYY.' 형식으로 시작하지 않는 날짜들을 저장할 집합 (중복 제거)
    relative_dates = set()
    
    # 현재 년도를 기준으로 'YYYY.' 형식 판별 (예: '2025.')
    # 크롤링 시점이나 분석 시점에 따라 유연하게 대처하기 위함
    # 여기서는 분석 시점의 년도를 사용합니다.
    current_year_str = str(datetime.now().year)
    
    print(f"--- '{processed_dir}' 폴더에서 7월 데이터 분석 시작 ---")
    print(f"--- '{current_year_str}.' 로 시작하지 않는 날짜를 찾습니다 ---")

    try:
        # 처리 대상 파일 목록
        target_files = [f for f in os.listdir(processed_dir) if f.startswith('2025-07') and f.endswith('_analyzed.json')]
        
        if not target_files:
            print("분석할 7월 데이터 파일이 없습니다.")
            return

        for filename in target_files:
            file_path = os.path.join(processed_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for post in data:
                # 'date' 필드가 없는 경우를 대비
                date_str = post.get('date')
                if date_str and not date_str.startswith(f"{current_year_str}."):
                    relative_dates.add(date_str)
                    
        print("\n===== 비표준 날짜 형식 목록 (중복 없음) =====")
        if relative_dates:
            for date_val in sorted(list(relative_dates)):
                print(date_val)
        else:
            print("모든 날짜가 'YYYY.' 형식입니다. 비표준 날짜가 발견되지 않았습니다.")

    except Exception as e:
        print(f"\n[오류] 파일 처리 중 예외가 발생했습니다: {e}")


if __name__ == "__main__":
    find_non_standard_dates() 