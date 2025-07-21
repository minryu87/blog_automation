import json
import os
import sys
import time
from tqdm import tqdm
from dotenv import load_dotenv
import re

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from blog_automation.config import KEYWORD_RESULTS_PATH
from blog_automation.app.services.naver_keyword_tool import get_headers, fetch_keyword_data, parse_keyword_data

def load_results_data():
    """결과 JSON 파일을 로드합니다."""
    if not os.path.exists(KEYWORD_RESULTS_PATH):
        print(f"오류: {KEYWORD_RESULTS_PATH} 파일을 찾을 수 없습니다.")
        return None
    with open(KEYWORD_RESULTS_PATH, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print(f"오류: {KEYWORD_RESULTS_PATH} 파일이 손상되었습니다.")
            return None

def find_failed_keywords(data):
    """데이터가 비어있는 키워드 목록을 찾습니다."""
    return [keyword for keyword, value in data.items() if not value]

def save_updated_results(data):
    """업데이트된 전체 데이터를 다시 저장합니다."""
    with open(KEYWORD_RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main():
    """메인 실행 함수"""
    load_dotenv()

    all_data = load_results_data()
    if not all_data:
        return

    failed_keywords = find_failed_keywords(all_data)

    if not failed_keywords:
        print("데이터가 비어있는 키워드가 없습니다. 모든 작업이 완료되었습니다.")
        return

    # 영어가 포함된 실패 키워드만 필터링
    english_failed_keywords = [
        kw for kw in failed_keywords if re.search('[a-zA-Z]', kw)
    ]

    if not english_failed_keywords:
        print("실패한 키워드 중 영어가 포함된 항목이 없습니다.")
        return

    print(f"총 {len(english_failed_keywords)}개의 실패한 영문 포함 키워드에 대해 재시도합니다.")
    
    try:
        headers = get_headers()
    except ValueError as e:
        print(f"오류: {e}")
        print(".env 파일에 필요한 모든 값이 설정되었는지 확인하세요.")
        return

    updated_count = 0
    for keyword in tqdm(english_failed_keywords, desc="실패 키워드 보완 중"):
        data = fetch_keyword_data(keyword, headers)
        
        success = False
        if data:
            parsed_data_dict = parse_keyword_data(data, keyword)
            # Check if the parsed data for the keyword is not empty
            if parsed_data_dict.get(keyword):
                all_data[keyword] = parsed_data_dict[keyword]
                updated_count += 1
                # Save the updated data immediately
                save_updated_results(all_data)
                success = True

        if success:
            tqdm.write(f"{keyword} | 성공")
        else:
            tqdm.write(f"{keyword} | 실패")

        time.sleep(0.5) # API 요청 간 딜레이

    if updated_count > 0:
        print(f"\n총 {updated_count}개의 키워드 데이터를 성공적으로 보완했습니다.")
    else:
        print("\n새롭게 보완된 데이터가 없습니다.")

if __name__ == "__main__":
    main() 