import pandas as pd
import time
import os
import sys
from tqdm import tqdm
from dotenv import load_dotenv, set_key
import json

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from blog_automation.config import KEYWORD_CSV_PATH, KEYWORD_RESULTS_PATH, NAVER_API_AUTHORIZATION, NAVER_API_COOKIE
from blog_automation.app.services.naver_keyword_tool import get_headers, fetch_keyword_data, parse_keyword_data, save_results

def load_keywords():
    """CSV 파일에서 키워드 목록을 로드합니다."""
    if not os.path.exists(KEYWORD_CSV_PATH):
        print(f"오류: {KEYWORD_CSV_PATH} 파일을 찾을 수 없습니다.")
        return []
    df = pd.read_csv(KEYWORD_CSV_PATH)
    return df['searchQuery'].tolist()

def get_processed_keywords():
    """이미 처리된 키워드 목록을 로드합니다."""
    if not os.path.exists(KEYWORD_RESULTS_PATH):
        return set()
    with open(KEYWORD_RESULTS_PATH, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            return set(data.keys())
        except json.JSONDecodeError:
            return set()

def update_env_file(key, value):
    """ .env 파일의 환경 변수 값을 업데이트합니다. """
    env_path = '.env'
    if not os.path.exists(env_path):
        with open(env_path, 'w') as f:
            pass
    
    load_dotenv(override=True)
    set_key(env_path, key, value)
    print(f"{key}가 .env 파일에 업데이트되었습니다.")

def main():
    """메인 실행 함수"""
    load_dotenv()

    keywords = load_keywords()
    if not keywords:
        return

    processed_keywords = get_processed_keywords()
    
    keywords_to_process = [kw for kw in keywords if kw not in processed_keywords]

    if not keywords_to_process:
        print("모든 키워드 처리가 완료되었습니다.")
        return

    print(f"총 {len(keywords)}개의 키워드 중 {len(keywords_to_process)}개를 처리합니다.")

    headers = get_headers()

    for keyword in tqdm(keywords_to_process, desc="키워드 처리 중"):
        while True:
            data = fetch_keyword_data(keyword, headers)

            if data is None:
                print("\nAPI 요청 실패. 인증 정보가 만료되었을 수 있습니다.")
                print("백그라운드 작업에서는 인증 정보를 업데이트할 수 없습니다.")
                print(".env 파일의 NAVER_API_AUTHORIZATION, NAVER_API_COOKIE, NAVER_API_REFERER 값을 수정한 후 서버를 재시작해주세요.")
                return # Stop the background task
            
            parsed_data = parse_keyword_data(data, keyword)
            save_results(parsed_data, KEYWORD_RESULTS_PATH)
            time.sleep(0.5)  # API 요청 간 딜레이
            break # 다음 키워드로

if __name__ == "__main__":
    main() 