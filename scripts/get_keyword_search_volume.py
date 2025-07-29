import os
import time
import hmac
import hashlib
import base64
import requests
from dotenv import load_dotenv
import argparse

# .env 파일에서 환경 변수 로드
load_dotenv()

# --- 설정 ---
# 환경 변수에서 API 정보 로드
# naver_ad_api_client.py와 동일한 환경 변수 사용
API_KEY = os.getenv("NAVER_AD_API_KEY")
API_SECRET = os.getenv("NAVER_AD_API_SECRET")
CUSTOMER_ID = os.getenv("NAVER_AD_CUSTOMER_ID")
BASE_URL = "https://api.searchad.naver.com"

def generate_signature(timestamp, method, uri, secret_key):
    """HMAC-SHA256 서명을 생성합니다."""
    message = f"{timestamp}.{method}.{uri}"
    hash = hmac.new(secret_key.encode('utf-8'), message.encode('utf-8'), hashlib.sha256)
    return base64.b64encode(hash.digest()).decode()

def get_keyword_stats(keywords: list, show_detail: int = 1):
    """
    키워드 도구 API를 호출하여 연관 키워드 및 통계 정보를 가져옵니다.
    """
    if not all([API_KEY, API_SECRET, CUSTOMER_ID]):
        print("오류: .env 파일에 NAVER_AD_API_KEY, NAVER_AD_API_SECRET, NAVER_AD_CUSTOMER_ID를 설정해주세요.")
        return

    # API 엔드포인트 및 파라미터
    uri = "/keywordstool"
    method = "GET"
    
    # 쿼리 파라미터 구성
    queryParams = {
        'hintKeywords': ','.join(keywords),
        'showDetail': str(show_detail)
    }
    
    # 요청 URI에 쿼리 파라미터 추가
    request_uri_with_params = uri + '?' + '&'.join([f'{k}={v}' for k, v in queryParams.items()])
    
    # 서명 생성 시에는 쿼리 파라미터를 제외한 순수 URI 경로만 사용합니다.
    timestamp = str(int(time.time() * 1000))
    signature = generate_signature(timestamp, method, uri, API_SECRET) 

    headers = {
        "X-Timestamp": timestamp,
        "X-API-KEY": API_KEY,
        "X-Customer": CUSTOMER_ID,
        "X-Signature": signature
    }

    url = f"{BASE_URL}{request_uri_with_params}"
    
    print("--- API 호출 ---")
    print(f"URL: {method} {url}")
    print("-------------------")

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        print("\n--- API 응답 (성공) ---")
        data = response.json()
        print(f"상태 코드: {response.status_code}")
        print(f"트랜잭션 ID: {response.headers.get('x-transaction-id')}")
        
        if isinstance(data, list) and data:
            # 입력한 키워드와 정확히 일치하는 결과만 필터링
            matched_keywords_data = [item for item in data if item.get('relKeyword') in keywords]
            
            if matched_keywords_data:
                print(f"\n'{','.join(keywords)}'에 대한 최근 30일 검색량:")
                print("-" * 70)
                print(f"{'키워드':<20} | {'월간 PC 검색수':<15} | {'월간 모바일 검색수':<18} | {'경쟁 강도':<10}")
                print("-" * 70)
                for item in matched_keywords_data:
                    print(f"{item.get('relKeyword', 'N/A'):<20} | {item.get('monthlyPcQcCnt', 'N/A'):<18} | {item.get('monthlyMobileQcCnt', 'N/A'):<21} | {item.get('compIdx', 'N/A'):<10}")
                print("-" * 70)
            else:
                print(f"\n입력하신 키워드 '{','.join(keywords)}'에 대한 정확한 데이터를 찾을 수 없습니다.")

        else:
            print("응답 본문 (데이터 없음 또는 예상치 못한 형식):")
            print(data)

        print("------------------------------")
        return data

    except requests.exceptions.RequestException as e:
        print("\n--- API 응답 (오류) ---")
        if e.response is not None:
            print(f"상태 코드: {e.response.status_code}")
            print(f"응답 본문: {e.response.text}")
        else:
            print(f"API 호출 오류: {e}")
        print("----------------------------")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="네이버 검색 광고 API를 사용하여 키워드 통계를 조회합니다.")
    parser.add_argument('keywords', nargs='+', help="통계를 조회할 키워드 (1개 이상, 최대 5개)")
    args = parser.parse_args()

    if len(args.keywords) > 5:
        print("오류: 최대 5개의 키워드만 한 번에 조회할 수 있습니다.")
    else:
        get_keyword_stats(args.keywords) 