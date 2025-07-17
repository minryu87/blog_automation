import requests
import json
import pandas as pd
from tqdm import tqdm
import os
import sys
import re

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import NAVER_API_URL, NAVER_API_AUTHORIZATION, NAVER_API_COOKIE, KEYWORD_CSV_PATH, KEYWORD_RESULTS_PATH, NAVER_API_REFERER

def get_headers():
    """API 요청을 위한 헤더를 생성합니다."""
    if not NAVER_API_AUTHORIZATION or not NAVER_API_COOKIE or not NAVER_API_REFERER:
        raise ValueError("환경변수에서 NAVER_API_AUTHORIZATION, NAVER_API_COOKIE, NAVER_API_REFERER를 설정해야 합니다.")
    
    headers = {
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
        'Authorization': NAVER_API_AUTHORIZATION,
        'Content-Type': 'application/json;charset=UTF-8',
        'Cookie': NAVER_API_COOKIE,
        'Origin': 'https://manage.searchad.naver.com',
        'Referer': NAVER_API_REFERER,
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'
    }
    return headers

def fetch_keyword_data(keyword, headers):
    """지정된 키워드에 대한 데이터를 네이버 API를 통해 가져옵니다. (GET 요청으로 변경)"""
    
    processed_keyword = keyword.replace(" ", "")
    # 영어 알파벳이 포함된 경우 대문자로 변환
    if re.search('[a-zA-Z]', processed_keyword):
        processed_keyword = processed_keyword.upper()

    params = {
        'format': 'json',
        'siteId': '',
        'month': '',
        'biztpId': '',
        'event': '',
        'includeHintKeywords': '0',
        'showDetail': '1',
        'keyword': processed_keyword
    }
    try:
        response = requests.get(NAVER_API_URL, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as err:
        if response.status_code in [401, 403]: # Unauthorized or Forbidden
            print(f"인증 실패 ({response.status_code}). 새로운 Authorization 또는 Cookie 값이 필요합니다.")
            return None
        else:
            print(f"HTTP 에러 발생: {err}")
            return None
    except requests.exceptions.RequestException as err:
        print(f"요청 중 에러 발생: {err}")
        return None

def parse_keyword_data(data, keyword):
    """API 응답 데이터를 파싱하여 월별 검색량 데이터를 추출합니다."""
    if not data or 'keywordList' not in data or not data['keywordList']:
        return {keyword: {}}

    keyword_info = data['keywordList'][0]
    monthly_progress_list = keyword_info.get('monthlyProgressList', {})
    
    monthly_labels = monthly_progress_list.get('monthlyLabel', [])
    pc_counts = monthly_progress_list.get('monthlyProgressPcQcCnt', [])
    mobile_counts = monthly_progress_list.get('monthlyProgressMobileQcCnt', [])

    result = {}
    for i, label in enumerate(monthly_labels):
        result[label] = {
            'monthlyProgressPcQcCnt': pc_counts[i] if i < len(pc_counts) else 0,
            'monthlyProgressMobileQcCnt': mobile_counts[i] if i < len(mobile_counts) else 0
        }
    return {keyword: result}

def save_results(data, filepath):
    """결과를 JSON 파일에 저장합니다."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    existing_data = {}
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: {filepath} is corrupted. A new file will be created.")
    
    existing_data.update(data)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4) 


def fetch_suggested_keyword_data(keyword, headers):
    """연관 키워드 포함 옵션으로 API를 호출합니다."""
    processed_keyword = keyword.replace(" ", "")
    # 영어 알파벳이 포함된 경우 대문자로 변환
    if re.search('[a-zA-Z]', processed_keyword):
        processed_keyword = processed_keyword.upper()
        
    params = {
        'format': 'json',
        'siteId': '',
        'month': '',
        'biztpId': '',
        'event': '',
        'includeHintKeywords': '1', # 연관 키워드 포함
        'showDetail': '1',
        'keyword': processed_keyword
    }
    try:
        response = requests.get(NAVER_API_URL, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as err:
        if response.status_code in [401, 403]:
            print(f"인증 실패 ({response.status_code}). .env 파일을 확인해주세요.")
        else:
            print(f"HTTP 에러 발생: {err}")
        return None
    except requests.exceptions.RequestException as err:
        print(f"요청 중 에러 발생: {err}")
        return None

def parse_suggested_data(data):
    """연관 키워드 응답에서 유효한 첫 번째 키워드 데이터를 파싱합니다."""
    if not data or 'keywordList' not in data or not data['keywordList']:
        return {}

    for keyword_info in data['keywordList']:
        monthly_progress_list = keyword_info.get('monthlyProgressList', {})
        monthly_labels = monthly_progress_list.get('monthlyLabel', [])
        
        if monthly_labels:
            pc_counts = monthly_progress_list.get('monthlyProgressPcQcCnt', [])
            mobile_counts = monthly_progress_list.get('monthlyProgressMobileQcCnt', [])
            
            result = {}
            for i, label in enumerate(monthly_labels):
                result[label] = {
                    'monthlyProgressPcQcCnt': pc_counts[i] if i < len(pc_counts) else 0,
                    'monthlyProgressMobileQcCnt': mobile_counts[i] if i < len(mobile_counts) else 0
                }
            return result
            
    return {} 