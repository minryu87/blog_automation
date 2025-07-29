import os
import requests
import logging
from urllib.parse import urlencode, quote
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
from tqdm.contrib.logging import tqdm_logging_redirect

# --- Custom Exception ---
class CookieExpiredError(Exception):
    """쿠키 만료 오류를 위한 사용자 정의 예외"""
    pass

# --- Configuration ---
# .env 파일에서 환경 변수 로드 (프로젝트 어디에서 실행하든 .env를 찾아냄)
load_dotenv(find_dotenv())
NAVER_API_COOKIE = os.getenv("NAVER_API_COOKIE")

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 상수 정의 (기존 creator_advisor_service.py 참조)
CHANNEL_ID = "elza79"
SERVICE = "naver_blog"
START_DATE = "2024-06-01"
END_DATE = "2025-06-30"

# --- Helper Functions (기존 creator_advisor_service.py에서 가져옴) ---

def _get_headers(referer_url: str):
    """API 요청에 필요한 헤더를 반환합니다."""
    return {
        "Cookie": NAVER_API_COOKIE,
        "Referer": referer_url,
        "Accept": "application/json, text/plain, */*",
    }

def _make_api_request(url: str, headers: dict):
    """주어진 URL과 헤더로 API를 요청하고 결과를 반환합니다."""
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code in [401, 403]:
            # 401 또는 403 코드는 쿠키 만료 가능성이 높음
            raise CookieExpiredError(f"인증 오류 (상태 코드: {e.response.status_code}). 쿠키가 만료되었을 수 있습니다.")
        logger.error(f"HTTP 오류 발생: {url}, 에러: {e}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"API 요청 실패: {url}, 에러: {e}")
        return None

def _make_referer_url(keyword: str) -> str:
    """API 요청에 필요한 Referer URL을 생성합니다. (한글 키워드 인코딩 포함)"""
    keyword_encoded = quote(keyword)
    return (
        f"https://creator-advisor.naver.com/new-windows/query-stats?"
        f"channelId={CHANNEL_ID}&contentType=text&endDate={END_DATE}&interval=month&metric=cv&query={keyword_encoded}&service={SERVICE}&startDate={START_DATE}"
    )

# --- Main Logic ---

def fetch_keyword_inflow(keywords: list, output_file: str, file_exists: bool):
    """
    주어진 키워드 리스트에 대해 월별 searchInflow 데이터를 수집하고 파일에 즉시 저장합니다.
    쿠키 만료 시 작업을 중단합니다.
    """
    logger.info(f"총 {len(keywords)}개 키워드에 대한 월별 유입량 수집 시작...")
    
    # tqdm 로깅 리디렉션을 통해 로그와 진행률 표시의 충돌 방지
    with tqdm_logging_redirect(disable=False):
        try:
            for keyword in tqdm(keywords, desc="키워드별 유입량 수집 중"):
                trend_params = {
                    "channelId": CHANNEL_ID,
                    "endDate": END_DATE,
                    "interval": "month",
                    "keyword": keyword,
                    "service": SERVICE,
                    "startDate": START_DATE
                }
                trend_url = f"https://creator-advisor.naver.com/api/v6/inflow-analysis/inflow-search-trend?{urlencode(trend_params)}"
                trend_headers = _get_headers(_make_referer_url(keyword))
                trend_data = _make_api_request(trend_url, trend_headers)

                if not trend_data or not trend_data.get("data"):
                    logger.warning(f"'{keyword}'에 대한 트렌드 데이터를 가져오지 못했습니다. 건너뜁니다.")
                    continue
                
                # 현재 키워드에 대한 데이터프레임 생성
                keyword_df = pd.DataFrame([
                    {
                        "searchquery": keyword,
                        "date": item.get("date"),
                        "searchinflow": item.get("searchInflow")
                    } for item in trend_data["data"]
                ])
                
                # CSV 파일에 즉시 추가 (append 모드)
                if not keyword_df.empty:
                    # 파일이 없으면 헤더를 쓰고, 있으면 헤더 없이 내용만 추가
                    keyword_df.to_csv(output_file, mode='a', index=False, header=(not file_exists), encoding='utf-8-sig')
                    file_exists = True # 첫 쓰기 이후에는 파일이 항상 존재하는 것으로 간주
        
        except CookieExpiredError as e:
            logger.error("\n" + "="*60)
            logger.error(f"오류: {e}")
            logger.error("스크립트를 중단합니다. .env 파일의 NAVER_API_COOKIE를 업데이트해주세요.")
            logger.error("업데이트 후 스크립트를 다시 실행하면 중단된 지점부터 자동으로 이어집니다.")
            logger.error("="*60)
            return False # 실패를 알림

    return True # 성공을 알림

def main():
    """스크립트 실행 메인 함수"""
    if not NAVER_API_COOKIE:
        logger.error("환경변수 'NAVER_API_COOKIE'가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        return

    # 동적으로 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__)) # /blog_automation/scripts
    project_root = os.path.dirname(os.path.dirname(script_dir)) # /medilawyer_sales
    
    input_file = os.path.join(project_root, 'blog_automation', 'data', 'data_input', 'searchQuery.csv')
    output_dir = os.path.join(project_root, 'blog_automation', 'searchquery_analysis', 'data')
    output_file = os.path.join(output_dir, 'search_inflow_by_month.csv')

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"결과물 저장 경로: {output_file}")

    # 전체 키워드 로드
    try:
        keywords_df = pd.read_csv(input_file)
        full_keywords_list = keywords_df.iloc[:, 0].dropna().unique().tolist()
        logger.info(f"'{os.path.basename(input_file)}' 파일에서 {len(full_keywords_list)}개의 유니크 키워드를 로드했습니다.")
    except FileNotFoundError:
        logger.error(f"입력 파일을 찾을 수 없습니다: {input_file}")
        return
    except Exception as e:
        logger.error(f"키워드 파일 로드 중 오류 발생: {e}")
        return

    # --- 이어하기 로직 ---
    processed_keywords = set()
    output_file_exists = os.path.exists(output_file)
    if output_file_exists:
        try:
            # 기존 결과 파일이 비어있는 경우를 대비하여 예외 처리
            processed_df = pd.read_csv(output_file)
            if not processed_df.empty and 'searchquery' in processed_df.columns:
                processed_keywords = set(processed_df['searchquery'].unique())
                logger.info(f"기존 결과 파일에서 {len(processed_keywords)}개의 처리된 키워드를 발견했습니다.")
        except pd.errors.EmptyDataError:
            logger.info("기존 결과 파일이 비어있어 처음부터 시작합니다.")
            output_file_exists = False # 비어있으면 헤더를 다시 써야 함
        except Exception as e:
            logger.error(f"기존 결과 파일을 읽는 중 오류 발생: {e}. 안전을 위해 파일을 비우고 다시 시작합니다.")
            # 문제가 있는 파일은 덮어쓰기 위해 초기화
            with open(output_file, 'w') as f:
                pass
            output_file_exists = False

    keywords_to_process = [k for k in full_keywords_list if k not in processed_keywords]

    if not keywords_to_process:
        logger.info("모든 키워드에 대한 데이터 수집이 이미 완료되었습니다.")
        return

    # 데이터 수집 실행
    logger.info(f"총 {len(full_keywords_list)}개 중 {len(keywords_to_process)}개 키워드에 대한 작업을 시작합니다.")
    success = fetch_keyword_inflow(keywords_to_process, output_file, output_file_exists)

    if success:
        logger.info(f"모든 작업을 성공적으로 완료했습니다.")
    else:
        logger.warning(f"작업이 중단되었습니다. 지금까지의 결과는 '{output_file}'에 저장되었습니다.")

if __name__ == "__main__":
    main() 