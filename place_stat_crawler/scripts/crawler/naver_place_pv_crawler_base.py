#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
네이버 크롤링 기본 클래스
인증 관리와 공통 기능을 제공합니다.
"""

import requests
import time
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

# .env 자동 로드 (place_stat_crawler/.env)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / '.env')
except Exception:
    pass

# 상대 import를 절대 import로 변경
try:
    from scripts.crawler.naver_place_pv_auth_manager import NaverAuthManager
    from ..util.data_saver import DataSaver
    from ..util.config import ClientInfo
except ImportError:
    # 직접 실행 시를 위한 절대 import
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    util_dir = os.path.join(parent_dir, 'util')
    sys.path.insert(0, parent_dir)
    sys.path.insert(0, current_dir)
    sys.path.insert(0, util_dir)
    from scripts.crawler.naver_place_pv_auth_manager import NaverAuthManager
    try:
        from data_saver import DataSaver
    except ImportError:
        DataSaver = None  # DataSaver가 없어도 작동하도록
    from ..util.config import ClientInfo

from ..util.config import get_config_manager

# 추가: API 호출 오류를 위한 사용자 정의 예외
class ApiCallError(Exception):
    """API 호출 관련 오류(인증 실패 등)"""
    pass

logger = logging.getLogger(__name__)

class NaverCrawlerBase:
    """네이버 스마트플레이스 API 크롤러의 기본 클래스"""

    def __init__(self, client_info: ClientInfo, auth_manager: Optional[NaverAuthManager] = None):
        self.config_manager = get_config_manager()
        self.auth_manager = auth_manager
        self.client_info = client_info # client_info 직접 저장
        self.session = requests.Session()
        self.timeout = 10
        self.base_url = ""
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        self.max_retries = 3
        self.retry_delay = 1
        
        # auth_manager로부터 직접 클라이언트 정보 가져오기 -> client_info를 직접 사용하도록 변경
        self.selected_client = self.client_info
        
        if self.selected_client:
            self.base_url = self.selected_client.naver_url.split('/statistics')[0]
            logger.info("네이버 크롤러 기본 클래스 초기화 완료")
        else:
            logger.warning("NaverCrawlerBase 초기화 시 클라이언트 정보가 없습니다.")

    def get_auth_headers(self) -> Dict[str, str]:
        """인증 헤더 반환 (서브클래스에서 필요시 오버라이드)"""
        return self.auth_manager.get_auth_headers()

    def get_cookies(self) -> Dict[str, str]:
        """쿠키 반환"""
        return self.auth_manager.get_cookies()

    def make_request(self, 
                    method: str, 
                    url: str, 
                    params: Dict = None, 
                    data: Dict = None,
                    json_data: Dict = None,
                    headers: Dict = None,
                    timeout: int = 30) -> Optional[Any]:
        """
        재시도 로직을 포함하여 HTTP 요청을 보냅니다.
        """
        # self.auth_manager.refresh_auth_if_needed() # 더 이상 필요 없으므로 삭제

        # 헤더 설정
        if self.auth_manager:
            request_headers = self.auth_manager.get_auth_headers()
        else: # booking crawler의 경우
            request_headers = self.get_auth_headers()
            
        if headers:
            request_headers.update(headers)

        # 쿠키 설정
        if self.auth_manager:
            cookies = self.auth_manager.get_cookies()
        else: # booking crawler의 경우
            cookies = self.get_cookies()

        self.request_count += 1
        for attempt in range(self.max_retries):
            try:
                response = self.session.request(
                    method,
                    url,
                    params=params,
                    data=data,
                    json=json_data,
                    headers=request_headers,
                    cookies=cookies,
                    timeout=timeout
                )
                
                # 200이 아닌 상태 코드에 대한 명시적 에러 발생
                response.raise_for_status()

                # 성공(200) 시, JSON 파싱 시도. 실패 시 ApiCallError 발생
                try:
                    self.success_count += 1
                    return response.json()
                except requests.exceptions.JSONDecodeError:
                    # 200 응답이지만 JSON이 아닌 경우, 인증 오류로 간주
                    raise ApiCallError("API 응답이 올바른 JSON 형식이 아닙니다. 쿠키(cookie) 또는 인증 토큰(auth_token)이 만료되었을 수 있습니다.")

            except requests.exceptions.HTTPError as e:
                # 4xx, 5xx 에러 처리
                self.error_count += 1
                logger.warning(f"⚠️ HTTP {e.response.status_code} 에러 (시도 {attempt + 1}/{self.max_retries}): {e.response.text[:200]}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise ApiCallError(f"HTTP {e.response.status_code} 에러 후 재시도 실패") from e
            
            except requests.exceptions.RequestException as e:
                self.error_count += 1
                logger.error(f"❌ HTTP 요청 실패 (시도 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        logger.error("⚠️ 모든 재시도 실패 후에도 응답을 받지 못했습니다.")
        return None

    def get(self, url: str, params: Dict = None, headers: Dict = None, timeout: int = 30) -> Optional[requests.Response]:
        """GET 요청"""
        return self.make_request('GET', url, params=params, headers=headers, timeout=timeout)

    def post(self, url: str, data: Dict = None, json_data: Dict = None, headers: Dict = None, timeout: int = 30) -> Optional[requests.Response]:
        """POST 요청"""
        return self.make_request('POST', url, data=data, json_data=json_data, headers=headers, timeout=timeout)

    def test_connection(self) -> bool:
        """연결 테스트"""
        try:
            response = self.get("https://www.naver.com", timeout=10)
            return response is not None and response.status_code == 200
        except Exception:
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        return {
            'request_count': self.request_count,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'success_rate': (self.success_count / self.request_count * 100) if self.request_count > 0 else 0
        }

    def reset_statistics(self):
        """통계 초기화"""
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0

    def save_session(self, file_path: str):
        """세션 저장"""
        session_data = {
            'cookies': dict(self.session.cookies),
            'headers': dict(self.session.headers),
            'auth_token': self.auth_manager.auth_token,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"세션 저장 완료: {file_path}")

    def load_session(self, file_path: str) -> bool:
        """세션 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            # 쿠키 복원
            for name, value in session_data.get('cookies', {}).items():
                self.session.cookies.set(name, value)
            
            # 헤더 복원
            if 'headers' in session_data:
                self.session.headers.update(session_data['headers'])
            
            # 인증 토큰 복원
            if 'auth_token' in session_data:
                self.auth_manager.auth_token = session_data['auth_token']
            
            logger.info(f"세션 로드 완료: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"세션 로드 실패: {e}")
            return False


def main():
    """테스트용 메인 함수"""
    # 이 부분은 실제 사용 시에는 클라이언트 정보를 가져와서 인스턴스화해야 합니다.
    # 예: client_info = get_config_manager().get_client_info("your_client_name")
    # crawler = NaverCrawlerBase(client_info)
    
    # 임시로 클라이언트 정보를 생성하여 사용
    client_info = ClientInfo(
        client_name="test_client",
        naver_url="https://www.naver.com/statistics",
        client_id="test_client_id",
        client_secret="test_client_secret",
        auth_token="test_auth_token",
        auth_token_expires_at=datetime.now() + timedelta(hours=1)
    )
    auth_manager = NaverAuthManager(client_info)
    crawler = NaverCrawlerBase(client_info, auth_manager)
    
    # 연결 테스트
    if crawler.test_connection():
        print("✅ 연결 테스트 성공")
    else:
        print("❌ 연결 테스트 실패")
    
    # 통계 출력
    stats = crawler.get_statistics()
    print(f"📊 통계: {stats}")


if __name__ == "__main__":
    main()
