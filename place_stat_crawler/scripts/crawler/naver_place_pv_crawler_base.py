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

from ..util.config import get_config_manager

logger = logging.getLogger(__name__)

class NaverCrawlerBase:
    """네이버 크롤링 기본 클래스"""
    
    def __init__(self, 
                 naver_id: str = None, 
                 naver_password: str = None,
                 auth_manager: NaverAuthManager = None):
        """
        Args:
            naver_id: 네이버 아이디 (환경 변수에서 자동 로드) - 클라이언트 선택 시 무시됨
            naver_password: 네이버 비밀번호 (환경 변수에서 자동 로드) - 클라이언트 선택 시 무시됨
            auth_manager: 기존 인증 매니저 인스턴스
        """
        import os
        
        # 설정 매니저에서 클라이언트 정보 조회
        client_id = None
        client_pw = None
        client_login_url = None
        client_name = None
        auth_file_path = None
        token_expiry_hours = 12
        try:
            config_manager = get_config_manager()
            auth_config = config_manager.get_auth_config()
            selected_client = config_manager.get_selected_client_config()
            if selected_client:
                client_id = selected_client.id
                client_pw = selected_client.pw
                client_login_url = selected_client.naver_url or "https://nid.naver.com/nidlogin.login"
                client_name = selected_client.name
                token_expiry_hours = auth_config.token_expiry_hours
                # 클라이언트별 인증 파일 경로
                template = auth_config.auth_file_path_template or "{client_name}_auth.json"
                auth_file_path = template.format(client_name=client_name)
        except Exception as e:
            logger.warning(f"설정 매니저 초기화 중 문제 발생: {e}. 환경 변수로 폴백합니다.")
        
        # 환경 변수에서 인증 정보 로드 (클라이언트 설정 미존재 시 폴백)
        if not client_id:
            client_id = naver_id or os.getenv('NAVER_ID')
        if not client_pw:
            client_pw = naver_password or os.getenv('NAVER_PASSWORD')
        if not client_login_url:
            client_login_url = "https://nid.naver.com/nidlogin.login"
        if not client_name:
            client_name = "default"
        if not auth_file_path:
            auth_file_path = "naver_auth.json"
        
        if not client_id or not client_pw:
            logger.warning("네이버 인증 정보가 설정되지 않았습니다. .env 또는 클라이언트 설정을 확인해주세요.")
        
        # 인증 매니저 설정
        if auth_manager:
            self.auth_manager = auth_manager
        else:
            self.auth_manager = NaverAuthManager(
                naver_id=client_id,
                naver_password=client_pw,
                auth_file_path=auth_file_path,
                token_expiry_hours=token_expiry_hours,
                naver_login_url=client_login_url,
                client_name=client_name
            )
        
        # 세션 설정
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
            'Referer': 'https://new.smartplace.naver.com/'
        })
        
        # 재시도 설정
        self.max_retries = 3
        self.retry_delay = 1
        
        # 통계
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        
        logger.info("네이버 크롤러 기본 클래스 초기화 완료")

    def get_auth_headers(self) -> Dict[str, str]:
        """인증 헤더 반환"""
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
                    timeout: int = 30) -> Optional[requests.Response]:
        """HTTP 요청 실행"""
        # 인증 정보 갱신
        self.auth_manager.refresh_auth_if_needed()
        
        # 헤더 설정
        request_headers = self.get_auth_headers()
        if headers:
            request_headers.update(headers)
        
        # 쿠키 설정
        cookies = self.get_cookies()
        
        for attempt in range(self.max_retries):
            try:
                self.request_count += 1
                
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    data=data,
                    json=json_data,
                    headers=request_headers,
                    cookies=cookies,
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    try:
                        return response.json()
                    except json.JSONDecodeError:
                        logger.error(f"JSON 디코딩 실패: {response.text[:200]}")
                        return None
                else:
                    logger.warning(f"HTTP {response.status_code} 에러 (시도 {attempt + 1}/{self.max_retries})")
                    if response.status_code == 401:
                        # 인증 오류 시 토큰 갱신 시도
                        self.auth_manager.refresh_auth_if_needed()
                    elif response.status_code == 429:
                        # Rate limit 시 더 긴 대기
                        time.sleep(self.retry_delay * 2)
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"요청 오류 (시도 {attempt + 1}/{self.max_retries}): {e}")
            
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)
        
        self.error_count += 1
        logger.error(f"⚠️ HTTP No response 에러")
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
    crawler = NaverCrawlerBase()
    
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
