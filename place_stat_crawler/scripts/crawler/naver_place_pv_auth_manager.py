#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
네이버 Place 통계용 인증 관리 시스템
.env 파일에서 직접 인증 정보를 로드합니다.
"""

import os
import json
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path
from scripts.util.logger import logger
from scripts.util.config import ClientInfo, AuthConfig

class NaverAuthManager:
    """네이버 Place 인증 처리를 담당하는 클래스"""

    def __init__(self, client_info: ClientInfo, auth_config: AuthConfig, auth_type: str = 'place'):
        self.client_info = client_info
        self.auth_config = auth_config
        self.auth_type = auth_type
        
        # .env에서 직접 로드
        self.auth_token = self._load_token_from_env()
        self.cookies = self._load_cookies_from_env()

    def _load_token_from_env(self) -> Optional[str]:
        """환경 변수에서 인증 토큰을 로드합니다."""
        prefix = self.client_info.name.upper()
        token = os.getenv(f"{prefix}_PLACE_AUTH")
        if token and not token.lower().startswith("bearer "):
            token = f"Bearer {token}"
        return token

    def _load_cookies_from_env(self) -> Dict[str, str]:
        """환경 변수에서 쿠키를 로드하고 파싱합니다."""
        prefix = self.client_info.name.upper()
        cookie_str = os.getenv(f"{prefix}_PLACE_COOKIE", "")
        cookies = {}
        for part in cookie_str.split(';'):
            part = part.strip()
            if not part:
                continue
            if '=' in part:
                name, value = part.split('=', 1)
                cookies[name.strip()] = value.strip()
        return cookies

    def get_auth_headers(self) -> Dict[str, str]:
        """인증 헤더 반환"""
        if not self.auth_token:
            raise ValueError(f"환경 변수 '{self.client_info.name.upper()}_PLACE_AUTH'를 찾을 수 없습니다.")
        
        return {
            'Authorization': self.auth_token,
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Referer': 'https://new.smartplace.naver.com/'
        }
    
    def get_cookies(self) -> Dict[str, str]:
        """쿠키 딕셔너리 반환"""
        if not self.cookies:
            logger.warning(f"환경 변수 '{self.client_info.name.upper()}_PLACE_COOKIE'가 비어있거나 설정되지 않았습니다.")
        return self.cookies


def main():
    """테스트용 메인 함수"""
    naver_id = os.getenv('NAVER_ID')
    naver_password = os.getenv('NAVER_PASSWORD')
    if not naver_id or not naver_password:
        print("환경 변수 NAVER_ID와 NAVER_PASSWORD를 설정해주세요.")
        return
    auth_manager = NaverAuthManager(naver_id, naver_password)
    try:
        headers = auth_manager.get_auth_headers()
        cookies = auth_manager.get_cookies()
        test_url = "https://new.smartplace.naver.com/proxy/bizadvisor/api/v3/sites/sp_311b9ba993e974/report"
        params = {
            'dimensions': 'mapped_channel_name',
            'startDate': '2025-07-01',
            'endDate': '2025-07-01',
            'metrics': 'pv',
            'sort': 'pv',
            'useIndex': 'revenue-all-channel-detail'
        }
        response = requests.get(
            test_url,
            params=params,
            headers=headers,
            cookies=cookies,
            timeout=10
        )
        if response.status_code == 200:
            logger.info("인증 테스트 성공!")
        else:
            logger.error(f"인증 테스트 실패: HTTP {response.status_code}")
    except Exception as e:
        logger.error(f"인증 테스트 중 오류: {e}")


if __name__ == "__main__":
    main()
