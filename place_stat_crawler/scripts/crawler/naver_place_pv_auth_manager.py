#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
네이버 인증 토큰 자동 관리 시스템
"""

import json
import time
import requests
import pickle
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import logging
from urllib.parse import quote

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NaverAuthManager:
    """네이버 인증 토큰 자동 관리 클래스"""
    
    def __init__(self, 
                 naver_id: str, 
                 naver_password: str,
                 auth_file_path: str = "naver_auth.json",
                 token_expiry_hours: int = 12,
                 naver_login_url: str = "https://nid.naver.com/nidlogin.login",
                 client_name: str = "default"):
        """
        Args:
            naver_id: 네이버 아이디
            naver_password: 네이버 비밀번호
            auth_file_path: 인증 정보 저장 파일 경로
            token_expiry_hours: 토큰 만료 시간 (시간)
            naver_login_url: 로그인에 사용할 URL (.env의 NAVER_URL)
            client_name: 클라이언트 식별용 이름 (로그 용도)
        """
        self.naver_id = naver_id
        self.naver_password = naver_password
        self.auth_file_path = auth_file_path
        self.token_expiry_hours = token_expiry_hours
        self.client_name = client_name
        
        # 로그인 URL/리디렉션 대상 처리
        self.login_url = "https://nid.naver.com/nidlogin.login"
        self.post_login_url = None
        if naver_login_url:
            if "nid.naver.com" in naver_login_url and "nidlogin" in naver_login_url:
                self.login_url = naver_login_url
            else:
                # 로그인 페이지로 이동 후, 성공 시 지정 URL로 리디렉션
                self.post_login_url = naver_login_url
                self.login_url = f"https://nid.naver.com/nidlogin.login?url={quote(self.post_login_url, safe='')}"
        
        # 인증 정보
        self.auth_token = None
        self.cookies = None
        self.last_refresh_time = None
        
        # 브라우저 옵션
        self.chrome_options = Options()
        headless_env = os.getenv("HEADLESS", "true").lower()
        if headless_env == "true":
            self.chrome_options.add_argument('--headless=new')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        self.chrome_options.add_argument('--disable-gpu')
        self.chrome_options.add_argument('--window-size=1280,900')
        self.chrome_options.add_argument('--lang=ko-KR')
        self.chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        # 자동화 탐지 완화 옵션
        self.chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
        self.chrome_options.add_experimental_option('useAutomationExtension', False)
        self.chrome_options.add_argument('--disable-blink-features=AutomationControlled')

    # ---------------------- 수동 인증 관련 도우미 ----------------------
    @staticmethod
    def _parse_cookie_string(cookie_str: str) -> Dict[str, str]:
        cookies: Dict[str, str] = {}
        for part in cookie_str.split(';'):
            part = part.strip()
            if not part:
                continue
            if '=' in part:
                name, value = part.split('=', 1)
                cookies[name.strip()] = value.strip()
        return cookies

    def prompt_and_set_manual_auth(self) -> bool:
        """콘솔 프롬프트로 Authorization 토큰과 쿠키를 입력받아 설정"""
        print("\n[ 수동 인증 입력 ]")
        print("- 브라우저 개발자도구 또는 기존 저장값에서 Authorization 토큰과 Cookie를 복사해 붙여넣으세요.")
        token_input = input("Authorization (예: Bearer x.y.z 또는 x.y.z): ").strip()
        cookie_input = input("Cookie (예: NNB=...; NID_AUT=...; NID_SES=...): ").strip()
        if not token_input or not cookie_input:
            logger.error("토큰 또는 쿠키가 입력되지 않았습니다.")
            return False
        if not token_input.lower().startswith("bearer "):
            token_input = f"Bearer {token_input}"
        cookies = self._parse_cookie_string(cookie_input)
        if not cookies:
            logger.error("쿠키 파싱에 실패했습니다. 형식을 확인하세요.")
            return False
        self.manual_auth_setup(token_input, cookies)
        return True
    # ------------------------------------------------------------------
    
    def load_auth_from_file(self) -> bool:
        """파일에서 인증 정보 로드"""
        try:
            if not os.path.exists(self.auth_file_path):
                logger.info("인증 파일이 존재하지 않습니다.")
                return False
            
            with open(self.auth_file_path, 'r', encoding='utf-8') as f:
                auth_data = json.load(f)
            
            # 만료 시간 확인
            last_refresh = datetime.fromisoformat(auth_data.get('last_refresh_time', '1970-01-01T00:00:00'))
            expiry_time = last_refresh + timedelta(hours=self.token_expiry_hours)
            
            if datetime.now() > expiry_time:
                logger.info("저장된 토큰이 만료되었습니다.")
                return False
            
            self.auth_token = auth_data.get('auth_token')
            self.cookies = auth_data.get('cookies')
            self.last_refresh_time = last_refresh
            
            logger.info("인증 정보를 파일에서 로드했습니다.")
            return True
            
        except Exception as e:
            logger.error(f"인증 파일 로드 실패: {e}")
            return False
    
    def save_auth_to_file(self):
        """인증 정보를 파일에 저장"""
        try:
            auth_data = {
                'auth_token': self.auth_token,
                'cookies': self.cookies,
                'last_refresh_time': self.last_refresh_time.isoformat() if self.last_refresh_time else None
            }
            
            with open(self.auth_file_path, 'w', encoding='utf-8') as f:
                json.dump(auth_data, f, ensure_ascii=False, indent=2)
            
            logger.info("인증 정보를 파일에 저장했습니다.")
            
        except Exception as e:
            logger.error(f"인증 파일 저장 실패: {e}")
    
    def _build_driver(self):
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=self.chrome_options)
        # 탐지 완화 스크립트 주입
        try:
            driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
                'source': 'Object.defineProperty(navigator, "webdriver", {get: () => undefined})'
            })
        except Exception:
            pass
        return driver
    
    def login_with_selenium(self) -> bool:
        """Selenium을 사용한 네이버 로그인"""
        driver = None
        try:
            logger.info(f"[{self.client_name}] 네이버 로그인 시작... ({self.login_url})")
            
            driver = self._build_driver()
            driver.get(self.login_url)
            
            wait = WebDriverWait(driver, 20)
            
            # 아이디 입력 (제공된 HTML 구조 기반)
            id_input = wait.until(EC.presence_of_element_located((By.ID, "id")))
            id_input.clear()
            id_input.send_keys(self.naver_id)
            
            # 비밀번호 입력
            pw_input = wait.until(EC.presence_of_element_located((By.ID, "pw")))
            pw_input.clear()
            pw_input.send_keys(self.naver_password)
            
            # 로그인 버튼 클릭 (제공된 ID: log.login)
            try:
                login_btn = driver.find_element(By.ID, "log.login")
            except Exception:
                # 대체 셀렉터
                try:
                    login_btn = driver.find_element(By.CSS_SELECTOR, "button#log\\.login, button.btn_login.next_step, button[type='submit']")
                except Exception:
                    login_btn = None
            if not login_btn:
                raise Exception("로그인 버튼을 찾을 수 없습니다.")
            login_btn.click()
            
            # 리다이렉트 대기: 로그인 페이지 도메인 벗어날 때까지
            wait.until(lambda d: "nid.naver.com" not in d.current_url)
            
            # 필요 시 후속 URL 진입
            if self.post_login_url:
                driver.get(self.post_login_url)
                time.sleep(2)
            else:
                # 스마트플레이스 진입 시도 (토큰 확보용)
                try:
                    driver.get("https://new.smartplace.naver.com/")
                    time.sleep(2)
                except Exception:
                    pass
            
            # 쿠키 수집
            self.cookies = {}
            for cookie in driver.get_cookies():
                self.cookies[cookie['name']] = cookie['value']
            
            # Authorization 토큰 추출
            self.auth_token = self._extract_auth_token(driver)
            
            if not self.auth_token:
                logger.warning("Authorization 토큰을 찾을 수 없습니다. 수동으로 설정해주세요.")
                return False
            
            self.last_refresh_time = datetime.now()
            return True
            
        except Exception as e:
            # 디버깅용 현재 URL 로그
            try:
                logger.error(f"로그인 실패 URL: {driver.current_url}")
            except Exception:
                pass
            logger.error(f"네이버 로그인 실패: {e}")
            return False
        finally:
            if driver:
                driver.quit()
    
    def _extract_auth_token(self, driver) -> Optional[str]:
        """브라우저에서 Authorization 토큰 추출"""
        try:
            token = driver.execute_script("return localStorage.getItem('ba_access_token');")
            if token:
                return f"Bearer {token}"
            return None
        except Exception as e:
            logger.error(f"토큰 추출 실패: {e}")
            return None
    
    def refresh_auth_if_needed(self) -> bool:
        """필요시 인증 정보 갱신"""
        # 1) 파일에서 로드 시도
        if self.load_auth_from_file():
            logger.info("저장된 인증 정보를 사용합니다.")
            return True
 
        # 1.5) 환경변수 기반 클라이언트 프리픽스 값 사용 시도
        try:
            # 호출자가 설정한 client_name을 프리픽스로 간주하고 환경에서 AUTH/COOKIE를 찾음
            prefix = self.client_name.upper()
            env_auth = os.getenv(f"{prefix}_AUTH_TOKEN") or os.getenv(f"{prefix}_AUTH")
            env_cookie = os.getenv(f"{prefix}_COOKIE")
            if env_auth and env_cookie:
                if not env_auth.lower().startswith("bearer "):
                    env_auth = f"Bearer {env_auth}"
                # 쿠키 문자열을 파싱하여 dict로 변환
                cookies = self._parse_cookie_string(env_cookie)
                if cookies:
                    self.manual_auth_setup(env_auth, cookies)
                    logger.info("환경변수 기반 인증 정보를 설정했습니다.")
                    return True
        except Exception as e:
            logger.warning(f"환경변수 기반 인증 설정 실패: {e}")

        # 2) 환경설정: MANUAL_AUTH 기본 true → 수동 입력 프롬프트
        manual = os.getenv('MANUAL_AUTH', 'true').lower() == 'true'
        if manual:
            logger.info("수동 인증 입력 모드로 전환합니다.")
            if self.prompt_and_set_manual_auth():
                self.save_auth_to_file()
                return True
            return False
        
        # 3) 수동 모드가 아니면 셀레니움 로그인 시도
        logger.info("새로운 인증이 필요합니다. (Selenium 로그인 시도)")
        if self.login_with_selenium():
            self.save_auth_to_file()
            return True
        return False
    
    def get_auth_headers(self) -> Dict[str, str]:
        """인증 헤더 반환"""
        if not self.refresh_auth_if_needed():
            raise Exception("인증 정보를 가져올 수 없습니다.")
        headers = {
            'Authorization': self.auth_token,
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
            'Referer': 'https://new.smartplace.naver.com/'
        }
        return headers
    
    def get_cookies_string(self) -> str:
        """쿠키 문자열 반환"""
        if not self.refresh_auth_if_needed():
            raise Exception("인증 정보를 가져올 수 없습니다.")
        if not self.cookies:
            return ""
        return "; ".join([f"{name}={value}" for name, value in self.cookies.items()])
    
    def get_cookies(self) -> Dict[str, str]:
        """쿠키 딕셔너리 반환"""
        if not self.refresh_auth_if_needed():
            raise Exception("인증 정보를 가져올 수 없습니다.")
        return self.cookies or {}
    
    def test_auth(self) -> bool:
        """인증 정보 테스트"""
        try:
            headers = self.get_auth_headers()
            cookies = self.get_cookies_string()
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
                cookies=dict(self.cookies) if self.cookies else {},
                timeout=10
            )
            if response.status_code == 200:
                logger.info("인증 테스트 성공!")
                return True
            else:
                logger.error(f"인증 테스트 실패: HTTP {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"인증 테스트 중 오류: {e}")
            return False
    
    def manual_auth_setup(self, auth_token: Optional[str], cookies: Dict[str, str]):
        """수동으로 인증 정보 설정"""
        if auth_token:
            self.auth_token = auth_token if auth_token.startswith("Bearer ") else f"Bearer {auth_token}"
        self.cookies = cookies or {}
        self.last_refresh_time = datetime.now()
        self.save_auth_to_file()
        logger.info("수동 인증 정보가 설정되었습니다.")


def main():
    """테스트용 메인 함수"""
    naver_id = os.getenv('NAVER_ID')
    naver_password = os.getenv('NAVER_PASSWORD')
    if not naver_id or not naver_password:
        print("환경 변수 NAVER_ID와 NAVER_PASSWORD를 설정해주세요.")
        return
    auth_manager = NaverAuthManager(naver_id, naver_password)
    if auth_manager.test_auth():
        print("✅ 인증 성공!")
    else:
        print("❌ 인증 실패!")


if __name__ == "__main__":
    main()
