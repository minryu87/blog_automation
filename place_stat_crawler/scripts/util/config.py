#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
설정 관리 모듈
크롤링 설정, API 설정, 로깅 설정 등을 중앙에서 관리합니다.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field, asdict

# .env 자동 로드
try:
    from dotenv import load_dotenv
    # 1. 스크립트 기준 상대 경로: .../place_stat_crawler/.env
    env_path = Path(__file__).resolve().parents[2] / '.env'
    
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        # 로거 설정 전이므로 print 사용
        print(f"[INFO] .env 파일을 다음 경로에서 로드했습니다: {env_path}")
    else:
        print(f"[WARNING] .env 파일을 다음 경로에서 찾을 수 없습니다: {env_path}")
        # 2. 프로젝트 루트(현재 실행 위치)에서 한번 더 찾아보기
        root_env_path = Path.cwd() / '.env'
        if root_env_path.exists() and root_env_path != env_path:
             load_dotenv(dotenv_path=root_env_path)
             print(f"[INFO] .env 파일을 다음 경로에서 로드했습니다: {root_env_path}")
        else:
             print(f"[WARNING] 프로젝트 루트({Path.cwd()})에서도 .env 파일을 찾을 수 없습니다.")

except ImportError:
    print("[WARNING] python-dotenv가 설치되지 않았습니다. .env 파일 로드를 건너뜁니다.")
except Exception as e:
    print(f"[ERROR] .env 파일 로딩 중 오류 발생: {e}")


logger = logging.getLogger(__name__)

@dataclass
class CrawlerConfig:
    """크롤러 설정"""
    crawling_size: int = 20
    initial_crawling_total_size: int = 1000
    crawling_sleep_time: float = 1.0
    wait_time_for_too_many_requests: int = 30
    max_crawling_retry_count: int = 3
    review_search_deadline_period_months: int = 1

@dataclass
class ClientInfo:
    """클라이언트별 인증 및 설정 정보"""
    name: str
    naver_url: str = ""
    id: str = ""
    pw: str = ""
    place_cookie: Optional[str] = None
    place_auth: Optional[str] = None
    booking_cookie: Optional[str] = None
    booking_key: Optional[str] = None

@dataclass
class AuthConfig:
    """인증 설정"""
    token_expiry_hours: int = 12
    clients: Dict[str, ClientInfo] = field(default_factory=dict)
    selected_client: Optional[str] = None
    auth_file_path_template: str = "{client_name}_auth.json"

@dataclass
class LoggingConfig:
    """로깅 설정"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "crawler.log"

@dataclass
class Config:
    """전체 설정"""
    auth: AuthConfig = field(default_factory=AuthConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    crawler: CrawlerConfig = field(default_factory=CrawlerConfig)

class ConfigManager:
    """설정 관리 클래스"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = Config()
        self._load_environment_variables()
        self._setup_logging()
    
    def _load_environment_variables(self):
        """환경 변수에서 설정을 로드합니다."""
        try:
            client_list_str = os.getenv("CLIENT_LIST")
            if not client_list_str:
                logger.info("CLIENT_LIST 환경 변수가 설정되지 않았습니다.")
                return

            # 문자열 형태의 리스트를 실제 리스트로 안전하게 변환
            import ast
            client_names = ast.literal_eval(client_list_str)
            
            if not isinstance(client_names, list):
                logger.error("CLIENT_LIST가 올바른 리스트 형식이 아닙니다. 예: [\"CLIENT1\", \"CLIENT2\"]")
                return

            for name in client_names:
                prefix = name.upper()
                client_info = ClientInfo(
                    name=name,
                    naver_url=os.getenv(f"{prefix}_NAVER_URL", ""),
                    id=os.getenv(f"{prefix}_ID", ""),
                    pw=os.getenv(f"{prefix}_PW", ""),
                    place_cookie=os.getenv(f"{prefix}_PLACE_COOKIE"),
                    place_auth=os.getenv(f"{prefix}_PLACE_AUTH"),
                    booking_cookie=os.getenv(f"{prefix}_BOOKING_COOKIE"),
                    booking_key=os.getenv(f"{prefix}_BOOKING_KEY")
                )
                self.config.auth.clients[name] = client_info
            
            logger.info(f"환경 변수에서 {len(client_names)}개의 클라이언트 설정을 로드했습니다.")
        except Exception as e:
            logger.error(f"환경 변수 로드 중 오류 발생: {e}", exc_info=True)

    def _setup_logging(self):
        """로깅 설정"""
        log_config = self.config.logging
        log_dir = Path(log_config.file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=getattr(logging, log_config.level, logging.INFO),
            format=log_config.format,
            handlers=[
                logging.FileHandler(log_config.file_path, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        logger.info(f"로깅 설정 완료: 레벨={log_config.level}")

    def get_selected_client_config(self) -> Optional[ClientInfo]:
        """사용자에게 클라이언트 선택을 요청하여 선택된 클라이언트 정보를 반환합니다."""
        if not self.config.auth.clients:
            logger.error("설정된 클라이언트가 없습니다. .env 파일을 확인해주세요.")
            return None

        clients = list(self.config.auth.clients.values())
        if len(clients) == 1:
            selected_client = clients[0]
            self.config.auth.selected_client = selected_client.name
            logger.info(f"단일 클라이언트 '{selected_client.name}'가 자동으로 선택되었습니다.")
            return selected_client

        print("\n[ 클라이언트 선택 ]")
        for i, client in enumerate(clients):
            print(f"  {i+1}. {client.name}")
        
        while True:
            try:
                choice = input(f"\n> 실행할 업체의 번호를 입력하세요 (1-{len(clients)}): ")
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(clients):
                    selected_client = clients[choice_idx]
                    self.config.auth.selected_client = selected_client.name
                    logger.info(f"클라이언트 '{selected_client.name}'가 선택되었습니다.")
                    return selected_client
                else:
                    print("  [!] 번호가 범위를 벗어났습니다.")
            except (ValueError, IndexError):
                print("  [!] 유효한 숫자를 입력해주세요.")
    
    def get_auth_config(self) -> AuthConfig:
        """인증 설정 반환"""
        return self.config.auth

# 전역 설정 매니저 인스턴스
_config_manager = None

def get_config_manager() -> ConfigManager:
    """전역 설정 매니저 인스턴스 반환"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager