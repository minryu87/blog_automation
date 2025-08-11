#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
설정 관리 모듈
크롤링 설정, API 설정, 로깅 설정 등을 중앙에서 관리합니다.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import timedelta

# .env 자동 로드 (place_stat_crawler/.env)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / '.env')
except Exception:
    pass

logger = logging.getLogger(__name__)

@dataclass
class CrawlerConfig:
    """크롤러 설정"""
    # 크롤링 기본 설정
    crawling_size: int = 20
    initial_crawling_total_size: int = 1000
    crawling_sleep_time: float = 1.0
    wait_time_for_too_many_requests: int = 30
    max_crawling_retry_count: int = 3
    review_search_deadline_period_months: int = 1
    
    # 초기값 설정
    initial_value_score: int = 0
    initial_value_isinsulting: bool = False
    initial_value_isdefamatory: bool = False
    initial_value_post_word_add_req_list: list = None
    
    def __post_init__(self):
        if self.initial_value_post_word_add_req_list is None:
            self.initial_value_post_word_add_req_list = []

@dataclass
class ClientInfo:
    """클라이언트별 인증 및 설정 정보"""
    name: str
    naver_url: str = ""
    id: str = ""
    pw: str = ""
    cookie: Optional[str] = None
    auth_token: Optional[str] = None
    booking_id: Optional[str] = None


@dataclass
class AuthConfig:
    """인증 설정"""
    token_expiry_hours: int = 12
    clients: Dict[str, ClientInfo] = None
    selected_client: str = None
    auth_file_path_template: str = "{client_name}_auth.json"
    
    def __post_init__(self):
        if self.clients is None:
            self.clients = {}


@dataclass
class LoggingConfig:
    """로깅 설정"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "crawler.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5

@dataclass
class Config:
    """전체 설정"""
    auth: AuthConfig = None
    logging: LoggingConfig = None
    
    def __post_init__(self):
        if self.auth is None:
            self.auth = AuthConfig()
        if self.logging is None:
            self.logging = LoggingConfig()

class ConfigManager:
    """설정 관리 클래스"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = Config()
        self._load_environment_variables()
        self._load_config_file()
        self._setup_logging()
    
    def _load_environment_variables(self):
        """환경 변수에서 클라이언트별 설정 로드"""
        clients_data = {}
        # 클라이언트별 환경 변수 접미사 (_AUTH_TOKEN 및 _AUTH 모두 지원)
        client_suffixes = ['_NAVER_URL', '_ID', '_PW', '_COOKIE', '_AUTH_TOKEN', '_AUTH', '_BOOKING_ID']

        for key, value in os.environ.items():
            for suffix in client_suffixes:
                if key.endswith(suffix):
                    client_name = key[:-len(suffix)]
                    # 접미사에 따른 명시적 키 매핑
                    if suffix == '_NAVER_URL':
                        mapped_key = 'naver_url'
                    elif suffix == '_ID':
                        mapped_key = 'id'
                    elif suffix == '_PW':
                        mapped_key = 'pw'
                    elif suffix == '_COOKIE':
                        mapped_key = 'cookie'
                    elif suffix in ('_AUTH_TOKEN', '_AUTH'):
                        mapped_key = 'auth_token'
                    elif suffix == '_BOOKING_ID':
                        mapped_key = 'booking_id'
                    else:
                        mapped_key = suffix[1:].lower()

                    if client_name not in clients_data:
                        clients_data[client_name] = {'name': client_name}

                    clients_data[client_name][mapped_key] = value
                    break
        
        # 불필요 키 제거 및 ClientInfo에 맞춰 생성
        valid_clients = {}
        for name, data in clients_data.items():
            if not data.get('id'):
                continue
            ci = ClientInfo(
                name=data.get('name', name),
                naver_url=data.get('naver_url', ''),
                id=data.get('id', ''),
                pw=data.get('pw', ''),
                cookie=data.get('cookie', ''),
                auth_token=data.get('auth_token', ''),
                booking_id=data.get('booking_id', ''),
            )
            valid_clients[name] = ci
        self.config.auth.clients = valid_clients
        
        if self.config.auth.clients:
            logger.info(f"{len(self.config.auth.clients)}개의 클라이언트 설정을 로드했습니다: {list(self.config.auth.clients.keys())}")
        
        # 기존의 다른 환경 변수 로드
        log_level = os.getenv('LOG_LEVEL', self.config.logging.level)
        self.config.logging.level = log_level.upper()
    
    def _load_config_file(self):
        """설정 파일에서 설정 로드"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # 설정 업데이트
                self._update_config_from_dict(config_data)
                logger.info(f"설정 파일 로드 완료: {self.config_file}")
            else:
                logger.info("설정 파일이 없습니다. 기본 설정을 사용합니다.")
                
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {e}")
    
    def _update_config_from_dict(self, config_data: Dict[str, Any]):
        """딕셔너리에서 설정 업데이트"""
        for section, data in config_data.items():
            if hasattr(self.config, section):
                section_config = getattr(self.config, section)
                if isinstance(section_config, object):
                    for key, value in data.items():
                        if hasattr(section_config, key):
                            setattr(section_config, key, value)
    
    def _setup_logging(self):
        """로깅 설정"""
        log_config = self.config.logging
        
        # 로그 디렉토리 생성
        log_dir = Path(log_config.file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 로깅 설정
        logging.basicConfig(
            level=getattr(logging, log_config.level),
            format=log_config.format,
            handlers=[
                logging.FileHandler(log_config.file_path, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        logger.info(f"로깅 설정 완료: 레벨={log_config.level}")
    
    def save_config(self):
        """현재 설정을 파일에 저장"""
        try:
            config_dict = asdict(self.config)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, ensure_ascii=False, indent=2)
            
            logger.info(f"설정 저장 완료: {self.config_file}")
            
        except Exception as e:
            logger.error(f"설정 저장 실패: {e}")

    def select_client(self) -> Optional[str]:
        """사용자에게 클라이언트 선택을 요청"""
        clients = self.config.auth.clients
        if not clients:
            logger.error("설정된 클라이언트가 없습니다. .env 파일을 확인해주세요.")
            return None
        
        if len(clients) == 1:
            selected_client_name = list(clients.keys())[0]
            self.config.auth.selected_client = selected_client_name
            logger.info(f"단일 클라이언트 '{selected_client_name}'가 자동으로 선택되었습니다.")
            return selected_client_name
            
        print("\n[ 클라이언트 선택 ]")
        client_list = sorted(list(clients.keys()))
        for i, name in enumerate(client_list, 1):
            print(f"  {i}. {name}")
            
        while True:
            try:
                choice = input(f"\n> 실행할 업체의 번호를 입력하세요 (1-{len(client_list)}): ")
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(client_list):
                    selected_client_name = client_list[choice_idx]
                    self.config.auth.selected_client = selected_client_name
                    logger.info(f"클라이언트 '{selected_client_name}'가 선택되었습니다.")
                    return selected_client_name
                else:
                    print(f"  [!] 잘못된 번호입니다. 1에서 {len(client_list)} 사이의 숫자를 입력해주세요.")
            except (ValueError, IndexError):
                print("  [!] 유효한 숫자를 입력해주세요.")

    def get_selected_client_config(self) -> Optional[ClientInfo]:
        """선택된 클라이언트의 설정을 반환"""
        selected_name = self.config.auth.selected_client
        # 선택되지 않았다면 선택 프롬프트 표시
        if not selected_name:
            selected_name = self.select_client()

        if selected_name:
            return self.config.auth.clients.get(selected_name)
            
        return None
    
    
    def get_crawler_config(self) -> CrawlerConfig:
        """크롤러 설정 반환"""
        return self.config.crawler
    
    def get_auth_config(self) -> AuthConfig:
        """인증 설정 반환"""
        return self.config.auth
    
    def get_logging_config(self) -> LoggingConfig:
        """로깅 설정 반환"""
        return self.config.logging
    
    def update_crawler_config(self, **kwargs):
        """크롤러 설정 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self.config.crawler, key):
                setattr(self.config.crawler, key, value)
    
    def update_auth_config(self, **kwargs):
        """인증 설정 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self.config.auth, key):
                setattr(self.config.auth, key, value)
    
    def create_default_config(self):
        """기본 설정 파일 생성"""
        default_config = {
            "crawler": {
                "crawling_size": 20,
                "initial_crawling_total_size": 1000,
                "crawling_sleep_time": 1.0,
                "wait_time_for_too_many_requests": 30,
                "max_crawling_retry_count": 3,
                "review_search_deadline_period_months": 1,
                "initial_value_score": 0,
                "initial_value_isinsulting": False,
                "initial_value_isdefamatory": False,
                "initial_value_post_word_add_req_list": []
            },
            "auth": {
                "naver_id": "",
                "naver_password": "",
                "auth_file_path": "naver_auth.json",
                "token_expiry_hours": 12,
                "auto_refresh": True
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file_path": "crawler.log",
                "max_file_size": 10485760,
                "backup_count": 5
            }
        }
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"기본 설정 파일 생성: {self.config_file}")
    
    def validate_config(self) -> bool:
        """설정 유효성 검사"""
        errors = []
        
        # 클라이언트 설정 검사
        selected_client_config = self.get_selected_client_config()
        
        if not selected_client_config:
            # select_client() 내부에서 이미 에러 로그를 출력하므로 여기서는 간단히 반환
            return False
        
        if not selected_client_config.id:
            errors.append(f"선택된 클라이언트 '{selected_client_config.name}'의 ID가 설정되지 않았습니다.")
        
        if not selected_client_config.pw:
            errors.append(f"선택된 클라이언트 '{selected_client_config.name}'의 PW가 설정되지 않았습니다.")
        
        if not selected_client_config.naver_url:
            errors.append(f"선택된 클라이언트 '{selected_client_config.name}'의 NAVER_URL이 설정되지 않았습니다.")
        
        # 크롤링 설정 검사
        if self.config.crawler.crawling_size <= 0:
            errors.append("crawling_size는 0보다 커야 합니다.")
        
        if self.config.crawler.initial_crawling_total_size <= 0:
            errors.append("initial_crawling_total_size는 0보다 커야 합니다.")
        
        # 로깅 레벨 검사
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.config.logging.level not in valid_levels:
            errors.append(f"잘못된 로깅 레벨: {self.config.logging.level}")
        
        if errors:
            for error in errors:
                logger.error(f"설정 오류: {error}")
            return False
        
        logger.info("설정 유효성 검사 통과")
        return True

# 전역 설정 매니저 인스턴스
_config_manager = None

def get_config_manager() -> ConfigManager:
    """전역 설정 매니저 인스턴스 반환"""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager()
    
    return _config_manager

def main():
    """테스트용 메인 함수"""
    config_manager = get_config_manager()
    
    print("=== 설정 매니저 테스트 ===")
    
    # 설정 유효성 검사
    if config_manager.validate_config():
        print("✅ 설정 유효성 검사 통과")
    else:
        print("❌ 설정 유효성 검사 실패")
    
    # 설정 정보 출력
    print(f"\n크롤러 설정:")
    crawler_config = config_manager.get_crawler_config()
    print(f"  크롤링 크기: {crawler_config.crawling_size}")
    print(f"  대기 시간: {crawler_config.crawling_sleep_time}초")
    
    print(f"\n인증 설정:")
    auth_config = config_manager.get_auth_config()
    print(f"  네이버 ID: {auth_config.selected_client}의 정보 사용" if auth_config.selected_client else "  클라이언트 선택되지 않음")
    print(f"  토큰 만료 시간: {auth_config.token_expiry_hours}시간")
    
    # 기본 설정 파일 생성
    config_manager.create_default_config()
    print(f"\n기본 설정 파일 생성 완료: {config_manager.config_file}")

if __name__ == "__main__":
    main()
