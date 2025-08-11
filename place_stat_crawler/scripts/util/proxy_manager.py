#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
프록시 관리 모듈
프록시 풀 관리, 통계 추적, 실패한 프록시 제거 기능을 제공합니다.
"""

import random
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class ProxyStats:
    """프록시 통계 정보"""
    success_count: int = 0
    failure_count: int = 0
    last_used: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    
    @property
    def total_requests(self) -> int:
        return self.success_count + self.failure_count
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.success_count / self.total_requests
    
    @property
    def is_healthy(self) -> bool:
        """프록시가 건강한지 판단"""
        if self.total_requests < 5:  # 최소 요청 수 미만
            return True
        
        if self.success_rate < 0.3:  # 성공률 30% 미만
            return False
        
        if self.last_failure and self.last_success:
            if self.last_failure > self.last_success:
                # 최근 실패가 성공보다 최신인 경우
                time_since_failure = datetime.now() - self.last_failure
                if time_since_failure < timedelta(minutes=30):
                    return False
        
        return True

class ProxyManager:
    """프록시 관리 클래스"""
    
    def __init__(self, 
                 proxy_list: List[str] = None,
                 max_failures: int = 5,
                 health_check_interval: int = 300):
        """
        Args:
            proxy_list: 프록시 목록 (host:port 형식)
            max_failures: 최대 실패 횟수
            health_check_interval: 건강 체크 간격 (초)
        """
        self.proxy_list = proxy_list or self._get_default_proxies()
        self.max_failures = max_failures
        self.health_check_interval = health_check_interval
        
        # 프록시 통계
        self.proxy_stats: Dict[str, ProxyStats] = {}
        self.last_health_check = datetime.now()
        
        # 초기화
        self._init_proxy_stats()
        logger.info(f"프록시 매니저 초기화 완료: {len(self.proxy_list)}개 프록시")
    
    def _get_default_proxies(self) -> List[str]:
        """기본 프록시 목록 반환 (실제 환경에서는 외부 API나 설정 파일에서 로드)"""
        # 예시 프록시 목록 (실제 사용시에는 유효한 프록시로 교체)
        return [
            # "proxy1.example.com:8080",
            # "proxy2.example.com:8080",
            # "proxy3.example.com:8080",
        ]
    
    def _init_proxy_stats(self):
        """프록시 통계 초기화"""
        for proxy in self.proxy_list:
            self.proxy_stats[proxy] = ProxyStats()
    
    def get_random_proxy(self) -> Optional[str]:
        """건강한 프록시 중에서 랜덤 선택"""
        self._health_check_if_needed()
        
        healthy_proxies = [
            proxy for proxy in self.proxy_list
            if self.proxy_stats[proxy].is_healthy
        ]
        
        if not healthy_proxies:
            logger.warning("사용 가능한 건강한 프록시가 없습니다.")
            return None
        
        selected_proxy = random.choice(healthy_proxies)
        self.proxy_stats[selected_proxy].last_used = datetime.now()
        
        logger.debug(f"프록시 선택: {selected_proxy}")
        return selected_proxy
    
    def update_proxy_stats(self, proxy: str, success: bool):
        """프록시 사용 결과 업데이트"""
        if proxy not in self.proxy_stats:
            logger.warning(f"알 수 없는 프록시: {proxy}")
            return
        
        stats = self.proxy_stats[proxy]
        now = datetime.now()
        
        if success:
            stats.success_count += 1
            stats.last_success = now
            logger.debug(f"프록시 성공: {proxy} (성공률: {stats.success_rate:.2%})")
        else:
            stats.failure_count += 1
            stats.last_failure = now
            logger.warning(f"프록시 실패: {proxy} (성공률: {stats.success_rate:.2%})")
        
        stats.last_used = now
    
    def remove_failed_proxy(self, proxy: str):
        """실패한 프록시 제거"""
        if proxy in self.proxy_list:
            self.proxy_list.remove(proxy)
            if proxy in self.proxy_stats:
                del self.proxy_stats[proxy]
            logger.info(f"실패한 프록시 제거: {proxy}")
    
    def _health_check_if_needed(self):
        """필요시 건강 체크 수행"""
        now = datetime.now()
        if (now - self.last_health_check).seconds < self.health_check_interval:
            return
        
        self._perform_health_check()
        self.last_health_check = now
    
    def _perform_health_check(self):
        """프록시 건강 체크 수행"""
        unhealthy_proxies = []
        
        for proxy, stats in self.proxy_stats.items():
            if not stats.is_healthy:
                unhealthy_proxies.append(proxy)
        
        if unhealthy_proxies:
            logger.info(f"건강하지 않은 프록시 발견: {len(unhealthy_proxies)}개")
            for proxy in unhealthy_proxies:
                self.remove_failed_proxy(proxy)
    
    def get_proxy_status(self) -> Dict[str, Any]:
        """프록시 상태 정보 반환"""
        total_proxies = len(self.proxy_list)
        healthy_proxies = len([
            proxy for proxy in self.proxy_list
            if self.proxy_stats[proxy].is_healthy
        ])
        
        return {
            "total_proxies": total_proxies,
            "healthy_proxies": healthy_proxies,
            "unhealthy_proxies": total_proxies - healthy_proxies,
            "proxy_details": {
                proxy: {
                    "success_count": stats.success_count,
                    "failure_count": stats.failure_count,
                    "success_rate": stats.success_rate,
                    "is_healthy": stats.is_healthy,
                    "last_used": stats.last_used.isoformat() if stats.last_used else None
                }
                for proxy, stats in self.proxy_stats.items()
            }
        }
    
    def add_proxy(self, proxy: str):
        """새 프록시 추가"""
        if proxy not in self.proxy_list:
            self.proxy_list.append(proxy)
            self.proxy_stats[proxy] = ProxyStats()
            logger.info(f"새 프록시 추가: {proxy}")
    
    def load_proxies_from_file(self, file_path: str):
        """파일에서 프록시 목록 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                proxies = [line.strip() for line in f if line.strip()]
            
            for proxy in proxies:
                self.add_proxy(proxy)
            
            logger.info(f"파일에서 {len(proxies)}개 프록시 로드: {file_path}")
            
        except Exception as e:
            logger.error(f"프록시 파일 로드 실패: {e}")
    
    def save_proxies_to_file(self, file_path: str):
        """프록시 목록을 파일에 저장"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for proxy in self.proxy_list:
                    f.write(f"{proxy}\n")
            
            logger.info(f"프록시 목록 저장: {file_path}")
            
        except Exception as e:
            logger.error(f"프록시 파일 저장 실패: {e}")

# 전역 프록시 매니저 인스턴스
_proxy_manager = None

async def get_proxy_manager() -> ProxyManager:
    """전역 프록시 매니저 인스턴스 반환"""
    global _proxy_manager
    
    if _proxy_manager is None:
        _proxy_manager = ProxyManager()
    
    return _proxy_manager

def main():
    """테스트용 메인 함수"""
    import asyncio
    
    async def test_proxy_manager():
        proxy_manager = await get_proxy_manager()
        
        # 상태 확인
        status = proxy_manager.get_proxy_status()
        print(f"프록시 상태: {status}")
        
        # 랜덤 프록시 선택
        proxy = proxy_manager.get_random_proxy()
        print(f"선택된 프록시: {proxy}")
        
        if proxy:
            # 성공/실패 시뮬레이션
            proxy_manager.update_proxy_stats(proxy, True)
            proxy_manager.update_proxy_stats(proxy, False)
            
            # 업데이트된 상태 확인
            status = proxy_manager.get_proxy_status()
            print(f"업데이트된 상태: {status}")
    
    asyncio.run(test_proxy_manager())

if __name__ == "__main__":
    main()
