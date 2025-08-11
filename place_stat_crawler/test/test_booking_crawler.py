#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NaverBookingStatCrawler 기능 테스트 스크립트
- 단일 클라이언트에 대해 특정 날짜의 예약 통계를 수집하여 정상 작동 여부를 확인합니다.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
import logging

# 프로젝트 루트 경로를 sys.path에 추가하여 모듈을 찾을 수 있도록 함
project_root = Path(__file__).resolve().parents[1] # .parents[2] -> .parents[1]로 수정
sys.path.insert(0, str(project_root))

from scripts.util.config import get_config_manager
from scripts.crawler.naver_booking_stat_crawler import NaverBookingStatCrawler
from scripts.crawler.naver_place_pv_crawler_base import ApiCallError

# 기본 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_booking_test():
    """예약 통계 수집기 테스트 실행"""
    logger.info("=" * 60)
    logger.info("📊 네이버 예약 통계 크롤러(NaverBookingStatCrawler) 단독 테스트를 시작합니다.")
    logger.info("=" * 60)

    # 1. 설정 및 클라이언트 선택
    config_manager = get_config_manager()
    client_info = config_manager.get_selected_client_config()
    if not client_info:
        logger.error("❌ 클라이언트를 선택하지 못했습니다. 테스트를 종료합니다.")
        return

    # BOOKING_KEY 설정 확인
    if not client_info.booking_key:
        logger.error(f"❌ '{client_info.name}' 클라이언트의 BOOKING_KEY가 .env에 설정되지 않았습니다. 테스트를 종료합니다.")
        return

    # 2. 크롤러 초기화
    try:
        auth_config = config_manager.get_auth_config()
        crawler = NaverBookingStatCrawler(client_info, auth_config)
        logger.info(f"✅ '{client_info.name}' 클라이언트로 크롤러를 초기화했습니다.")
    except Exception as e:
        logger.error(f"❌ 크롤러 초기화 중 오류 발생: {e}", exc_info=True)
        return

    # 3. 테스트할 날짜 설정 (2025-08-08)
    target_date = '2025-08-08'
    logger.info(f"📅 테스트 대상 날짜: {target_date}")

    # 4. 데이터 수집 테스트
    try:
        logger.info("-" * 60)
        logger.info("📢 예약 통계 데이터 수집을 테스트합니다...")
        
        booking_data = crawler.fetch_booking_data_for_date(target_date)
        
        if booking_data:
            logger.info("✔️ 예약 데이터 수집 성공!")
            page_visits = booking_data.get('page_visits', [])
            booking_requests = booking_data.get('booking_requests', [])
            channel_stats = booking_data.get('channel_stats', [])

            pv_count = page_visits[0]['count'] if page_visits else 0
            req_count = booking_requests[0]['count'] if booking_requests else 0
            
            logger.info(f"  - 예약 페이지 유입 수: {pv_count}")
            logger.info(f"  - 예약 신청 수: {req_count}")
            logger.info(f"  - 유입 채널 수: {len(channel_stats)}")

            if channel_stats:
                for item in channel_stats[:5]:
                    logger.info(f"    - 채널: {item.get('channel_name', 'N/A')}, 유입: {item.get('count', 0)}")
                if len(channel_stats) > 5:
                    logger.info("    ...")
        else:
            logger.warning("⚠️ 예약 데이터가 수집되지 않았거나 데이터가 없습니다.")

    except ApiCallError as e:
        logger.error(f"❌ API 인증 오류가 발생했습니다. .env 파일의 _BOOKING_COOKIE 값을 확인해주세요.")
        logger.error(f"   > {e}")
    except Exception as e:
        logger.error(f"❌ 데이터 수집 중 예상치 못한 오류 발생: {e}", exc_info=True)

    logger.info("=" * 60)
    logger.info("📊 테스트가 완료되었습니다.")
    logger.info("=" * 60)

if __name__ == "__main__":
    run_booking_test()
