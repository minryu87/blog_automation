#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NaverStatCrawler 기능 테스트 스크립트
- 단일 클라이언트에 대해 특정 날짜의 플레이스 PV 통계(채널, 키워드)를 수집하여 정상 작동 여부를 확인합니다.
"""

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import logging

# 프로젝트 루트 경로를 sys.path에 추가하여 모듈을 찾을 수 있도록 함
project_root = Path(__file__).resolve().parents[1] # .parents[2] -> .parents[1]로 수정
sys.path.insert(0, str(project_root))

from scripts.util.config import get_config_manager
from scripts.crawler.naver_place_pv_stat_crawler import NaverStatCrawler
from scripts.crawler.naver_place_pv_crawler_base import ApiCallError

# 기본 로깅 설정은 config.py에 위임합니다.
logger = logging.getLogger(__name__)

def run_place_test():
    """플레이스 통계 수집기 테스트 실행"""
    logger.info("=" * 60)
    logger.info("📊 네이버 플레이스 통계 크롤러(NaverStatCrawler) 단독 테스트를 시작합니다.")
    logger.info("=" * 60)

    # 1. 설정 및 클라이언트 선택
    config_manager = get_config_manager()
    client_info = config_manager.get_selected_client_config()
    if not client_info:
        logger.error("❌ 클라이언트를 선택하지 못했습니다. 테스트를 종료합니다.")
        return

    # 2. 크롤러 초기화
    try:
        auth_config = config_manager.get_auth_config()
        crawler = NaverStatCrawler(client_info, auth_config)
        logger.info(f"✅ '{client_info.name}' 클라이언트로 크롤러를 초기화했습니다.")
    except Exception as e:
        logger.error(f"❌ 크롤러 초기화 중 오류 발생: {e}", exc_info=True)
        return

    # 3. 테스트할 날짜 설정 (어제)
    target_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    logger.info(f"📅 테스트 대상 날짜: {target_date}")

    # 4. 데이터 수집 테스트
    try:
        # 4-1. 채널 데이터 테스트
        logger.info("-" * 60)
        logger.info("📢 [1/2] 채널(Channel) 데이터 수집을 테스트합니다...")
        channel_data = crawler.fetch_channel_data_for_date(target_date)
        
        if channel_data:
            logger.info(f"✔️ 채널 데이터 수집 성공! (총 {len(channel_data)}개 채널)")
            for item in channel_data[:5]: # 상위 5개만 출력
                logger.info(f"  - 채널: {item.get('mapped_channel_name', 'N/A')}, PV: {item.get('pv', 0)}")
            if len(channel_data) > 5:
                logger.info("  ...")
        else:
            logger.warning("⚠️ 채널 데이터가 수집되지 않았거나 데이터가 없습니다.")

        # 4-2. 키워드 데이터 테스트
        logger.info("-" * 60)
        logger.info("📢 [2/2] 키워드(Keyword) 데이터 수집을 테스트합니다...")
        keyword_data = crawler.fetch_keyword_data_for_date(target_date)

        if keyword_data:
            logger.info(f"✔️ 키워드 데이터 수집 성공! (총 {len(keyword_data)}개 키워드)")
            for item in keyword_data[:5]: # 상위 5개만 출력
                logger.info(f"  - 키워드: {item.get('ref_keyword', 'N/A')}, PV: {item.get('pv', 0)}")
            if len(keyword_data) > 5:
                logger.info("  ...")
        else:
            logger.warning("⚠️ 키워드 데이터가 수집되지 않았거나 데이터가 없습니다.")

    except ApiCallError as e:
        logger.error(f"❌ API 인증 오류가 발생했습니다. .env 파일의 _PLACE_AUTH, _PLACE_COOKIE 값을 확인해주세요.")
        logger.error(f"   > {e}")
    except Exception as e:
        logger.error(f"❌ 데이터 수집 중 예상치 못한 오류 발생: {e}", exc_info=True)

    logger.info("=" * 60)
    logger.info("📊 테스트가 완료되었습니다.")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_place_test()
