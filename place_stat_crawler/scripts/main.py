#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
네이버 맵 리뷰 크롤링 메인 워크플로우
전체 크롤링 프로세스를 관리하고 실행합니다.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.crawler.place_review_crawler import NaverPlaceReviewCrawler
from scripts.util.config import get_config_manager
from scripts.util.data_saver import DataSaver
from scripts.util.proxy_manager import get_proxy_manager

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CrawlingWorkflow:
    """크롤링 워크플로우 클래스"""
    
    def __init__(self):
        self.config_manager = get_config_manager()
        self.data_saver = DataSaver()
        self.crawler = None
        self.proxy_manager = None
    
    async def initialize(self):
        """워크플로우 초기화"""
        logger.info("크롤링 워크플로우 초기화 시작")
        
        # 설정 유효성 검사
        if not self.config_manager.validate_config():
            logger.error("설정 유효성 검사 실패")
            return False
        
        # 인증 설정 가져오기
        auth_config = self.config_manager.get_auth_config()
        
        # 크롤러 생성
        self.crawler = NaverPlaceReviewCrawler(
            naver_id=auth_config.naver_id,
            naver_password=auth_config.naver_password,
            use_proxy=auth_config.use_proxy
        )
        
        # 프록시 매니저 초기화
        if auth_config.use_proxy:
            self.proxy_manager = await get_proxy_manager()
        
        logger.info("크롤링 워크플로우 초기화 완료")
        return True
    
    async def crawl_single_store(self, 
                                store_name: str, 
                                store_detail: str,
                                channel_id: str = None,
                                channel_place_id: str = None,
                                recent_crawling_at: datetime = None) -> Dict[str, Any]:
        """단일 매장 크롤링"""
        logger.info(f"단일 매장 크롤링 시작: {store_name} ({store_detail})")
        
        try:
            # 크롤링 실행
            result = await self.crawler.crawl_place_reviews(
                channel_id=channel_id or f"channel_{store_name}",
                store_name=store_name,
                store_detail=store_detail,
                channel_place_id=channel_place_id,
                store_channel_recent_crawling_at=recent_crawling_at
            )
            
            # 결과 저장
            if result.get('success', False):
                saved_files = self.data_saver.save_crawling_result(
                    result, store_name, datetime.now()
                )
                result['saved_files'] = saved_files
                logger.info(f"크롤링 완료: {store_name} - {result.get('total_reviews', 0)}개 리뷰")
            else:
                logger.error(f"크롤링 실패: {store_name} - {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"크롤링 중 오류: {store_name}, {e}")
            return {
                "success": False,
                "error": str(e),
                "store_name": store_name,
                "store_detail": store_detail
            }
    
    async def crawl_multiple_stores(self, 
                                  stores: List[Dict[str, Any]],
                                  batch_name: str = None) -> Dict[str, Any]:
        """여러 매장 배치 크롤링"""
        if not batch_name:
            batch_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"배치 크롤링 시작: {len(stores)}개 매장")
        
        results = []
        successful_count = 0
        failed_count = 0
        
        for i, store in enumerate(stores, 1):
            logger.info(f"진행률: {i}/{len(stores)} - {store.get('store_name', 'Unknown')}")
            
            result = await self.crawl_single_store(
                store_name=store.get('store_name'),
                store_detail=store.get('store_detail'),
                channel_id=store.get('channel_id'),
                channel_place_id=store.get('channel_place_id'),
                recent_crawling_at=store.get('recent_crawling_at')
            )
            
            results.append(result)
            
            if result.get('success', False):
                successful_count += 1
            else:
                failed_count += 1
            
            # 배치 간 대기
            if i < len(stores):
                await asyncio.sleep(2)
        
        # 배치 결과 저장
        batch_result = {
            "batch_name": batch_name,
            "total_stores": len(stores),
            "successful_count": successful_count,
            "failed_count": failed_count,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        saved_files = self.data_saver.save_batch_results(results, batch_name)
        batch_result['saved_files'] = saved_files
        
        logger.info(f"배치 크롤링 완료: {successful_count}개 성공, {failed_count}개 실패")
        return batch_result
    
    async def crawl_from_file(self, 
                             file_path: str,
                             batch_name: str = None) -> Dict[str, Any]:
        """파일에서 매장 목록을 읽어서 크롤링"""
        import csv
        import json
        
        logger.info(f"파일에서 매장 목록 로드: {file_path}")
        
        stores = []
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.csv':
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    stores = list(reader)
            elif file_ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    stores = json.load(f)
            else:
                raise ValueError(f"지원하지 않는 파일 형식: {file_ext}")
            
            logger.info(f"매장 목록 로드 완료: {len(stores)}개")
            
            return await self.crawl_multiple_stores(stores, batch_name)
            
        except Exception as e:
            logger.error(f"파일 로드 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }
    
    def generate_sample_data(self, output_file: str = "sample_stores.csv"):
        """샘플 매장 데이터 생성"""
        import csv
        
        sample_stores = [
            {
                "store_name": "스타벅스 강남점",
                "store_detail": "서울시 강남구 테헤란로 123",
                "channel_id": "channel_starbucks_gangnam",
                "channel_place_id": None,
                "recent_crawling_at": (datetime.now() - timedelta(days=7)).isoformat()
            },
            {
                "store_name": "올리브영 명동점",
                "store_detail": "서울시 중구 명동길 45",
                "channel_id": "channel_oliveyoung_myeongdong",
                "channel_place_id": None,
                "recent_crawling_at": (datetime.now() - timedelta(days=7)).isoformat()
            },
            {
                "store_name": "이마트 잠실점",
                "store_detail": "서울시 송파구 올림픽로 240",
                "channel_id": "channel_emart_jamsil",
                "channel_place_id": None,
                "recent_crawling_at": (datetime.now() - timedelta(days=7)).isoformat()
            }
        ]
        
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sample_stores[0].keys())
            writer.writeheader()
            writer.writerows(sample_stores)
        
        logger.info(f"샘플 데이터 생성 완료: {output_file}")
        return output_file
    
    def get_statistics(self) -> Dict[str, Any]:
        """크롤링 통계 반환"""
        if not self.crawler:
            return {"error": "크롤러가 초기화되지 않았습니다."}
        
        return self.crawler.get_statistics()

async def main():
    """메인 함수"""
    workflow = CrawlingWorkflow()
    
    # 워크플로우 초기화
    if not await workflow.initialize():
        logger.error("워크플로우 초기화 실패")
        return
    
    # 샘플 데이터 생성 (첫 실행시)
    sample_file = workflow.generate_sample_data()
    
    # 파일에서 매장 목록을 읽어서 크롤링
    batch_result = await workflow.crawl_from_file(sample_file, "sample_batch")
    
    if batch_result.get('success', False):
        print("✅ 배치 크롤링 성공!")
        print(f"📊 결과: {batch_result['successful_count']}개 성공, {batch_result['failed_count']}개 실패")
        print(f"💾 저장된 파일: {batch_result.get('saved_files', {})}")
    else:
        print(f"❌ 배치 크롤링 실패: {batch_result.get('error', 'Unknown error')}")
    
    # 통계 출력
    stats = workflow.get_statistics()
    print(f"📈 크롤링 통계: {stats}")

if __name__ == "__main__":
    asyncio.run(main())
