#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
데이터 저장 모듈
크롤링 결과를 다양한 형식으로 저장하고 관리합니다.
"""

import json
import csv
import pickle
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import asdict, is_dataclass

logger = logging.getLogger(__name__)

class DataSaver:
    """데이터 저장 클래스"""
    
    def __init__(self, 
                 output_dir: str = "data",
                 raw_dir: str = "raw",
                 processed_dir: str = "processed",
                 analyzed_dir: str = "analyzed"):
        """
        Args:
            output_dir: 출력 디렉토리
            raw_dir: 원시 데이터 디렉토리
            processed_dir: 가공된 데이터 디렉토리
            analyzed_dir: 분석된 데이터 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.raw_dir = self.output_dir / raw_dir
        self.processed_dir = self.output_dir / processed_dir
        self.analyzed_dir = self.output_dir / analyzed_dir
        
        # 디렉토리 생성
        self._create_directories()
    
    def _create_directories(self):
        """필요한 디렉토리들을 생성"""
        directories = [self.output_dir, self.raw_dir, self.processed_dir, self.analyzed_dir]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"디렉토리 확인/생성: {directory}")
    
    def save_json(self, data: Any, filename: str, directory: str = "raw") -> str:
        """JSON 형식으로 데이터 저장"""
        file_path = self._get_file_path(filename, directory, "json")
        
        try:
            # dataclass 객체를 딕셔너리로 변환
            if is_dataclass(data):
                data = asdict(data)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"JSON 데이터 저장 완료: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"JSON 데이터 저장 실패: {e}")
            raise
    
    def save_csv(self, data: List[Dict], filename: str, directory: str = "raw") -> str:
        """CSV 형식으로 데이터 저장"""
        file_path = self._get_file_path(filename, directory, "csv")
        
        try:
            if not data:
                logger.warning("저장할 데이터가 없습니다.")
                return str(file_path)
            
            # 헤더 추출
            headers = list(data[0].keys())
            
            with open(file_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(data)
            
            logger.info(f"CSV 데이터 저장 완료: {file_path} ({len(data)}행)")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"CSV 데이터 저장 실패: {e}")
            raise
    
    def save_pickle(self, data: Any, filename: str, directory: str = "raw") -> str:
        """Pickle 형식으로 데이터 저장"""
        file_path = self._get_file_path(filename, directory, "pkl")
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Pickle 데이터 저장 완료: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Pickle 데이터 저장 실패: {e}")
            raise
    
    def save_crawling_result(self, 
                           result: Dict[str, Any], 
                           store_name: str,
                           timestamp: Optional[datetime] = None) -> Dict[str, str]:
        """크롤링 결과를 저장"""
        if timestamp is None:
            timestamp = datetime.now()
        
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        base_filename = f"{store_name}_{timestamp_str}"
        
        saved_files = {}
        
        try:
            # 전체 결과를 JSON으로 저장
            result_filename = f"{base_filename}_result"
            saved_files['result'] = self.save_json(result, result_filename, "raw")
            
            # 리뷰 데이터를 CSV로 저장
            if 'new_post_list' in result and result['new_post_list']:
                reviews_filename = f"{base_filename}_reviews"
                saved_files['reviews'] = self.save_csv(result['new_post_list'], reviews_filename, "raw")
            
            # 답글 데이터를 CSV로 저장
            if 'new_post_reply_add_req_list' in result and result['new_post_reply_add_req_list']:
                replies_filename = f"{base_filename}_replies"
                saved_files['replies'] = self.save_csv(result['new_post_reply_add_req_list'], replies_filename, "raw")
            
            # 통계 정보를 JSON으로 저장
            if 'statistics' in result:
                stats_filename = f"{base_filename}_statistics"
                saved_files['statistics'] = self.save_json(result['statistics'], stats_filename, "raw")
            
            logger.info(f"크롤링 결과 저장 완료: {store_name} ({len(saved_files)}개 파일)")
            return saved_files
            
        except Exception as e:
            logger.error(f"크롤링 결과 저장 실패: {e}")
            raise
    
    def save_batch_results(self, 
                          results: List[Dict[str, Any]], 
                          batch_name: str,
                          timestamp: Optional[datetime] = None) -> Dict[str, str]:
        """배치 크롤링 결과를 저장"""
        if timestamp is None:
            timestamp = datetime.now()
        
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        base_filename = f"{batch_name}_{timestamp_str}"
        
        # 모든 리뷰와 답글을 하나의 파일로 통합
        all_reviews = []
        all_replies = []
        all_statistics = []
        
        for result in results:
            if 'new_post_list' in result:
                all_reviews.extend(result['new_post_list'])
            
            if 'new_post_reply_add_req_list' in result:
                all_replies.extend(result['new_post_reply_add_req_list'])
            
            if 'statistics' in result:
                all_statistics.append(result['statistics'])
        
        saved_files = {}
        
        try:
            # 통합된 리뷰 데이터 저장
            if all_reviews:
                reviews_filename = f"{base_filename}_all_reviews"
                saved_files['all_reviews'] = self.save_csv(all_reviews, reviews_filename, "processed")
            
            # 통합된 답글 데이터 저장
            if all_replies:
                replies_filename = f"{base_filename}_all_replies"
                saved_files['all_replies'] = self.save_csv(all_replies, replies_filename, "processed")
            
            # 통계 데이터 저장
            if all_statistics:
                stats_filename = f"{base_filename}_all_statistics"
                saved_files['all_statistics'] = self.save_json(all_statistics, stats_filename, "processed")
            
            # 배치 요약 정보 저장
            batch_summary = {
                "batch_name": batch_name,
                "timestamp": timestamp.isoformat(),
                "total_results": len(results),
                "total_reviews": len(all_reviews),
                "total_replies": len(all_replies),
                "successful_results": len([r for r in results if r.get('success', False)]),
                "failed_results": len([r for r in results if not r.get('success', False)])
            }
            
            summary_filename = f"{base_filename}_summary"
            saved_files['summary'] = self.save_json(batch_summary, summary_filename, "processed")
            
            logger.info(f"배치 결과 저장 완료: {batch_name} ({len(saved_files)}개 파일)")
            return saved_files
            
        except Exception as e:
            logger.error(f"배치 결과 저장 실패: {e}")
            raise
    
    def load_json(self, filename: str, directory: str = "raw") -> Any:
        """JSON 파일 로드"""
        file_path = self._get_file_path(filename, directory, "json")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.debug(f"JSON 데이터 로드 완료: {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"JSON 데이터 로드 실패: {e}")
            raise
    
    def load_csv(self, filename: str, directory: str = "raw") -> List[Dict]:
        """CSV 파일 로드"""
        file_path = self._get_file_path(filename, directory, "csv")
        
        try:
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                data = list(reader)
            
            logger.debug(f"CSV 데이터 로드 완료: {file_path} ({len(data)}행)")
            return data
            
        except Exception as e:
            logger.error(f"CSV 데이터 로드 실패: {e}")
            raise
    
    def load_pickle(self, filename: str, directory: str = "raw") -> Any:
        """Pickle 파일 로드"""
        file_path = self._get_file_path(filename, directory, "pkl")
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            logger.debug(f"Pickle 데이터 로드 완료: {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Pickle 데이터 로드 실패: {e}")
            raise
    
    def _get_file_path(self, filename: str, directory: str, extension: str) -> Path:
        """파일 경로 생성"""
        if directory == "raw":
            target_dir = self.raw_dir
        elif directory == "processed":
            target_dir = self.processed_dir
        elif directory == "analyzed":
            target_dir = self.analyzed_dir
        else:
            target_dir = self.output_dir
        
        # 확장자가 이미 포함되어 있지 않으면 추가
        if not filename.endswith(f".{extension}"):
            filename = f"{filename}.{extension}"
        
        return target_dir / filename
    
    def list_files(self, directory: str = "raw", pattern: str = "*") -> List[str]:
        """디렉토리의 파일 목록 반환"""
        if directory == "raw":
            target_dir = self.raw_dir
        elif directory == "processed":
            target_dir = self.processed_dir
        elif directory == "analyzed":
            target_dir = self.analyzed_dir
        else:
            target_dir = self.output_dir
        
        files = []
        for file_path in target_dir.glob(pattern):
            if file_path.is_file():
                files.append(str(file_path))
        
        return sorted(files)
    
    def get_file_info(self, filename: str, directory: str = "raw") -> Dict[str, Any]:
        """파일 정보 반환"""
        file_path = self._get_file_path(filename, directory, "")
        
        if not file_path.exists():
            return {"exists": False}
        
        stat = file_path.stat()
        return {
            "exists": True,
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime),
            "modified": datetime.fromtimestamp(stat.st_mtime),
            "path": str(file_path)
        }
    
    def cleanup_old_files(self, 
                         directory: str = "raw", 
                         days: int = 30,
                         pattern: str = "*") -> int:
        """오래된 파일 정리"""
        if directory == "raw":
            target_dir = self.raw_dir
        elif directory == "processed":
            target_dir = self.processed_dir
        elif directory == "analyzed":
            target_dir = self.analyzed_dir
        else:
            target_dir = self.output_dir
        
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        deleted_count = 0
        
        for file_path in target_dir.glob(pattern):
            if file_path.is_file():
                if file_path.stat().st_mtime < cutoff_date:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        logger.info(f"오래된 파일 삭제: {file_path}")
                    except Exception as e:
                        logger.error(f"파일 삭제 실패: {file_path}, {e}")
        
        logger.info(f"파일 정리 완료: {deleted_count}개 파일 삭제")
        return deleted_count

def main():
    """테스트용 메인 함수"""
    # 데이터 저장기 생성
    saver = DataSaver()
    
    # 테스트 데이터
    test_data = {
        "store_name": "테스트 매장",
        "total_reviews": 10,
        "total_replies": 5,
        "success": True,
        "timestamp": datetime.now().isoformat()
    }
    
    test_reviews = [
        {"id": 1, "content": "좋은 매장입니다.", "rating": 5},
        {"id": 2, "content": "괜찮습니다.", "rating": 4},
        {"id": 3, "content": "보통입니다.", "rating": 3}
    ]
    
    # 데이터 저장 테스트
    print("=== 데이터 저장 테스트 ===")
    
    # JSON 저장
    json_file = saver.save_json(test_data, "test_data")
    print(f"JSON 저장: {json_file}")
    
    # CSV 저장
    csv_file = saver.save_csv(test_reviews, "test_reviews")
    print(f"CSV 저장: {csv_file}")
    
    # 크롤링 결과 저장
    crawling_result = {
        "success": True,
        "store_name": "테스트 매장",
        "new_post_list": test_reviews,
        "new_post_reply_add_req_list": [],
        "statistics": {"total_requests": 100, "success_count": 95}
    }
    
    saved_files = saver.save_crawling_result(crawling_result, "테스트매장")
    print(f"크롤링 결과 저장: {saved_files}")
    
    # 파일 목록 확인
    files = saver.list_files("raw", "test_*")
    print(f"저장된 파일들: {files}")

if __name__ == "__main__":
    main()
