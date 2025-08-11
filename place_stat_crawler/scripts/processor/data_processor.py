#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
데이터 가공 클래스
수집된 원시 데이터를 가공하는 기능을 제공합니다.
"""

import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """데이터 가공 클래스"""
    
    def __init__(self, raw_data_dir: str = "../data/raw", processed_data_dir: str = "../data/processed"):
        """
        Args:
            raw_data_dir: 원시 데이터 디렉토리
            processed_data_dir: 가공된 데이터 저장 디렉토리
        """
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        
        # 디렉토리 생성
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
    
    def process_review_data(self, raw_data: List[Dict]) -> pd.DataFrame:
        """리뷰 데이터 가공"""
        if not raw_data:
            logger.warning("가공할 리뷰 데이터가 없습니다.")
            return pd.DataFrame()
        
        processed_data = []
        
        for review in raw_data:
            try:
                processed_review = {
                    'review_id': review.get('postId', ''),
                    'content': review.get('content', ''),
                    'author': review.get('author', ''),
                    'created_at': review.get('authorDtm', ''),
                    'store_name': review.get('storeName', ''),
                    'store_detail': review.get('storeDetail', ''),
                    'channel_id': review.get('channelId', ''),
                    'score': review.get('score', 0),
                    'is_insulting': review.get('isInsulting', False),
                    'is_defamatory': review.get('isDefamatory', False),
                    'type': review.get('type', 'REVIEW')
                }
                processed_data.append(processed_review)
                
            except Exception as e:
                logger.error(f"리뷰 데이터 가공 중 오류: {e}")
                continue
        
        df = pd.DataFrame(processed_data)
        
        # 날짜 컬럼 변환
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
            df['created_date'] = df['created_at'].dt.date
        
        logger.info(f"리뷰 데이터 가공 완료: {len(df)}개")
        return df
    
    def process_stat_data(self, raw_data: Dict[str, List[Dict]]) -> pd.DataFrame:
        """통계 데이터 가공"""
        if not raw_data:
            logger.warning("가공할 통계 데이터가 없습니다.")
            return pd.DataFrame()
        
        processed_data = []
        
        for date, daily_data in raw_data.items():
            for channel_data in daily_data:
                try:
                    processed_stat = {
                        'date': date,
                        'channel_name': channel_data.get('mapped_channel_name', ''),
                        'pv': channel_data.get('pv', 0),
                        'channel_id': channel_data.get('channel_id', ''),
                        'processed_at': datetime.now().isoformat()
                    }
                    processed_data.append(processed_stat)
                    
                except Exception as e:
                    logger.error(f"통계 데이터 가공 중 오류: {e}")
                    continue
        
        df = pd.DataFrame(processed_data)
        
        # 날짜 컬럼 변환
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # PV 컬럼을 숫자로 변환
        if 'pv' in df.columns:
            df['pv'] = pd.to_numeric(df['pv'], errors='coerce').fillna(0)
        
        logger.info(f"통계 데이터 가공 완료: {len(df)}개")
        return df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str, data_type: str = "review"):
        """가공된 데이터 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # CSV 파일 저장
        csv_filename = os.path.join(self.processed_data_dir, f"{data_type}_{filename}_{timestamp}.csv")
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        logger.info(f"CSV 파일 저장: {csv_filename}")
        
        # Excel 파일 저장
        try:
            excel_filename = os.path.join(self.processed_data_dir, f"{data_type}_{filename}_{timestamp}.xlsx")
            df.to_excel(excel_filename, index=False)
            logger.info(f"Excel 파일 저장: {excel_filename}")
        except Exception as e:
            logger.error(f"Excel 저장 실패: {e}")
        
        # JSON 파일 저장
        json_filename = os.path.join(self.processed_data_dir, f"{data_type}_{filename}_{timestamp}.json")
        df.to_json(json_filename, orient='records', force_ascii=False, indent=2)
        logger.info(f"JSON 파일 저장: {json_filename}")
        
        return {
            'csv': csv_filename,
            'excel': excel_filename if 'excel_filename' in locals() else None,
            'json': json_filename
        }
    
    def load_raw_data(self, filename: str) -> Any:
        """원시 데이터 로드"""
        file_path = os.path.join(self.raw_data_dir, filename)
        
        if not os.path.exists(file_path):
            logger.error(f"파일을 찾을 수 없습니다: {file_path}")
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if filename.endswith('.json'):
                    return json.load(f)
                else:
                    logger.error(f"지원하지 않는 파일 형식: {filename}")
                    return None
        except Exception as e:
            logger.error(f"파일 로드 실패: {e}")
            return None
    
    def process_all_review_files(self) -> Dict[str, pd.DataFrame]:
        """모든 리뷰 파일 가공"""
        processed_files = {}
        
        for filename in os.listdir(self.raw_data_dir):
            if filename.startswith('review_') and filename.endswith('.json'):
                logger.info(f"리뷰 파일 가공 중: {filename}")
                
                raw_data = self.load_raw_data(filename)
                if raw_data:
                    df = self.process_review_data(raw_data)
                    if not df.empty:
                        # 파일명에서 타임스탬프 추출
                        name_parts = filename.replace('.json', '').split('_')
                        timestamp = name_parts[-1] if len(name_parts) > 1 else 'unknown'
                        
                        # 가공된 데이터 저장
                        saved_files = self.save_processed_data(df, f"processed_{timestamp}", "review")
                        processed_files[filename] = {
                            'dataframe': df,
                            'files': saved_files
                        }
        
        logger.info(f"총 {len(processed_files)}개 리뷰 파일 가공 완료")
        return processed_files
    
    def process_all_stat_files(self) -> Dict[str, pd.DataFrame]:
        """모든 통계 파일 가공"""
        processed_files = {}
        
        for filename in os.listdir(self.raw_data_dir):
            if filename.startswith('stat_') and filename.endswith('.json'):
                logger.info(f"통계 파일 가공 중: {filename}")
                
                raw_data = self.load_raw_data(filename)
                if raw_data:
                    df = self.process_stat_data(raw_data)
                    if not df.empty:
                        # 파일명에서 타임스탬프 추출
                        name_parts = filename.replace('.json', '').split('_')
                        timestamp = name_parts[-1] if len(name_parts) > 1 else 'unknown'
                        
                        # 가공된 데이터 저장
                        saved_files = self.save_processed_data(df, f"processed_{timestamp}", "stat")
                        processed_files[filename] = {
                            'dataframe': df,
                            'files': saved_files
                        }
        
        logger.info(f"총 {len(processed_files)}개 통계 파일 가공 완료")
        return processed_files


def main():
    """테스트용 메인 함수"""
    processor = DataProcessor()
    
    # 모든 파일 가공
    review_results = processor.process_all_review_files()
    stat_results = processor.process_all_stat_files()
    
    print(f"리뷰 파일 가공: {len(review_results)}개")
    print(f"통계 파일 가공: {len(stat_results)}개")


if __name__ == "__main__":
    main()
