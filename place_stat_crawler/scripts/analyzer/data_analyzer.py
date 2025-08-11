#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
데이터 분석 클래스
가공된 데이터를 분석하는 기능을 제공합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class DataAnalyzer:
    """데이터 분석 클래스"""
    
    def __init__(self, processed_data_dir: str = "../data/processed", analyzed_data_dir: str = "../data/analyzed"):
        """
        Args:
            processed_data_dir: 가공된 데이터 디렉토리
            analyzed_data_dir: 분석 결과 저장 디렉토리
        """
        self.processed_data_dir = processed_data_dir
        self.analyzed_data_dir = analyzed_data_dir
        
        # 디렉토리 생성
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.analyzed_data_dir, exist_ok=True)
        
        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
    
    def analyze_review_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """리뷰 데이터 분석"""
        if df.empty:
            logger.warning("분석할 리뷰 데이터가 없습니다.")
            return {}
        
        analysis_results = {
            'total_reviews': len(df),
            'unique_stores': df['store_name'].nunique(),
            'unique_authors': df['author'].nunique(),
            'date_range': {
                'start': df['created_at'].min(),
                'end': df['created_at'].max()
            },
            'store_distribution': df['store_name'].value_counts().to_dict(),
            'author_distribution': df['author'].value_counts().head(10).to_dict(),
            'daily_review_count': df.groupby(df['created_at'].dt.date).size().to_dict(),
            'average_score': df['score'].mean(),
            'score_distribution': df['score'].value_counts().sort_index().to_dict(),
            'insulting_count': df['is_insulting'].sum(),
            'defamatory_count': df['is_defamatory'].sum()
        }
        
        logger.info(f"리뷰 데이터 분석 완료: {len(df)}개 리뷰")
        return analysis_results
    
    def analyze_stat_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """통계 데이터 분석"""
        if df.empty:
            logger.warning("분석할 통계 데이터가 없습니다.")
            return {}
        
        # 채널별 총 PV
        channel_totals = df.groupby('channel_name')['pv'].sum().sort_values(ascending=False)
        
        # 일별 총 PV
        daily_totals = df.groupby('date')['pv'].sum()
        
        # 채널별 일평균 PV
        channel_daily_avg = df.groupby('channel_name')['pv'].mean().sort_values(ascending=False)
        
        analysis_results = {
            'total_records': len(df),
            'unique_channels': df['channel_name'].nunique(),
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max()
            },
            'total_pv': df['pv'].sum(),
            'average_daily_pv': daily_totals.mean(),
            'channel_totals': channel_totals.to_dict(),
            'daily_totals': daily_totals.to_dict(),
            'channel_daily_avg': channel_daily_avg.to_dict(),
            'top_channels': channel_totals.head(10).to_dict(),
            'pv_distribution': {
                'min': df['pv'].min(),
                'max': df['pv'].max(),
                'mean': df['pv'].mean(),
                'median': df['pv'].median(),
                'std': df['pv'].std()
            }
        }
        
        logger.info(f"통계 데이터 분석 완료: {len(df)}개 레코드")
        return analysis_results
    
    def create_review_visualizations(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List[str]:
        """리뷰 데이터 시각화"""
        if df.empty:
            logger.warning("시각화할 리뷰 데이터가 없습니다.")
            return []
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_files = []
        
        # 1. 일별 리뷰 수 추이
        plt.figure(figsize=(12, 6))
        daily_counts = df.groupby(df['created_at'].dt.date).size()
        plt.plot(daily_counts.index, daily_counts.values, marker='o')
        plt.title('일별 리뷰 수 추이')
        plt.xlabel('날짜')
        plt.ylabel('리뷰 수')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = os.path.join(self.analyzed_data_dir, f"review_daily_trend_{timestamp}.png")
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        saved_files.append(filename)
        
        # 2. 매장별 리뷰 수 분포
        plt.figure(figsize=(12, 6))
        store_counts = df['store_name'].value_counts().head(10)
        store_counts.plot(kind='bar')
        plt.title('매장별 리뷰 수 (상위 10개)')
        plt.xlabel('매장명')
        plt.ylabel('리뷰 수')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        filename = os.path.join(self.analyzed_data_dir, f"review_store_distribution_{timestamp}.png")
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        saved_files.append(filename)
        
        # 3. 점수 분포
        plt.figure(figsize=(10, 6))
        df['score'].value_counts().sort_index().plot(kind='bar')
        plt.title('리뷰 점수 분포')
        plt.xlabel('점수')
        plt.ylabel('리뷰 수')
        plt.tight_layout()
        
        filename = os.path.join(self.analyzed_data_dir, f"review_score_distribution_{timestamp}.png")
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        saved_files.append(filename)
        
        logger.info(f"리뷰 시각화 완료: {len(saved_files)}개 파일")
        return saved_files
    
    def create_stat_visualizations(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List[str]:
        """통계 데이터 시각화"""
        if df.empty:
            logger.warning("시각화할 통계 데이터가 없습니다.")
            return []
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_files = []
        
        # 1. 일별 PV 추이
        plt.figure(figsize=(12, 6))
        daily_pv = df.groupby('date')['pv'].sum()
        plt.plot(daily_pv.index, daily_pv.values, marker='o')
        plt.title('일별 총 PV 추이')
        plt.xlabel('날짜')
        plt.ylabel('총 PV')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = os.path.join(self.analyzed_data_dir, f"stat_daily_pv_{timestamp}.png")
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        saved_files.append(filename)
        
        # 2. 채널별 총 PV (상위 10개)
        plt.figure(figsize=(12, 6))
        channel_totals = df.groupby('channel_name')['pv'].sum().sort_values(ascending=False).head(10)
        channel_totals.plot(kind='bar')
        plt.title('채널별 총 PV (상위 10개)')
        plt.xlabel('채널명')
        plt.ylabel('총 PV')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        filename = os.path.join(self.analyzed_data_dir, f"stat_channel_totals_{timestamp}.png")
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        saved_files.append(filename)
        
        # 3. 채널별 일평균 PV
        plt.figure(figsize=(12, 6))
        channel_avg = df.groupby('channel_name')['pv'].mean().sort_values(ascending=False).head(10)
        channel_avg.plot(kind='bar')
        plt.title('채널별 일평균 PV (상위 10개)')
        plt.xlabel('채널명')
        plt.ylabel('일평균 PV')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        filename = os.path.join(self.analyzed_data_dir, f"stat_channel_avg_{timestamp}.png")
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        saved_files.append(filename)
        
        # 4. PV 분포 히스토그램
        plt.figure(figsize=(10, 6))
        plt.hist(df['pv'], bins=30, alpha=0.7, edgecolor='black')
        plt.title('PV 분포 히스토그램')
        plt.xlabel('PV')
        plt.ylabel('빈도')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = os.path.join(self.analyzed_data_dir, f"stat_pv_distribution_{timestamp}.png")
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        saved_files.append(filename)
        
        logger.info(f"통계 시각화 완료: {len(saved_files)}개 파일")
        return saved_files
    
    def save_analysis_report(self, analysis_results: Dict[str, Any], data_type: str = "review") -> str:
        """분석 결과 리포트 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # JSON 파일로 저장
        report_filename = os.path.join(self.analyzed_data_dir, f"{data_type}_analysis_report_{timestamp}.json")
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            import json
            # datetime 객체를 문자열로 변환
            def convert_datetime(obj):
                if isinstance(obj, (datetime, pd.Timestamp)):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_datetime(item) for item in obj]
                else:
                    return obj
            
            converted_results = convert_datetime(analysis_results)
            json.dump(converted_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"분석 리포트 저장: {report_filename}")
        return report_filename
    
    def load_processed_data(self, filename: str) -> pd.DataFrame:
        """가공된 데이터 로드"""
        file_path = os.path.join(self.processed_data_dir, filename)
        
        if not os.path.exists(file_path):
            logger.error(f"파일을 찾을 수 없습니다: {file_path}")
            return pd.DataFrame()
        
        try:
            if filename.endswith('.csv'):
                return pd.read_csv(file_path)
            elif filename.endswith('.xlsx'):
                return pd.read_excel(file_path)
            elif filename.endswith('.json'):
                return pd.read_json(file_path)
            else:
                logger.error(f"지원하지 않는 파일 형식: {filename}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"파일 로드 실패: {e}")
            return pd.DataFrame()
    
    def analyze_all_processed_files(self) -> Dict[str, Dict[str, Any]]:
        """모든 가공된 파일 분석"""
        analysis_results = {}
        
        for filename in os.listdir(self.processed_data_dir):
            if filename.endswith(('.csv', '.xlsx', '.json')):
                logger.info(f"파일 분석 중: {filename}")
                
                df = self.load_processed_data(filename)
                if not df.empty:
                    # 파일 타입에 따라 분석
                    if 'review' in filename.lower():
                        analysis_result = self.analyze_review_data(df)
                        visualizations = self.create_review_visualizations(df, analysis_result)
                        report_file = self.save_analysis_report(analysis_result, "review")
                        
                        analysis_results[filename] = {
                            'type': 'review',
                            'analysis': analysis_result,
                            'visualizations': visualizations,
                            'report': report_file
                        }
                    
                    elif 'stat' in filename.lower():
                        analysis_result = self.analyze_stat_data(df)
                        visualizations = self.create_stat_visualizations(df, analysis_result)
                        report_file = self.save_analysis_report(analysis_result, "stat")
                        
                        analysis_results[filename] = {
                            'type': 'stat',
                            'analysis': analysis_result,
                            'visualizations': visualizations,
                            'report': report_file
                        }
        
        logger.info(f"총 {len(analysis_results)}개 파일 분석 완료")
        return analysis_results


def main():
    """테스트용 메인 함수"""
    analyzer = DataAnalyzer()
    
    # 모든 가공된 파일 분석
    results = analyzer.analyze_all_processed_files()
    
    print(f"분석 완료: {len(results)}개 파일")
    for filename, result in results.items():
        print(f"  {filename}: {result['type']} 분석 완료")


if __name__ == "__main__":
    main()
