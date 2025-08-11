#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
네이버 스마트플레이스 통계 크롤러 (인증 관리 통합 버전)
채널별 PV, 방문자, 리뷰 등 다양한 메트릭을 수집합니다.
"""

import pandas as pd
import json
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging

# 상대 import를 절대 import로 변경
try:
    from scripts.crawler.naver_place_pv_crawler_base import NaverCrawlerBase
    from scripts.crawler.naver_place_pv_auth_manager import NaverAuthManager
    from scripts.util.logger import logger
    from scripts.util.config import ClientInfo, AuthConfig
except ImportError:
    # 직접 실행 시를 위한 절대 import
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    sys.path.insert(0, current_dir)
    from scripts.crawler.naver_place_pv_crawler_base import NaverCrawlerBase
    from scripts.crawler.naver_place_pv_auth_manager import NaverAuthManager
    from scripts.util.logger import logger
    from scripts.util.config import ClientInfo, AuthConfig

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NaverStatCrawler(NaverCrawlerBase):
    """네이버 스마트플레이스 통계 데이터 크롤러"""

    def __init__(self, client_info: ClientInfo, auth_config: AuthConfig):
        """
        NaverStatCrawler 초기화
        Args:
            client_info (ClientInfo): 클라이언트 정보
            auth_config (AuthConfig): 인증 설정
        """
        auth_manager = NaverAuthManager(
            client_info=client_info,
            auth_config=auth_config,
            auth_type='place'
        )
        super().__init__(client_info=client_info, auth_manager=auth_manager) # 부모 클래스 초기화
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client_info = client_info
        self.naver_place_id = None
        self.smart_place_id = None
        
        # 스마트플레이스 API 설정
        self.base_url = "https://new.smartplace.naver.com/proxy/bizadvisor/api/v3/sites/sp_311b9ba993e974/report"
        
        # 데이터 저장소
        self.all_data = {}
        self.channels = set()
        self.channel_appearances = {}  # 채널별 출현 횟수
        
        # 크롤링 설정
        self.request_delay = 0.5  # 요청 간 딜레이 (초)
        self.max_retries = 3
        self.timeout = 30
        
        # 지원하는 메트릭과 차원
        self.available_metrics = ['pv', 'visitors', 'reviews', 'rating', 'clicks', 'impressions']
        self.available_dimensions = ['mapped_channel_name', 'mapped_channel_id', 'mapped_channel_type', 'mapped_channel_category', 'ref_keyword']
    
    def fetch_channel_data_for_date(self, date: str) -> List[Dict[str, Any]]:
        """특정 날짜의 채널별 데이터를 가져오기"""
        url = self.base_url
        params = {
            'dimensions': 'mapped_channel_name',
            'startDate': date,
            'endDate': date,
            'metrics': 'pv',
            'sort': 'pv',
            'useIndex': 'revenue-all-channel-detail'
        }
        
        try:
            logger.info(f"📊 {date} 채널 데이터 수집 중...")
            data = self.make_request('GET', url, params=params)

            print("\n--- [채널 API] 서버 응답 ---")
            print(data)
            print("--------------------------\n")

            if data and isinstance(data, list):
                logger.info(f"✅ {date} 채널 데이터 수집 성공: {len(data)}개 항목")
                return data
            elif data:
                logger.warning(f"⚠️ 채널 데이터가 리스트 형태가 아님: {data}")
                return []
            else:
                logger.info(f"ℹ️ {date} 채널 데이터 없음.")
                return []
        except ApiCallError as e:
            logger.error(f"❌ {date} 채널 데이터 수집 중 API 오류: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ {date} 채널 데이터 수집 중 예상치 못한 오류: {e}", exc_info=True)
            return []

    def fetch_keyword_data_for_date(self, date: str) -> List[Dict[str, Any]]:
        """특정 날짜의 키워드별 데이터를 가져오기"""
        url = self.base_url
        params = {
            'dimensions': 'ref_keyword',
            'startDate': date,
            'endDate': date,
            'metrics': 'pv',
            'sort': 'pv',
            'useIndex': 'revenue-search-channel-detail'
        }
        
        try:
            logger.info(f"📊 {date} 키워드 데이터 수집 중...")
            data = self.make_request('GET', url, params=params)

            print("\n--- [키워드 API] 서버 응답 ---")
            print(data)
            print("----------------------------\n")

            if data and isinstance(data, list):
                logger.info(f"✅ {date} 키워드 데이터 수집 성공: {len(data)}개 항목")
                return data
            elif data:
                logger.warning(f"⚠️ 키워드 데이터가 리스트 형태가 아님: {data}")
                return []
            else:
                logger.info(f"ℹ️ {date} 키워드 데이터 없음.")
                return []
        except ApiCallError as e:
            logger.error(f"❌ {date} 키워드 데이터 수집 중 API 오류: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ {date} 키워드 데이터 수집 중 예상치 못한 오류: {e}", exc_info=True)
            return []
    
    def fetch_comprehensive_data_for_date(self, date: str) -> Dict[str, List[Dict]]:
        """특정 날짜의 종합 데이터를 가져오기 (여러 메트릭)"""
        comprehensive_data = {}
        
        # 1. 기본 PV 데이터
        pv_data = self.fetch_channel_data_for_date(date)
        comprehensive_data['pv'] = pv_data
        
        # 2. 방문자 데이터
        visitors_data = self.fetch_channel_data_for_date(date)
        comprehensive_data['visitors'] = visitors_data
        
        # 3. 리뷰 데이터
        reviews_data = self.fetch_channel_data_for_date(date)
        comprehensive_data['reviews'] = reviews_data
        
        # 4. 평점 데이터
        rating_data = self.fetch_channel_data_for_date(date)
        comprehensive_data['rating'] = rating_data
        
        # 5. 클릭 데이터
        clicks_data = self.fetch_channel_data_for_date(date)
        comprehensive_data['clicks'] = clicks_data
        
        # 6. 노출 데이터
        impressions_data = self.fetch_channel_data_for_date(date)
        comprehensive_data['impressions'] = impressions_data
        
        return comprehensive_data
    
    def collect_all_data(self, start_date: str, end_date: str, comprehensive: bool = False) -> Dict[str, List[Dict]]:
        """지정된 기간의 전체 데이터 수집"""
        logger.info("=" * 60)
        logger.info("🚀 네이버 스마트플레이스 채널별 데이터 수집 시작")
        logger.info(f"📅 기간: {start_date} ~ {end_date}")
        logger.info(f"📊 모드: {'종합 데이터' if comprehensive else '기본 PV 데이터'}")
        logger.info("=" * 60)
        
        # 날짜 범위 생성
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        current_date = start_dt
        while current_date <= end_dt:
            date_str = current_date.strftime('%Y-%m-%d')
            
            if comprehensive:
                # 종합 데이터 수집
                comprehensive_data = self.fetch_comprehensive_data_for_date(date_str)
                self.all_data[date_str] = comprehensive_data
            else:
                # 기본 PV 데이터 수집
                data = self.fetch_channel_data_for_date(date_str)
                self.all_data[date_str] = data
            
            # 다음 날짜로
            current_date += timedelta(days=1)
            
            # 요청 간 딜레이
            time.sleep(self.request_delay)
        
        # 전체 채널 목록 파악
        logger.info("\n📊 전체 채널 목록 파악 중...")
        
        for date_str, data_list in self.all_data.items():
            if isinstance(data_list, dict):
                # 종합 데이터 모드
                for metric, metric_data in data_list.items():
                    for item in metric_data:
                        if 'mapped_channel_name' in item and item['mapped_channel_name']:
                            channel_name = item['mapped_channel_name']
                            self.channels.add(channel_name)
                            
                            if channel_name not in self.channel_appearances:
                                self.channel_appearances[channel_name] = 0
                            self.channel_appearances[channel_name] += 1
            else:
                # 기본 데이터 모드
                for item in data_list:
                    if 'mapped_channel_name' in item and item['mapped_channel_name']:
                        channel_name = item['mapped_channel_name']
                        self.channels.add(channel_name)
                        
                        if channel_name not in self.channel_appearances:
                            self.channel_appearances[channel_name] = 0
                        self.channel_appearances[channel_name] += 1
        
        logger.info(f"\n✅ 데이터 수집 완료!")
        logger.info(f"📊 총 {len(self.channels)}개 채널 발견:")
        
        # 출현 빈도순으로 정렬하여 출력
        sorted_channels = sorted(self.channel_appearances.items(), key=lambda x: x[1], reverse=True)
        for channel, count in sorted_channels:
            logger.info(f"   - {channel}: {count}일 출현")
        
        return self.all_data
    
    def collect_keyword_data(self, start_date: str, end_date: str) -> Dict[str, List[Dict]]:
        """지정된 기간의 키워드별 데이터 수집"""
        logger.info("=" * 60)
        logger.info("🚀 네이버 스마트플레이스 키워드별 데이터 수집 시작")
        logger.info(f"📅 기간: {start_date} ~ {end_date}")
        logger.info("=" * 60)
        
        # 날짜 범위 생성
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        current_date = start_dt
        while current_date <= end_dt:
            date_str = current_date.strftime('%Y-%m-%d')
            
            # 키워드 데이터 수집
            data = self.fetch_keyword_data_for_date(date_str)
            self.all_data[date_str] = data
            
            # 다음 날짜로
            current_date += timedelta(days=1)
            
            # 요청 간 딜레이
            time.sleep(self.request_delay)
        
        # 전체 키워드 목록 파악
        logger.info("\n📊 전체 키워드 목록 파악 중...")
        
        keywords = set()
        keyword_appearances = {}
        
        for date_str, data_list in self.all_data.items():
            for item in data_list:
                if 'ref_keyword' in item and item['ref_keyword']:
                    keyword_name = item['ref_keyword']
                    keywords.add(keyword_name)
                    
                    if keyword_name not in keyword_appearances:
                        keyword_appearances[keyword_name] = 0
                    keyword_appearances[keyword_name] += 1
        
        logger.info(f"\n✅ 키워드 데이터 수집 완료!")
        logger.info(f"📊 총 {len(keywords)}개 키워드 발견:")
        
        # 출현 빈도순으로 정렬하여 출력
        sorted_keywords = sorted(keyword_appearances.items(), key=lambda x: x[1], reverse=True)
        for keyword, count in sorted_keywords[:10]:  # 상위 10개만 출력
            logger.info(f"   - {keyword}: {count}일 출현")
        
        return self.all_data
    
    def collect_channel_data(self, start_date: str, end_date: str) -> Dict[str, List[Dict]]:
        """지정된 기간의 채널별 데이터 수집"""
        logger.info("=" * 60)
        logger.info("🚀 네이버 스마트플레이스 채널별 데이터 수집 시작")
        logger.info(f"📅 기간: {start_date} ~ {end_date}")
        logger.info("=" * 60)
        
        # 날짜 범위 생성
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        current_date = start_dt
        while current_date <= end_dt:
            date_str = current_date.strftime('%Y-%m-%d')
            
            # 채널 데이터 수집
            data = self.fetch_channel_data_for_date(date_str)
            self.all_data[date_str] = data
            
            # 다음 날짜로
            current_date += timedelta(days=1)
            
            # 요청 간 딜레이
            time.sleep(self.request_delay)
        
        # 전체 채널 목록 파악
        logger.info("\n📊 전체 채널 목록 파악 중...")
        
        channels = set()
        channel_appearances = {}
        
        for date_str, data_list in self.all_data.items():
            for item in data_list:
                if 'mapped_channel_name' in item and item['mapped_channel_name']:
                    channel_name = item['mapped_channel_name']
                    channels.add(channel_name)
                    
                    if channel_name not in channel_appearances:
                        channel_appearances[channel_name] = 0
                    channel_appearances[channel_name] += 1
        
        logger.info(f"\n✅ 채널 데이터 수집 완료!")
        logger.info(f"📊 총 {len(channels)}개 채널 발견:")
        
        # 출현 빈도순으로 정렬하여 출력
        sorted_channels = sorted(channel_appearances.items(), key=lambda x: x[1], reverse=True)
        for channel, count in sorted_channels[:10]:  # 상위 10개만 출력
            logger.info(f"   - {channel}: {count}일 출현")
        
        return self.all_data
    
    def create_dataframe(self, comprehensive: bool = False) -> pd.DataFrame:
        """수집된 데이터를 DataFrame으로 변환"""
        if not self.all_data:
            logger.warning("⚠️ 수집된 데이터가 없습니다.")
            return pd.DataFrame()
        
        if comprehensive:
            return self._create_comprehensive_dataframe()
        else:
            return self._create_basic_dataframe()
    
    def create_keyword_dataframe(self) -> pd.DataFrame:
        """키워드별 데이터를 DataFrame으로 변환"""
        if not self.all_data:
            logger.warning("⚠️ 수집된 키워드 데이터가 없습니다.")
            return pd.DataFrame()
        
        logger.info("📊 키워드 DataFrame 생성 중...")
        
        rows = []
        for date, data_list in self.all_data.items():
            for item in data_list:
                row = {
                    'date': date,
                    'keyword': item.get('ref_keyword', 'Unknown'),
                    'pv': item.get('pv', 0),
                    'keyword_id': item.get('ref_keyword_id', ''),
                    'keyword_type': item.get('ref_keyword_type', ''),
                    'keyword_category': item.get('ref_keyword_category', '')
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # 날짜 컬럼을 datetime으로 변환
        df['date'] = pd.to_datetime(df['date'])
        
        # 키워드별 총 PV 계산
        df['total_pv'] = df.groupby('keyword')['pv'].transform('sum')
        
        logger.info(f"✅ 키워드 DataFrame 생성 완료: {len(df)}행, {len(df['keyword'].unique())}개 키워드")
        return df
    
    def create_channel_dataframe(self) -> pd.DataFrame:
        """채널별 데이터를 DataFrame으로 변환"""
        if not self.all_data:
            logger.warning("⚠️ 수집된 채널 데이터가 없습니다.")
            return pd.DataFrame()
        
        logger.info("📊 채널 DataFrame 생성 중...")
        
        rows = []
        for date, data_list in self.all_data.items():
            for item in data_list:
                row = {
                    'date': date,
                    'channel': item.get('mapped_channel_name', 'Unknown'),
                    'pv': item.get('pv', 0),
                    'channel_id': item.get('mapped_channel_id', ''),
                    'channel_type': item.get('mapped_channel_type', ''),
                    'channel_category': item.get('mapped_channel_category', '')
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # 날짜 컬럼을 datetime으로 변환
        df['date'] = pd.to_datetime(df['date'])
        
        # 채널별 총 PV 계산
        df['total_pv'] = df.groupby('channel')['pv'].transform('sum')
        
        logger.info(f"✅ 채널 DataFrame 생성 완료: {len(df)}행, {len(df['channel'].unique())}개 채널")
        return df
    
    def _create_basic_dataframe(self) -> pd.DataFrame:
        """기본 PV 데이터 DataFrame 생성"""
        logger.info("📊 기본 DataFrame 생성 중...")
        
        rows = []
        for date, data_list in self.all_data.items():
            for item in data_list:
                row = {
                    'date': date,
                    'channel': item.get('mapped_channel_name', 'Unknown'),
                    'pv': item.get('pv', 0),
                    'channel_id': item.get('mapped_channel_id', ''),
                    'channel_type': item.get('mapped_channel_type', ''),
                    'channel_category': item.get('mapped_channel_category', '')
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # 날짜 컬럼을 datetime으로 변환
        df['date'] = pd.to_datetime(df['date'])
        
        # 채널별 총 PV 계산
        df['total_pv'] = df.groupby('channel')['pv'].transform('sum')
        
        logger.info(f"✅ 기본 DataFrame 생성 완료: {len(df)}행, {len(df['channel'].unique())}개 채널")
        return df
    
    def _create_comprehensive_dataframe(self) -> pd.DataFrame:
        """종합 데이터 DataFrame 생성"""
        logger.info("📊 종합 DataFrame 생성 중...")
        
        rows = []
        for date, comprehensive_data in self.all_data.items():
            # 각 메트릭별 데이터 통합
            channel_data = {}
            
            for metric, metric_data in comprehensive_data.items():
                for item in metric_data:
                    channel_name = item.get('mapped_channel_name', 'Unknown')
                    if channel_name not in channel_data:
                        channel_data[channel_name] = {
                            'date': date,
                            'channel': channel_name,
                            'channel_id': item.get('mapped_channel_id', ''),
                            'channel_type': item.get('mapped_channel_type', ''),
                            'channel_category': item.get('mapped_channel_category', '')
                        }
                    
                    # 메트릭 값 추가
                    channel_data[channel_name][metric] = item.get(metric, 0)
            
            # 행 데이터 추가
            for channel_name, data in channel_data.items():
                rows.append(data)
        
        df = pd.DataFrame(rows)
        
        # 날짜 컬럼을 datetime으로 변환
        df['date'] = pd.to_datetime(df['date'])
        
        # 누락된 메트릭 컬럼을 0으로 채우기
        for metric in self.available_metrics:
            if metric not in df.columns:
                df[metric] = 0
        
        logger.info(f"✅ 종합 DataFrame 생성 완료: {len(df)}행, {len(df['channel'].unique())}개 채널")
        return df
    
    def calculate_statistics(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """통계 계산"""
        logger.info("📈 통계 계산 중...")
        
        # 키워드 데이터인지 확인
        if 'keyword' in df.columns:
            return self._calculate_keyword_statistics(df)
        
        # 채널별 통계
        if 'pv' in df.columns:
            channel_stats = df.groupby('channel').agg({
                'pv': ['sum', 'mean', 'std', 'count'],
                'date': ['min', 'max']
            }).round(2)
            
            # 컬럼명 정리
            channel_stats.columns = [
                'total_pv', 'avg_pv', 'std_pv', 'data_count',
                'start_date', 'end_date'
            ]
        else:
            # 종합 데이터의 경우
            agg_dict = {}
            for metric in self.available_metrics:
                if metric in df.columns:
                    agg_dict[metric] = ['sum', 'mean', 'std', 'count']
            
            if 'date' in df.columns:
                agg_dict['date'] = ['min', 'max']
            
            channel_stats = df.groupby('channel').agg(agg_dict).round(2)
        
        # 전체 통계
        total_stats = {
            'total_channels': len(df['channel'].unique()),
            'date_range': f"{df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}"
        }
        
        # 각 메트릭별 총합 추가
        for metric in self.available_metrics:
            if metric in df.columns:
                total_stats[f'total_{metric}'] = df[metric].sum()
                total_stats[f'avg_{metric}_per_channel'] = df.groupby('channel')[metric].sum().mean()
        
        logger.info(f"✅ 통계 계산 완료: {len(channel_stats)}개 채널")
        return channel_stats, total_stats
    
    def _calculate_keyword_statistics(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """키워드 데이터 통계 계산"""
        logger.info("📈 키워드 통계 계산 중...")
        
        # 키워드별 통계
        if 'pv' in df.columns:
            keyword_stats = df.groupby('keyword').agg({
                'pv': ['sum', 'mean', 'std', 'count'],
                'date': ['min', 'max']
            }).round(2)
            
            # 컬럼명 정리
            keyword_stats.columns = [
                'total_pv', 'avg_pv', 'std_pv', 'data_count',
                'start_date', 'end_date'
            ]
        else:
            keyword_stats = pd.DataFrame()
        
        # 전체 통계
        total_stats = {
            'total_keywords': len(df['keyword'].unique()),
            'date_range': f"{df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}"
        }
        
        # 각 메트릭별 총합 추가
        for metric in self.available_metrics:
            if metric in df.columns:
                total_stats[f'total_{metric}'] = df[metric].sum()
                total_stats[f'avg_{metric}_per_keyword'] = df.groupby('keyword')[metric].sum().mean()
        
        logger.info(f"✅ 키워드 통계 계산 완료: {len(keyword_stats)}개 키워드")
        return keyword_stats, total_stats
    
    def save_to_files(self, df: pd.DataFrame, stats: pd.DataFrame, output_dir: str = ".", comprehensive: bool = False):
        """결과를 파일로 저장"""
        # 키워드 데이터인지 확인
        if 'keyword' in df.columns:
            return self._save_keyword_files(df, stats, output_dir, comprehensive)
        
        logger.info("💾 파일 저장 중...")
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 파일명에 모드 표시
        mode_suffix = "_comprehensive" if comprehensive else "_basic"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # CSV 파일 저장
        csv_file = os.path.join(output_dir, f"channel_data{mode_suffix}_{timestamp}.csv")
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        logger.info(f"📄 CSV 저장: {csv_file}")
        
        # Excel 파일 저장
        excel_file = os.path.join(output_dir, f"channel_data{mode_suffix}_{timestamp}.xlsx")
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Raw Data', index=False)
            stats.to_excel(writer, sheet_name='Channel Statistics')
            
            # 종합 데이터의 경우 메트릭별 시트 추가
            if comprehensive:
                for metric in self.available_metrics:
                    if metric in df.columns:
                        metric_df = df[['date', 'channel', metric]].pivot_table(
                            index='date', columns='channel', values=metric, fill_value=0
                        )
                        metric_df.to_excel(writer, sheet_name=f'{metric.upper()}_Data')
        
        logger.info(f"📊 Excel 저장: {excel_file}")
        
        # JSON 파일 저장
        json_file = os.path.join(output_dir, f"channel_data{mode_suffix}_{timestamp}.json")
        data_dict = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_channels': len(df['channel'].unique()),
                'total_records': len(df),
                'date_range': f"{df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}",
                'comprehensive_mode': comprehensive,
                'available_metrics': self.available_metrics
            },
            'raw_data': df.to_dict('records'),
            'channel_statistics': stats.to_dict('index'),
            'channel_appearances': self.channel_appearances
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"📋 JSON 저장: {json_file}")
    
    def _save_keyword_files(self, df: pd.DataFrame, stats: pd.DataFrame, output_dir: str = ".", comprehensive: bool = False):
        """키워드 데이터를 파일로 저장"""
        logger.info("💾 키워드 파일 저장 중...")
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 파일명에 모드 표시
        mode_suffix = "_comprehensive" if comprehensive else "_basic"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # CSV 파일 저장
        csv_file = os.path.join(output_dir, f"keyword_data{mode_suffix}_{timestamp}.csv")
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        logger.info(f"📄 CSV 저장: {csv_file}")
        
        # Excel 파일 저장
        excel_file = os.path.join(output_dir, f"keyword_data{mode_suffix}_{timestamp}.xlsx")
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Raw Data', index=False)
            stats.to_excel(writer, sheet_name='Keyword Statistics')
            
            # 종합 데이터의 경우 메트릭별 시트 추가
            if comprehensive:
                for metric in self.available_metrics:
                    if metric in df.columns:
                        metric_df = df[['date', 'keyword', metric]].pivot_table(
                            index='date', columns='keyword', values=metric, fill_value=0
                        )
                        metric_df.to_excel(writer, sheet_name=f'{metric.upper()}_Data')
        
        logger.info(f"📊 Excel 저장: {excel_file}")
        
        # JSON 파일 저장
        json_file = os.path.join(output_dir, f"keyword_data{mode_suffix}_{timestamp}.json")
        data_dict = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_keywords': len(df['keyword'].unique()),
                'total_records': len(df),
                'date_range': f"{df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}",
                'comprehensive_mode': comprehensive,
                'available_metrics': self.available_metrics
            },
            'raw_data': df.to_dict('records'),
            'keyword_statistics': stats.to_dict('index')
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"📋 JSON 저장: {json_file}")
    
    def display_results(self, df: pd.DataFrame, stats: pd.DataFrame, comprehensive: bool = False):
        """결과를 콘솔에 출력"""
        print("\n" + "=" * 60)
        print("📊 수집 결과 요약")
        print("=" * 60)
        
        # 기본 정보
        print(f"📅 기간: {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}")
        print(f"📊 총 채널 수: {len(df['channel'].unique())}")
        print(f"📋 총 레코드 수: {len(df)}")
        
        if comprehensive:
            # 종합 데이터의 경우 각 메트릭별 정보 출력
            for metric in self.available_metrics:
                if metric in df.columns:
                    total_value = df[metric].sum()
                    print(f"📈 총 {metric.upper()}: {total_value:,}")
        else:
            # 기본 PV 데이터의 경우
            total_pv = df['pv'].sum()
            print(f"📈 총 PV: {total_pv:,}")
        
        # 상위 채널 (PV 기준)
        if 'pv' in df.columns:
            top_channels = df.groupby('channel')['pv'].sum().sort_values(ascending=False).head(10)
            print(f"\n🏆 상위 10개 채널 (PV 기준):")
            for i, (channel, pv) in enumerate(top_channels.items(), 1):
                print(f"  {i:2d}. {channel}: {pv:,} PV")
        
        # 통계 요약
        print(f"\n📈 통계 요약:")
        if 'pv' in df.columns:
            print(f"  평균 PV/채널: {df.groupby('channel')['pv'].sum().mean():.2f}")
            print(f"  최대 PV/채널: {df.groupby('channel')['pv'].sum().max():,}")
            print(f"  최소 PV/채널: {df.groupby('channel')['pv'].sum().min():,}")
    
    def create_visual_report(self, df: pd.DataFrame, output_dir: str = ".", comprehensive: bool = False):
        """시각화 리포트 생성"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            logger.info("📊 시각화 리포트 생성 중...")
            
            # 한글 폰트 설정
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['axes.unicode_minus'] = False
            
            if comprehensive:
                # 종합 데이터 시각화
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                fig.suptitle('Comprehensive Channel Analysis Report', fontsize=16)
                
                metrics_to_plot = [m for m in self.available_metrics if m in df.columns][:6]
                
                for i, metric in enumerate(metrics_to_plot):
                    row, col = i // 3, i % 3
                    metric_data = df.groupby('channel')[metric].sum().sort_values(ascending=False).head(10)
                    metric_data.plot(kind='bar', ax=axes[row, col])
                    axes[row, col].set_title(f'Top 10 Channels by {metric.upper()}')
                    axes[row, col].set_xlabel('Channel')
                    axes[row, col].set_ylabel(metric.upper())
                    axes[row, col].tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
            else:
                # 기본 PV 데이터 시각화
                plt.figure(figsize=(12, 8))
                channel_pv = df.groupby('channel')['pv'].sum().sort_values(ascending=False).head(15)
                channel_pv.plot(kind='bar')
                plt.title('Top 15 Channels by PV', fontsize=14)
                plt.xlabel('Channel')
                plt.ylabel('Total PV')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
            
            chart_file = os.path.join(output_dir, f"channel_analysis_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📊 차트 저장: {chart_file}")
            
        except ImportError:
            logger.warning("matplotlib 또는 seaborn이 설치되지 않아 시각화를 건너뜁니다.")
        except Exception as e:
            logger.error(f"시각화 생성 실패: {e}")
    
    def run(self, start_date: str, end_date: str, output_dir: str = ".", comprehensive: bool = False):
        """전체 워크플로우 실행"""
        logger.info("🚀 네이버 스마트플레이스 크롤링 시작")
        
        try:
            # 1. 데이터 수집
            self.collect_all_data(start_date, end_date, comprehensive)
            
            # 2. DataFrame 생성
            df = self.create_dataframe(comprehensive)
            
            # 3. 통계 계산
            stats, total_stats = self.calculate_statistics(df)
            
            # 4. 파일 저장
            self.save_to_files(df, stats, output_dir, comprehensive)
            
            # 5. 결과 출력
            self.display_results(df, stats, comprehensive)
            
            # 6. 시각화
            self.create_visual_report(df, output_dir, comprehensive)
            
            logger.info("✅ 크롤링 완료!")
            return True
            
        except Exception as e:
            logger.error(f"❌ 크롤링 실패: {e}")
            return False
    
    def run_keyword_analysis(self, start_date: str, end_date: str, output_dir: str = "."):
        """키워드별 분석 워크플로우 실행"""
        logger.info("🚀 네이버 스마트플레이스 키워드 분석 시작")
        
        try:
            # 1. 키워드 데이터 수집
            self.collect_keyword_data(start_date, end_date)
            
            # 2. 키워드 DataFrame 생성
            df = self.create_keyword_dataframe()
            
            if df.empty:
                logger.error("❌ 키워드 데이터 처리 실패")
                return False
            
            # 3. 통계 계산
            stats, total_stats = self.calculate_statistics(df)
            
            # 4. 파일 저장
            self.save_to_files(df, stats, output_dir, comprehensive=False)
            
            # 5. 결과 출력
            self.display_results(df, stats, comprehensive=False)
            
            # 6. 시각화
            self.create_visual_report(df, output_dir, comprehensive=False)
            
            logger.info("✅ 키워드 분석 완료!")
            return True
            
        except Exception as e:
            logger.error(f"❌ 키워드 분석 실패: {e}")
            return False
    
    def run_channel_analysis(self, start_date: str, end_date: str, output_dir: str = "."):
        """채널별 분석 워크플로우 실행"""
        logger.info("🚀 네이버 스마트플레이스 채널 분석 시작")
        
        try:
            # 1. 채널 데이터 수집
            self.collect_channel_data(start_date, end_date)
            
            # 2. 채널 DataFrame 생성
            df = self.create_channel_dataframe()
            
            if df.empty:
                logger.error("❌ 채널 데이터 처리 실패")
                return False
            
            # 3. 통계 계산
            stats, total_stats = self.calculate_statistics(df)
            
            # 4. 파일 저장
            self.save_to_files(df, stats, output_dir, comprehensive=False)
            
            # 5. 결과 출력
            self.display_results(df, stats, comprehensive=False)
            
            # 6. 시각화
            self.create_visual_report(df, output_dir, comprehensive=False)
            
            logger.info("✅ 채널 분석 완료!")
            return True
            
        except Exception as e:
            logger.error(f"❌ 채널 분석 실패: {e}")
            return False


def main():
    """테스트용 메인 함수"""
    import asyncio
    
    try:
        # 설정 매니저에서 클라이언트 정보 조회
        from scripts.util.config import get_config_manager
        config_manager = get_config_manager()
        client = config_manager.get_selected_client_config()
        
        if not client:
            print("❌ 클라이언트 설정을 찾을 수 없습니다. .env를 확인하세요.")
            return
        
        print(f"✅ 선택된 클라이언트: {client.name}")
        
        # 크롤러 생성 (클라이언트 설정 사용)
        crawler = NaverStatCrawler(client, client.auth_config)
        
        # 테스트 기간 설정
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        print("\n네이버 스마트플레이스 분석 유형을 선택하세요:")
        print("1. 키워드별 분석 (유입 키워드 통계)")
        print("2. 채널별 분석 (유입 채널 통계)")
        print("3. 종합 분석 (모든 메트릭)")
        
        choice = input("\n선택하세요 (1-3): ").strip()
        
        # 출력 디렉토리 생성
        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)
        
        if choice == "1":
            # 키워드별 분석
            success = crawler.run_keyword_analysis(start_date, end_date, output_dir)
            if success:
                print("✅ 키워드 분석 성공!")
            else:
                print("❌ 키워드 분석 실패!")
                
        elif choice == "2":
            # 채널별 분석
            success = crawler.run_channel_analysis(start_date, end_date, output_dir)
            if success:
                print("✅ 채널 분석 성공!")
            else:
                print("❌ 채널 분석 실패!")
                
        elif choice == "3":
            # 종합 분석 (기존 기능)
            success = crawler.run(start_date, end_date, output_dir, comprehensive=True)
            if success:
                print("✅ 종합 분석 성공!")
            else:
                print("❌ 종합 분석 실패!")
        else:
            print("❌ 잘못된 선택입니다.")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
