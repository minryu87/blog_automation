#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
월별 통계 데이터 수집 및 통합 스크립트
2025년 7월 1일부터 31일까지의 일자별 데이터를 수집하여 통합 테이블 생성
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import logging

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from scripts.util.logger import logger
from scripts.util.config import ClientInfo, AuthConfig, get_config_manager
from scripts.crawler.naver_place_pv_stat_crawler import NaverStatCrawler
from scripts.crawler.naver_place_pv_crawler_base import ApiCallError

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonthlyStatisticsCrawler:
    """월별 통계 데이터 수집 및 통합 클래스"""
    
    def __init__(self, client_info: ClientInfo, auth_config: AuthConfig):
        """
        Args:
            client_info: 사용할 클라이언트의 정보
            auth_config: 인증 관련 설정 정보
        """
        self.client = client_info
        self.crawler = NaverStatCrawler(client_info, auth_config)
        
        # 데이터 저장 경로
        self.base_dir = Path(__file__).resolve().parents[2]
        self.raw_data_dir = self.base_dir / 'data' / 'raw'
        self.processed_data_dir = self.base_dir / 'data' / 'processed'
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        self.column_order = [
            'date', 'year', 'month', 'day', 'day_of_week', 'data_type', 'name',
            'pv', 'total_pv', 'channel_id', 'channel_type', 'channel_category',
            'keyword', 'keyword_id', 'keyword_type', 'keyword_category'
        ]
        
        # 수집된 데이터 저장소
        self.daily_channel_data = {}  # {date: [{channel, pv, ...}]}
        self.daily_keyword_data = {}  # {date: [{keyword, pv, ...}]}
        self.daily_total_pv = {}      # {date: total_pv}
    
    def collect_monthly_data(self, year: int, month: int):
        """지정된 년월의 데이터 수집"""
        logger.info(f"🚀 {year}년 {month}월 데이터 수집 시작")
        
        # 월의 시작일과 마지막일 계산
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(days=1)
        
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        logger.info(f"📅 수집 기간: {start_date_str} ~ {end_date_str}")
        
        # 일자별 데이터 수집
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            logger.info(f"📊 {date_str} 데이터 수집 중...")
            
            try:
                # 채널별 데이터 수집
                channel_data = self.crawler.fetch_channel_data_for_date(date_str)
                self.daily_channel_data[date_str] = channel_data
                
                # 키워드별 데이터 수집
                keyword_data = self.crawler.fetch_keyword_data_for_date(date_str)
                self.daily_keyword_data[date_str] = keyword_data
                
                # 일자별 총 PV 계산 (모든 채널의 PV 합계)
                total_pv = sum(item.get('pv', 0) for item in channel_data)
                self.daily_total_pv[date_str] = total_pv
                
                logger.info(f"✅ {date_str} 완료 - 채널: {len(channel_data)}개, 키워드: {len(keyword_data)}개, 총 PV: {total_pv}")
                
            except Exception as e:
                logger.error(f"❌ {date_str} 데이터 수집 실패: {e}")
                # 빈 데이터로 설정
                self.daily_channel_data[date_str] = []
                self.daily_keyword_data[date_str] = []
                self.daily_total_pv[date_str] = 0
            
            current_date += timedelta(days=1)
        
        logger.info(f"🎉 {year}년 {month}월 데이터 수집 완료")
        
        # 마지막에 수집된 데이터를 반환하도록 수정
        return {
            "channel_data": self.daily_channel_data,
            "keyword_data": self.daily_keyword_data,
            "total_pv": self.daily_total_pv,
        }

    def save_raw_data(self, collected_data: Dict, year: int, month: int, client_name: str):
        """수집한 원본 데이터를 날짜별 JSON 파일로 저장합니다."""
        logger.info("💾 원본 데이터 저장 중...")
        month_str = f"{month:02d}"

        data_to_save = {
            "channel_data": collected_data["channel_data"],
            "keyword_data": collected_data["keyword_data"],
            "total_pv": collected_data["total_pv"],
        }

        saved_files = []
        for data_type, data in data_to_save.items():
            file_path = self.raw_data_dir / f"{client_name}_{data_type}_{year}_{month_str}.json"
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(data, f, ensure_ascii=False, indent=4, default=str)
                saved_files.append(str(file_path))
            except (IOError, TypeError) as e:
                logger.error(f"❌ {file_path} 저장 실패: {e}", exc_info=True)
        
        if saved_files:
            logger.info(f"✅ 원본 데이터 저장 완료: {', '.join(saved_files)}")

    def create_integrated_table(self, collected_data: Dict[str, Dict]) -> pd.DataFrame:
        """수집된 데이터를 기반으로 통합 테이블 생성"""
        logger.info("📊 통합 테이블 생성 중...")

        all_rows = []
        # total_pv에 있는 모든 날짜를 기준으로 반복
        all_dates = sorted(collected_data.get('total_pv', {}).keys())

        for date_str in all_dates:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            total_pv = collected_data.get('total_pv', {}).get(date_str, 0)
            
            base_row = {
                'date': date_str,
                'year': date_obj.year,
                'month': date_obj.month,
                'day': date_obj.day,
                'day_of_week': date_obj.strftime('%A'),
                'total_pv': total_pv
            }
            
            # 채널 데이터 처리
            channel_items = collected_data.get('channel_data', {}).get(date_str, [])
            if channel_items:
                for item in channel_items:
                    row = base_row.copy()
                    row.update({'data_type': 'channel', 'name': item.get('mapped_channel_name'), 'pv': item.get('pv')})
                    # API 응답의 다른 모든 키도 추가
                    row.update(item)
                    all_rows.append(row)
            
            # 키워드 데이터 처리
            keyword_items = collected_data.get('keyword_data', {}).get(date_str, [])
            if keyword_items:
                for item in keyword_items:
                    row = base_row.copy()
                    row.update({'data_type': 'keyword', 'name': item.get('ref_keyword'), 'pv': item.get('pv')})
                    # API 응답의 다른 모든 키도 추가
                    row.update(item)
                    all_rows.append(row)
            
            # 해당 날짜에 채널/키워드 데이터가 모두 없는 경우
            if not channel_items and not keyword_items:
                row = base_row.copy()
                row.update({'data_type': 'summary_only', 'name': None, 'pv': 0})
                all_rows.append(row)

        if not all_rows:
            logger.warning("통합할 데이터가 없습니다.")
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        
        # 모든 컬럼이 존재하는지 확인하고 없으면 None으로 추가
        for col in self.column_order:
            if col not in df.columns:
                df[col] = None
        
        logger.info(f"✅ 통합 테이블 생성 완료: {len(df)}행")
        return df[self.column_order] # 정의된 순서로 컬럼 정렬
    
    def save_processed_data(self, df: pd.DataFrame, year: int, month: int, client_name: str):
        """가공된 통합 데이터를 CSV 및 Excel 파일로 저장합니다."""
        logger.info("💾 가공 데이터 저장 중...")
        month_str = f"{month:02d}"

        # CSV 저장
        csv_file = self.processed_data_dir / f"{client_name}_{year}_{month_str}_integrated_statistics.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')

        # Excel 저장
        excel_file = self.processed_data_dir / f"{client_name}_{year}_{month_str}_daily_summary.xlsx"
        try:
            self._create_daily_summary_excel(df, excel_file, client_name, year, month)
            logger.info(f"✅ 가공 데이터 저장 완료: {csv_file}, {excel_file}")
        except Exception as e:
            logger.error(f"❌ Excel 파일 저장 실패: {e}", exc_info=True)
    
    def _create_daily_summary_excel(self, df: pd.DataFrame, excel_file: Path, client_name: str, year: int, month: int):
        """일자별 요약 엑셀 파일 생성"""
        logger.info("📊 일자별 요약 엑셀 파일 생성 중...")
        
        # 날짜별 데이터 그룹화
        daily_summary = []
        
        for date_str in sorted(df['date'].unique()):
            date_data = df[df['date'] == date_str]
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            
            # 기본 행 데이터
            row = {
                '날짜': date_str,
                '년': date_obj.year,
                '월': date_obj.month,
                '일': date_obj.day,
                '요일': date_obj.strftime('%A'),
                '플레이스 조회수 합계': date_data['total_pv'].iloc[0]  # 모든 행이 동일한 값
            }
            
            # 채널별 데이터 추가
            channel_data = date_data[date_data['data_type'] == 'channel']
            for _, channel_row in channel_data.iterrows():
                channel_name = channel_row['name']
                pv = channel_row['pv']
                row[f'채널_{channel_name}'] = pv
            
            # 키워드별 데이터 추가
            keyword_data = date_data[date_data['data_type'] == 'keyword']
            for _, keyword_row in keyword_data.iterrows():
                keyword_name = keyword_row['name']
                pv = keyword_row['pv']
                row[f'키워드_{keyword_name}'] = pv
            
            daily_summary.append(row)
        
        # DataFrame 생성
        summary_df = pd.DataFrame(daily_summary)
        
        # 채널별 총 PV 계산하여 정렬
        channel_columns = [col for col in summary_df.columns if col.startswith('채널_')]
        channel_totals = {}
        for col in channel_columns:
            channel_name = col.replace('채널_', '')
            total_pv = summary_df[col].sum()
            channel_totals[col] = total_pv
        
        # 채널 컬럼을 총 PV 기준으로 정렬 (높은 순)
        sorted_channel_columns = sorted(channel_columns, key=lambda x: channel_totals[x], reverse=True)
        
        # 키워드별 총 PV 계산하여 정렬
        keyword_columns = [col for col in summary_df.columns if col.startswith('키워드_')]
        keyword_totals = {}
        for col in keyword_columns:
            keyword_name = col.replace('키워드_', '')
            total_pv = summary_df[col].sum()
            keyword_totals[col] = total_pv
        
        # 키워드 컬럼을 총 PV 기준으로 정렬 (높은 순)
        sorted_keyword_columns = sorted(keyword_columns, key=lambda x: keyword_totals[x], reverse=True)
        
        # 최종 컬럼 순서 정리
        all_columns = ['날짜', '년', '월', '일', '요일', '플레이스 조회수 합계']
        all_columns.extend(sorted_channel_columns)
        all_columns.extend(sorted_keyword_columns)
        
        # 컬럼 순서 정리
        summary_df = summary_df[all_columns]
        
        # 엑셀 파일 저장
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # 일자별 요약 시트
            summary_df.to_excel(writer, sheet_name='일자별 요약', index=False)
            
            # 통계 요약 시트
            self._create_statistics_sheet(writer, summary_df, client_name, year, month)
            
            # 채널별 통계 시트
            self._create_channel_statistics_sheet(writer, summary_df, year, month)
            
            # 키워드별 통계 시트
            self._create_keyword_statistics_sheet(writer, summary_df, year, month)
        
        logger.info(f"✅ 엑셀 파일 생성 완료: {excel_file}")
        logger.info(f"  - 총 {len(summary_df)}일 데이터")
        logger.info(f"  - 채널 수: {len(sorted_channel_columns)}개")
        logger.info(f"  - 키워드 수: {len(sorted_keyword_columns)}개")
        
        # 상위 채널과 키워드 정보 출력
        top_channels = [(col.replace('채널_', ''), channel_totals[col]) for col in sorted_channel_columns[:5]]
        top_keywords = [(col.replace('키워드_', ''), keyword_totals[col]) for col in sorted_keyword_columns[:5]]
        
        logger.info(f"  - 상위 5개 채널: {[f'{name}({pv:.0f})' for name, pv in top_channels]}")
        logger.info(f"  - 상위 5개 키워드: {[f'{name}({pv:.0f})' for name, pv in top_keywords]}")
    
    def _create_statistics_sheet(self, writer, summary_df: pd.DataFrame, client_name: str, year: int, month: int):
        """통계 요약 시트 생성"""
        stats_data = []
        
        # 기본 통계
        total_pv = summary_df['플레이스 조회수 합계'].sum()
        avg_daily_pv = summary_df['플레이스 조회수 합계'].mean()
        max_daily_pv = summary_df['플레이스 조회수 합계'].max()
        min_daily_pv = summary_df['플레이스 조회수 합계'].min()
        
        stats_data.append([f'{client_name} {year}년 {month}월 통계 요약', ''])
        stats_data.append(['', ''])
        stats_data.append(['기본 통계', ''])
        stats_data.append(['총 PV', f"{total_pv:,.0f}"])
        stats_data.append(['일평균 PV', f"{avg_daily_pv:.1f}"])
        stats_data.append(['최대 일 PV', f"{max_daily_pv:,.0f}"])
        stats_data.append(['최소 일 PV', f"{min_daily_pv:,.0f}"])
        stats_data.append(['', ''])
        
        # 상위 10일
        top_days = summary_df.nlargest(10, '플레이스 조회수 합계')[['날짜', '요일', '플레이스 조회수 합계']]
        stats_data.append(['상위 10일 (PV 기준)', ''])
        stats_data.append(['날짜', '요일', '총 PV'])
        for _, row in top_days.iterrows():
            stats_data.append([row['날짜'], row['요일'], f"{row['플레이스 조회수 합계']:,.0f}"])
        
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_excel(writer, sheet_name='통계 요약', index=False, header=False)
    
    def _create_channel_statistics_sheet(self, writer, summary_df: pd.DataFrame, year: int, month: int):
        """채널별 통계 시트 생성"""
        channel_columns = [col for col in summary_df.columns if col.startswith('채널_')]
        
        if not channel_columns:
            return
        
        channel_stats = []
        for col in channel_columns:
            channel_name = col.replace('채널_', '')
            total_pv = summary_df[col].sum()
            avg_pv = summary_df[col].mean()
            max_pv = summary_df[col].max()
            days_with_data = (summary_df[col] > 0).sum()
            
            channel_stats.append({
                '채널명': channel_name,
                '총 PV': total_pv,
                '일평균 PV': avg_pv,
                '최대 일 PV': max_pv,
                '데이터 있는 일수': days_with_data
            })
        
        channel_stats_df = pd.DataFrame(channel_stats)
        channel_stats_df = channel_stats_df.sort_values('총 PV', ascending=False)
        channel_stats_df.to_excel(writer, sheet_name='채널별 통계', index=False)
    
    def _create_keyword_statistics_sheet(self, writer, summary_df: pd.DataFrame, year: int, month: int):
        """키워드별 통계 시트 생성"""
        keyword_columns = [col for col in summary_df.columns if col.startswith('키워드_')]
        
        if not keyword_columns:
            return
        
        keyword_stats = []
        for col in keyword_columns:
            keyword_name = col.replace('키워드_', '')
            total_pv = summary_df[col].sum()
            avg_pv = summary_df[col].mean()
            max_pv = summary_df[col].max()
            days_with_data = (summary_df[col] > 0).sum()
            
            keyword_stats.append({
                '키워드명': keyword_name,
                '총 PV': total_pv,
                '일평균 PV': avg_pv,
                '최대 일 PV': max_pv,
                '데이터 있는 일수': days_with_data
            })
        
        keyword_stats_df = pd.DataFrame(keyword_stats)
        keyword_stats_df = keyword_stats_df.sort_values('총 PV', ascending=False)
        keyword_stats_df.to_excel(writer, sheet_name='키워드별 통계', index=False)
    
    def run_monthly_analysis(self, year: int, month: int, client_name: str):
        """월별 분석 실행"""
        try:
            logger.info(f"🚀 {year}년 {month}월 플레이스 PV 통계 분석 시작")
            
            # 1. 데이터 수집
            collected_data = self.collect_monthly_data(year, month)
            if not collected_data.get("total_pv"):
                logger.warning(f"{year}년 {month}월 수집된 플레이스 PV 데이터가 없습니다.")
                return

            # 2. 원본 데이터 저장
            self.save_raw_data(collected_data, year, month, client_name)
            
            # 3. 통합 테이블 생성
            df = self.create_integrated_table(collected_data)
            if df.empty:
                logger.warning(f"{year}년 {month}월 데이터가 비어있어 가공 파일을 생성하지 않습니다.")
                return

            # 4. 가공 데이터 저장
            self.save_processed_data(df, year, month, client_name)
            
            logger.info(f"🎉 {year}년 {month}월 분석 완료!")

        except ApiCallError as e:
            logger.error(f"❌ API 호출 오류가 발생하여 {year}년 {month}월 분석을 중단합니다.")
            logger.error(f"   오류 메시지: {e}")
            logger.error("   .env 파일의 AUTH_TOKEN과 COOKIE 값을 최신으로 갱신해주세요.")
            # sys.exit(1) # 전체 스크립트 중단
            raise  # 예외를 다시 발생시켜 상위 호출자(run_crawler.py)가 처리하도록 함
        except Exception as e:
            logger.error(f"❌ {year}년 {month}월 분석 중 예상치 못한 오류 발생: {e}", exc_info=True)


def main_test():
    """스크립트 개별 테스트를 위한 메인 함수"""
    config_manager = get_config_manager()
    client = config_manager.get_selected_client_config()
    if not client:
        print("❌ 클라이언트 설정을 찾을 수 없습니다.")
        return
    
    auth_config = config_manager.get_auth_config()
    
    crawler = MonthlyStatisticsCrawler(client, auth_config)
    crawler.run_monthly_analysis(2024, 9, client.name)

if __name__ == "__main__":
    # 이 스크립트는 이제 run_crawler.py를 통해 실행되는 것이 기본입니다.
    # 단독으로 테스트하고 싶을 경우 아래 함수를 호출하세요.
    # main_test()
    print("이 스크립트는 단독 실행용이 아닙니다. scripts/run_crawler.py를 실행해주세요.")
