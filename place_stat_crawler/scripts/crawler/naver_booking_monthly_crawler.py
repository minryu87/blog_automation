#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
월별 예약 통계 데이터 수집 및 통합 스크립트
지정된 연월의 일자별 예약 통계를 수집하여 통합 테이블 생성
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

from scripts.util.config import get_config_manager, ClientInfo, AuthConfig
from scripts.crawler.naver_booking_stat_crawler import NaverBookingStatCrawler
from scripts.crawler.naver_place_pv_crawler_base import ApiCallError

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonthlyBookingCrawler:
    """월별 예약 통계 데이터 수집 및 통합 클래스"""
    
    def __init__(self, client_info: ClientInfo, auth_config: AuthConfig):
        """
        Args:
            client_info: 사용할 클라이언트의 정보
            auth_config: 인증 관련 설정 정보
        """
        self.client = client_info
        self.crawler = NaverBookingStatCrawler(client_info, auth_config)
        
        # 데이터 저장 경로
        self.raw_data_dir = project_root / "data" / "raw"
        self.processed_data_dir = project_root / "data" / "processed"
        
        # 디렉토리 생성
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # 수집된 데이터 저장소
        self.daily_booking_data = {}  # {date: {page_visits, booking_requests, channel_stats}}
    
    def collect_monthly_data(self, year: int, month: int) -> List[Dict]:
        """지정된 년월의 데이터 수집"""
        logger.info(f"🚀 {year}년 {month}월 예약 데이터 수집 시작")
        
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
            logger.info(f"📊 {date_str} 예약 데이터 수집 중...")
            
            try:
                # 일자별 예약 데이터 수집
                daily_data = self.crawler.fetch_booking_data_for_date(date_str)
                self.daily_booking_data[date_str] = daily_data
                
                # 요약 정보 출력
                page_visits = daily_data.get('page_visits', [])
                booking_requests = daily_data.get('booking_requests', [])
                channel_stats = daily_data.get('channel_stats', [])
                
                page_visit_count = page_visits[0]['count'] if page_visits else 0
                booking_request_count = booking_requests[0]['count'] if booking_requests else 0
                
                logger.info(f"✅ {date_str} 완료 - 페이지 유입: {page_visit_count}, 예약 신청: {booking_request_count}, 채널: {len(channel_stats)}개")
                
            except Exception as e:
                logger.error(f"❌ {date_str} 데이터 수집 실패: {e}")
                # 빈 데이터로 설정
                self.daily_booking_data[date_str] = {
                    'date': date_str,
                    'page_visits': [],
                    'booking_requests': [],
                    'channel_stats': []
                }
            
            current_date += timedelta(days=1)
        
        logger.info(f"🎉 {year}년 {month}월 예약 데이터 수집 완료")
        return self.daily_booking_data
    
    def save_raw_data(self, all_data: List[Dict], year: int, month: int, client_name: str):
        """수집한 원본 데이터를 JSON 파일로 저장합니다."""
        logger.info("💾 원본 데이터 저장 중...")
        month_str = f"{month:02d}"
        file_path = self.raw_data_dir / f"{client_name}_booking_data_{year}_{month_str}.json"
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(all_data, f, ensure_ascii=False, indent=4)
            if not file_path.exists():
                logger.error(f"‼️ 파일 생성 실패! 저장 경로를 확인하세요: {file_path}")
                sys.exit(1)
            logger.info(f"✅ 파일 생성 확인 완료: {file_path}")
        except (IOError, TypeError) as e:
            logger.error(f"❌ 원본 데이터 저장 실패: {e}", exc_info=True)

    def create_integrated_table(self, collected_data: Dict[str, Dict]) -> pd.DataFrame:
        """수집된 데이터를 기반으로 통합 테이블 생성"""
        logger.info("📊 통합 테이블 생성 중...")
        all_rows = []
        for date_str, daily_data in collected_data.items():
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            
            # 페이지 유입 수와 예약 신청 수
            page_visits = daily_data.get('page_visits', [])
            booking_requests = daily_data.get('booking_requests', [])
            
            page_visit_count = page_visits[0]['count'] if page_visits else 0
            booking_request_count = booking_requests[0]['count'] if booking_requests else 0
            
            # 기본 행 (요약)
            base_row = {
                'date': date_str,
                'year': date_obj.year,
                'month': date_obj.month,
                'day': date_obj.day,
                'day_of_week': date_obj.strftime('%A'),
                'page_visits': page_visit_count,
                'booking_requests': booking_request_count
            }
            
            # 채널별 데이터 추가
            channel_stats = daily_data.get('channel_stats', [])
            for channel in channel_stats:
                row = base_row.copy()
                row['data_type'] = 'channel'
                row['channel_code'] = channel.get('channel_code', '')
                row['channel_name'] = channel.get('channel_name', '')
                row['channel_count'] = channel.get('count', 0)
                
                all_rows.append(row)
            
            # 요약 행 추가 (채널별 데이터가 없는 경우)
            if not channel_stats:
                row = base_row.copy()
                row['data_type'] = 'summary'
                row['channel_code'] = ''
                row['channel_name'] = ''
                row['channel_count'] = 0
                
                all_rows.append(row)
        
        df = pd.DataFrame(all_rows)
        
        # 컬럼 순서 정리
        column_order = [
            'date', 'year', 'month', 'day', 'day_of_week', 'data_type',
            'channel_code', 'channel_name', 'channel_count', 'page_visits', 'booking_requests'
        ]
        
        df = df[column_order]
        
        logger.info(f"✅ 통합 테이블 생성 완료: {len(df)}행")
        return df
    
    def save_processed_data(self, df: pd.DataFrame, year: int, month: int, client_name: str):
        """가공된 통합 데이터를 CSV 및 Excel 파일로 저장합니다."""
        logger.info("💾 가공 데이터 저장 중...")
        month_str = f"{month:02d}"

        # CSV 저장
        csv_file = self.processed_data_dir / f"{client_name}_{year}_{month_str}_booking_integrated_statistics.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')

        # Excel 저장
        excel_file = self.processed_data_dir / f"{client_name}_{year}_{month_str}_booking_daily_summary.xlsx"
        try:
            self._create_booking_daily_summary_excel(df, excel_file, client_name, year, month)
            logger.info(f"✅ 가공 데이터 저장 완료: {csv_file}, {excel_file}")
        except Exception as e:
            logger.error(f"❌ Excel 파일 저장 실패: {e}", exc_info=True)
    
    def _create_booking_daily_summary_excel(self, df: pd.DataFrame, excel_file: Path, year: int, month: int, client_name: str):
        """일자별 요약 엑셀 파일 생성"""
        logger.info("📊 일자별 예약 요약 엑셀 파일 생성 중...")
        
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
                '예약 페이지 유입 수': date_data['page_visits'].iloc[0],
                '예약 신청 수': date_data['booking_requests'].iloc[0]
            }
            
            # 채널별 데이터 추가
            channel_data = date_data[date_data['data_type'] == 'channel']
            for _, channel_row in channel_data.iterrows():
                channel_name = channel_row['channel_name']
                count = channel_row['channel_count']
                row[f'채널_{channel_name}'] = count
            
            daily_summary.append(row)
        
        # DataFrame 생성
        summary_df = pd.DataFrame(daily_summary)
        
        # 채널별 총 count 계산하여 정렬
        channel_columns = [col for col in summary_df.columns if col.startswith('채널_')]
        channel_totals = {}
        for col in channel_columns:
            channel_name = col.replace('채널_', '')
            total_count = summary_df[col].sum()
            channel_totals[col] = total_count
        
        # 채널 컬럼을 총 count 기준으로 정렬 (높은 순)
        sorted_channel_columns = sorted(channel_columns, key=lambda x: channel_totals[x], reverse=True)
        
        # 최종 컬럼 순서 정리
        all_columns = ['날짜', '년', '월', '일', '요일', '예약 페이지 유입 수', '예약 신청 수']
        all_columns.extend(sorted_channel_columns)
        
        # 컬럼 순서 정리
        summary_df = summary_df[all_columns]
        
        # 엑셀 파일 저장
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # 일자별 요약 시트
            summary_df.to_excel(writer, sheet_name='일자별 요약', index=False)
            
            # 통계 요약 시트
            self._create_booking_statistics_sheet(writer, summary_df, year, month, client_name)
            
            # 채널별 통계 시트
            self._create_booking_channel_statistics_sheet(writer, summary_df, year, month)
        
        logger.info(f"✅ 엑셀 파일 생성 완료: {excel_file}")
        logger.info(f"  - 총 {len(summary_df)}일 데이터")
        logger.info(f"  - 채널 수: {len(sorted_channel_columns)}개")
        
        # 상위 채널 정보 출력
        top_channels = [(col.replace('채널_', ''), channel_totals[col]) for col in sorted_channel_columns[:5]]
        logger.info(f"  - 상위 5개 채널: {[f'{name}({count:.0f})' for name, count in top_channels]}")
    
    def _create_booking_statistics_sheet(self, writer, summary_df: pd.DataFrame, year: int, month: int, client_name: str):
        """예약 통계 요약 시트 생성"""
        stats_data = []
        
        # 기본 통계
        total_page_visits = summary_df['예약 페이지 유입 수'].sum()
        total_booking_requests = summary_df['예약 신청 수'].sum()
        avg_page_visits = summary_df['예약 페이지 유입 수'].mean()
        avg_booking_requests = summary_df['예약 신청 수'].mean()
        
        stats_data.append([f'{client_name} {year}년 {month}월 예약 통계 요약', ''])
        stats_data.append(['', ''])
        stats_data.append(['기본 통계', ''])
        stats_data.append(['총 예약 페이지 유입 수', f"{total_page_visits:,.0f}"])
        stats_data.append(['총 예약 신청 수', f"{total_booking_requests:,.0f}"])
        stats_data.append(['일평균 페이지 유입 수', f"{avg_page_visits:.1f}"])
        stats_data.append(['일평균 예약 신청 수', f"{avg_booking_requests:.1f}"])
        stats_data.append(['', ''])
        
        # 상위 10일 (페이지 유입 기준)
        top_days = summary_df.nlargest(10, '예약 페이지 유입 수')[['날짜', '요일', '예약 페이지 유입 수', '예약 신청 수']]
        stats_data.append(['상위 10일 (페이지 유입 기준)', ''])
        stats_data.append(['날짜', '요일', '페이지 유입 수', '예약 신청 수'])
        for _, row in top_days.iterrows():
            stats_data.append([row['날짜'], row['요일'], f"{row['예약 페이지 유입 수']:,.0f}", f"{row['예약 신청 수']:,.0f}"])
        
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_excel(writer, sheet_name='통계 요약', index=False, header=False)
    
    def _create_booking_channel_statistics_sheet(self, writer, summary_df: pd.DataFrame, year: int, month: int):
        """채널별 예약 통계 시트 생성"""
        channel_columns = [col for col in summary_df.columns if col.startswith('채널_')]
        
        if not channel_columns:
            return
        
        channel_stats = []
        for col in channel_columns:
            channel_name = col.replace('채널_', '')
            total_count = summary_df[col].sum()
            avg_count = summary_df[col].mean()
            max_count = summary_df[col].max()
            days_with_data = (summary_df[col] > 0).sum()
            
            channel_stats.append({
                '채널명': channel_name,
                '총 유입 수': total_count,
                '일평균 유입 수': avg_count,
                '최대 일 유입 수': max_count,
                '데이터 있는 일수': days_with_data
            })
        
        channel_stats_df = pd.DataFrame(channel_stats)
        channel_stats_df = channel_stats_df.sort_values('총 유입 수', ascending=False)
        channel_stats_df.to_excel(writer, sheet_name='채널별 통계', index=False)
    
    def run_monthly_analysis(self, year: int, month: int, client_name: str):
        """월별 분석 실행"""
        try:
            logger.info(f"🚀 {year}년 {month}월 예약 통계 분석 시작")

            # 1. 데이터 수집
            all_data = self.collect_monthly_data(year, month)
            if not all_data:
                logger.warning(f"{year}년 {month}월 수집된 예약 데이터가 없습니다. 다음 달로 넘어갑니다.")
                return

            total_requests = sum(d.get('booking_requests', [{}])[0].get('count', 0) for d in all_data.values() if d.get('booking_requests'))
            if total_requests == 0:
                logger.warning(f"⚠️ {year}년 {month}월의 총 예약 신청 수가 0입니다.")

            # 2. 원본 데이터 저장
            self.save_raw_data(all_data, year, month, client_name)

            # 3. 통합 테이블 생성
            df = self.create_integrated_table(all_data)
            if df.empty:
                logger.warning(f"{year}년 {month}월 데이터가 비어있어 가공 파일을 생성하지 않습니다.")
                return

            # 4. 가공 데이터 저장
            self.save_processed_data(df, year, month, client_name)
            
            logger.info(f"🎉 {year}년 {month}월 분석 완료!")

        except ApiCallError as e:
            logger.error(f"❌ API 호출 오류가 발생하여 {year}년 {month}월 예약 분석을 중단합니다.")
            logger.error(f"   오류 메시지: {e}")
            logger.error("   .env 파일의 AUTH_TOKEN과 COOKIE 값을 최신으로 갱신해주세요.")
            raise # 예외를 다시 발생시켜 상위 호출자(run_crawler.py)가 처리하도록 함
        except Exception as e:
            logger.critical(f"💥 {year}년 {month}월 분석 중 심각한 오류 발생. 스크립트를 중단합니다.", exc_info=True)
            sys.exit(1)


def main_test():
    """스크립트 개별 테스트를 위한 메인 함수"""
    config_manager = get_config_manager()
    client = config_manager.get_selected_client_config()
    if not client:
        print("❌ 클라이언트 설정을 찾을 수 없습니다.")
        return
        
    auth_config = config_manager.get_auth_config()

    crawler = MonthlyBookingCrawler(client, auth_config)
    crawler.run_monthly_analysis(2024, 9, client.name)

if __name__ == "__main__":
    # 이 스크립트는 이제 run_crawler.py를 통해 실행되는 것이 기본입니다.
    # 단독으로 테스트하고 싶을 경우 아래 함수를 호출하세요.
    # main_test()
    print("이 스크립트는 단독 실행용이 아닙니다. scripts/run_crawler.py를 실행해주세요.")
