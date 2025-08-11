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

from scripts.util.config import get_config_manager
from scripts.crawler.stat_crawler import NaverStatCrawler

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonthlyStatisticsCrawler:
    """월별 통계 데이터 수집 및 통합 클래스"""
    
    def __init__(self):
        self.config_manager = get_config_manager()
        self.client = self.config_manager.get_selected_client_config()
        self.crawler = NaverStatCrawler()
        
        # 데이터 저장 경로
        self.raw_data_dir = project_root / "data" / "raw"
        self.processed_data_dir = project_root / "data" / "processed"
        
        # 디렉토리 생성
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
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
    
    def save_raw_data(self, year: int, month: int, client_name: str):
        """원본 데이터를 raw 폴더에 저장"""
        logger.info("💾 원본 데이터 저장 중...")
        
        # 채널 데이터 저장
        channel_file = self.raw_data_dir / f"{client_name}_channel_data_{year}_{month:02d}.json"
        with open(channel_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(self.daily_channel_data, f, ensure_ascii=False, indent=2, default=str)
        
        # 키워드 데이터 저장
        keyword_file = self.raw_data_dir / f"{client_name}_keyword_data_{year}_{month:02d}.json"
        with open(keyword_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(self.daily_keyword_data, f, ensure_ascii=False, indent=2, default=str)
        
        # 총 PV 데이터 저장
        total_pv_file = self.raw_data_dir / f"{client_name}_total_pv_{year}_{month:02d}.json"
        with open(total_pv_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(self.daily_total_pv, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"✅ 원본 데이터 저장 완료: {channel_file}, {keyword_file}, {total_pv_file}")
    
    def create_integrated_table(self, year: int, month: int) -> pd.DataFrame:
        """통합 테이블 생성"""
        logger.info("📊 통합 테이블 생성 중...")
        
        rows = []
        
        # 모든 날짜에 대해 데이터 생성
        for date_str in sorted(self.daily_channel_data.keys()):
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            
            # 기본 행 (총 PV)
            base_row = {
                'date': date_str,
                'year': date_obj.year,
                'month': date_obj.month,
                'day': date_obj.day,
                'day_of_week': date_obj.strftime('%A'),
                'total_pv': self.daily_total_pv.get(date_str, 0)
            }
            
            # 채널별 데이터 추가
            channel_data = self.daily_channel_data.get(date_str, [])
            for item in channel_data:
                channel_name = item.get('mapped_channel_name', 'Unknown')
                pv = item.get('pv', 0)
                
                row = base_row.copy()
                row['data_type'] = 'channel'
                row['name'] = channel_name
                row['pv'] = pv
                row['channel_id'] = item.get('mapped_channel_id', '')
                row['channel_type'] = item.get('mapped_channel_type', '')
                row['channel_category'] = item.get('mapped_channel_category', '')
                row['keyword'] = ''  # 채널 데이터는 키워드 없음
                row['keyword_id'] = ''
                row['keyword_type'] = ''
                row['keyword_category'] = ''
                
                rows.append(row)
            
            # 키워드별 데이터 추가
            keyword_data = self.daily_keyword_data.get(date_str, [])
            for item in keyword_data:
                keyword_name = item.get('ref_keyword', 'Unknown')
                pv = item.get('pv', 0)
                
                row = base_row.copy()
                row['data_type'] = 'keyword'
                row['name'] = keyword_name
                row['pv'] = pv
                row['channel_id'] = ''  # 키워드 데이터는 채널 정보 없음
                row['channel_type'] = ''
                row['channel_category'] = ''
                row['keyword'] = keyword_name
                row['keyword_id'] = item.get('ref_keyword_id', '')
                row['keyword_type'] = item.get('ref_keyword_type', '')
                row['keyword_category'] = item.get('ref_keyword_category', '')
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # 컬럼 순서 정리
        column_order = [
            'date', 'year', 'month', 'day', 'day_of_week', 'data_type',
            'name', 'pv', 'total_pv',
            'channel_id', 'channel_type', 'channel_category',
            'keyword', 'keyword_id', 'keyword_type', 'keyword_category'
        ]
        
        df = df[column_order]
        
        logger.info(f"✅ 통합 테이블 생성 완료: {len(df)}행")
        return df
    
    def save_processed_data(self, df: pd.DataFrame, year: int, month: int, client_name: str):
        """가공된 데이터를 processed 폴더에 저장"""
        logger.info("💾 가공 데이터 저장 중...")
        
        # CSV 파일로 저장
        output_file = self.processed_data_dir / f"{client_name}_{year}_{month:02d}_integrated_statistics.csv"
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        # 엑셀 파일 생성 (날짜별 요약 테이블)
        excel_file = self.processed_data_dir / f"{client_name}_{year}_{month:02d}_daily_summary.xlsx"
        self._create_daily_summary_excel(df, excel_file, year, month, client_name)
        
        logger.info(f"✅ 가공 데이터 저장 완료: {output_file}, {excel_file}")
        logger.info(f"📊 통계 요약:")
        logger.info(f"  - 총 레코드 수: {len(df):,}개")
        logger.info(f"  - 채널 데이터: {len(df[df['data_type'] == 'channel']):,}개")
        logger.info(f"  - 키워드 데이터: {len(df[df['data_type'] == 'keyword']):,}개")
        logger.info(f"  - 총 PV: {df['pv'].sum():,}")
        logger.info(f"  - 일평균 총 PV: {df.groupby('date')['total_pv'].first().mean():.1f}")
    
    def _create_daily_summary_excel(self, df: pd.DataFrame, excel_file: Path, year: int, month: int, client_name: str):
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
            self._create_statistics_sheet(writer, summary_df, year, month, client_name)
            
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
    
    def _create_statistics_sheet(self, writer, summary_df: pd.DataFrame, year: int, month: int, client_name: str):
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
        logger.info(f"🚀 {year}년 {month}월 통계 분석 시작")
        
        try:
            # 1. 데이터 수집
            self.collect_monthly_data(year, month)
            
            # 2. 원본 데이터 저장
            self.save_raw_data(year, month, client_name)
            
            # 3. 통합 테이블 생성
            df = self.create_integrated_table(year, month)
            
            # 4. 가공 데이터 저장
            self.save_processed_data(df, year, month, client_name)
            
            logger.info(f"🎉 {year}년 {month}월 분석 완료!")
            return True
            
        except Exception as e:
            logger.error(f"❌ 월별 분석 실패: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """메인 함수"""
    print("🚀 월별 통계 데이터 수집 및 통합 스크립트")
    
    try:
        # 설정 확인
        config_manager = get_config_manager()
        client = config_manager.get_selected_client_config()
        
        if not client:
            print("❌ 클라이언트 설정을 찾을 수 없습니다.")
            return
        
        print(f"✅ 선택된 클라이언트: {client.name}")
        
        # 연월 입력 받기
        while True:
            try:
                year_month = input("\n📅 수집할 연월을 입력하세요 (예: 2025-07): ").strip()
                if len(year_month) == 7 and year_month[4] == '-':
                    year = int(year_month[:4])
                    month = int(year_month[5:7])
                    if 2020 <= year <= 2030 and 1 <= month <= 12:
                        break
                    else:
                        print("❌ 유효하지 않은 연월입니다. 2020-2030년, 1-12월 범위로 입력해주세요.")
                else:
                    print("❌ 형식이 올바르지 않습니다. 'YYYY-MM' 형식으로 입력해주세요.")
            except ValueError:
                print("❌ 숫자 형식이 올바르지 않습니다. 'YYYY-MM' 형식으로 입력해주세요.")
        
        print(f"📊 {year}년 {month}월 데이터 수집을 시작합니다...")
        
        # 월별 분석 실행
        crawler = MonthlyStatisticsCrawler()
        success = crawler.run_monthly_analysis(year, month, client.name)
        
        if success:
            print("🎉 월별 분석이 성공적으로 완료되었습니다!")
            print(f"📁 결과 파일 위치:")
            print(f"  - 원본 데이터: {crawler.raw_data_dir}")
            print(f"  - 가공 데이터: {crawler.processed_data_dir}")
        else:
            print("❌ 월별 분석이 실패했습니다.")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
