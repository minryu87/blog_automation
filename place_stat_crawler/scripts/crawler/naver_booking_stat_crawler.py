#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
네이버 예약 통계 크롤러
예약 신청 수, 예약 페이지 유입 수, 채널별 예약 페이지 유입 수를 수집합니다.
"""

import sys
import os
import pandas as pd
import requests
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# 상대 import를 절대 import로 변경
try:
    from scripts.crawler.naver_place_pv_crawler_base import NaverCrawlerBase
    from scripts.crawler.naver_place_pv_auth_manager import NaverAuthManager
except ImportError:
    # 직접 실행 시를 위한 절대 import
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent.parent
    sys.path.insert(0, str(parent_dir))
    sys.path.insert(0, str(current_dir))
    from scripts.crawler.naver_place_pv_crawler_base import NaverCrawlerBase
    from scripts.crawler.naver_place_pv_auth_manager import NaverAuthManager

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NaverBookingStatCrawler(NaverCrawlerBase):
    """네이버 예약 통계 크롤러"""
    
    def __init__(self, auth_manager: Optional[NaverAuthManager] = None):
        super().__init__(auth_manager)
        
        # 예약 통계 API URL
        self.booking_stat_url = "https://new.smartplace.naver.com/api/statistics/booking"
        self.booking_channel_url = "https://partner.booking.naver.com/api/businesses"
        
        # 채널 코드 매핑
        self.channel_mapping = {
            'bee': '기타',
            'bet': '외부서비스',
            'bmp': '지도',
            'bnb': '블로그',
            'bne': '네이버기타',
            'ple': '플레이스상세',
            'pll': '플레이스목록',
            'plt': 'PC플랫폼',
            'psa': '플레이스광고'
        }
        
        # 클라이언트 ID (기본값, 실제로는 설정에서 가져옴)
        self.client_id = "563688"  # 예시 ID
    
    def set_client_id(self, client_id: str):
        """클라이언트 ID 설정"""
        if not client_id:
            raise ValueError("클라이언트 ID(booking_id)가 설정되지 않았습니다.")
        self.client_id = client_id
        logger.info(f"클라이언트 ID 설정: {client_id}")
    
    def fetch_booking_statistics(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """예약 통계 데이터 수집 (예약 신청 수, 예약 페이지 유입 수)"""
        logger.info(f"📊 {start_date} ~ {end_date} 예약 통계 수집 중...")
        
        url = f"{self.booking_stat_url}/{self.client_id}"
        params = {
            'bucket': 'sessionCount_sum,day_trend',
            'startDate': start_date,
            'endDate': end_date,
            'metric': 'UV,REQUESTED,COMPLETED,CONFIRMED'
        }
        
        try:
            response = self.make_request('GET', url, params=params)
            
            if response and 'result' in response:
                logger.info(f"✅ 예약 통계 수집 완료: {len(response['result'])}개 데이터")
                return response
            else:
                logger.error(f"❌ 예약 통계 응답 형식 오류: {response}")
                return {'result': []}
                
        except Exception as e:
            logger.error(f"❌ 예약 통계 수집 실패: {e}")
            return {'result': []}
    
    def fetch_booking_channel_statistics(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """채널별 예약 페이지 유입 수 수집"""
        logger.info(f"📊 {start_date} ~ {end_date} 채널별 예약 통계 수집 중...")
        
        url = f"{self.booking_channel_url}/{self.client_id}/reports"
        params = {
            'bucket': 'areaCode,day_trend',
            'startDate': start_date,
            'endDate': end_date,
            'metric': 'UV'
        }
        
        try:
            response = self.make_request('GET', url, params=params)
            
            if response and 'result' in response:
                logger.info(f"✅ 채널별 예약 통계 수집 완료: {len(response['result'])}개 데이터")
                return response
            else:
                logger.error(f"❌ 채널별 예약 통계 응답 형식 오류: {response}")
                return {'result': []}
                
        except Exception as e:
            logger.error(f"❌ 채널별 예약 통계 수집 실패: {e}")
            return {'result': []}
    
    def parse_booking_statistics(self, data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """예약 통계 데이터 파싱"""
        result = {
            'page_visits': [],  # 예약 페이지 유입 수 (UV)
            'booking_requests': []  # 예약 신청 수 (REQUESTED)
        }
        
        if 'result' not in data:
            return result
        
        for item in data['result']:
            date = item.get('day_trend', '')
            count = item.get('count', 0)
            metric = item.get('metric', '')
            
            if metric == 'UV':
                result['page_visits'].append({
                    'date': date,
                    'count': count,
                    'sessionCount_sum': item.get('sessionCount_sum', 0)
                })
            elif metric == 'REQUESTED':
                result['booking_requests'].append({
                    'date': date,
                    'count': count,
                    'sessionCount_sum': item.get('sessionCount_sum', 0)
                })
        
        logger.info(f"📊 파싱 완료 - 페이지 유입: {len(result['page_visits'])}일, 예약 신청: {len(result['booking_requests'])}일")
        return result
    
    def parse_channel_statistics(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """채널별 예약 통계 데이터 파싱"""
        result = []
        
        if 'result' not in data:
            return result
        
        for item in data['result']:
            date = item.get('day_trend', '')
            count = item.get('count', 0)
            area_code = item.get('areaCode', '')
            
            # 채널명 매핑
            channel_name = self.channel_mapping.get(area_code, area_code)
            
            result.append({
                'date': date,
                'channel_code': area_code,
                'channel_name': channel_name,
                'count': count
            })
        
        logger.info(f"📊 채널별 통계 파싱 완료: {len(result)}개 데이터")
        return result
    
    def fetch_booking_data_for_date(self, date: str) -> Dict[str, Any]:
        """특정 날짜의 예약 데이터 수집"""
        logger.info(f"📊 {date} 예약 데이터 수집 중...")
        
        # 예약 통계 수집
        booking_stats = self.fetch_booking_statistics(date, date)
        parsed_stats = self.parse_booking_statistics(booking_stats)
        
        # 채널별 통계 수집
        channel_stats = self.fetch_booking_channel_statistics(date, date)
        parsed_channels = self.parse_channel_statistics(channel_stats)
        
        # 결과 통합
        result = {
            'date': date,
            'page_visits': parsed_stats['page_visits'],
            'booking_requests': parsed_stats['booking_requests'],
            'channel_stats': parsed_channels
        }
        
        # 요약 정보
        total_page_visits = sum(item['count'] for item in parsed_stats['page_visits'])
        total_booking_requests = sum(item['count'] for item in parsed_stats['booking_requests'])
        total_channels = len(parsed_channels)
        
        logger.info(f"✅ {date} 완료 - 페이지 유입: {total_page_visits}, 예약 신청: {total_booking_requests}, 채널: {total_channels}개")
        
        return result
    
    def collect_booking_data(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """기간별 예약 데이터 수집"""
        logger.info(f"🚀 {start_date} ~ {end_date} 예약 데이터 수집 시작")
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        current = start
        
        all_data = []
        
        while current <= end:
            date_str = current.strftime('%Y-%m-%d')
            
            try:
                daily_data = self.fetch_booking_data_for_date(date_str)
                all_data.append(daily_data)
            except Exception as e:
                logger.error(f"❌ {date_str} 데이터 수집 실패: {e}")
                # 빈 데이터로 추가
                all_data.append({
                    'date': date_str,
                    'page_visits': [],
                    'booking_requests': [],
                    'channel_stats': []
                })
            
            current += timedelta(days=1)
        
        logger.info(f"🎉 예약 데이터 수집 완료: {len(all_data)}일")
        return all_data
    
    def create_booking_dataframe(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """예약 데이터를 DataFrame으로 변환"""
        logger.info("📊 예약 데이터 DataFrame 생성 중...")
        
        rows = []
        
        for daily_data in data:
            date = daily_data['date']
            
            # 페이지 유입 수
            page_visits = daily_data.get('page_visits', [])
            page_visit_count = page_visits[0]['count'] if page_visits else 0
            
            # 예약 신청 수
            booking_requests = daily_data.get('booking_requests', [])
            booking_request_count = booking_requests[0]['count'] if booking_requests else 0
            
            # 채널별 데이터
            channel_stats = daily_data.get('channel_stats', [])
            for channel in channel_stats:
                row = {
                    'date': date,
                    'data_type': 'channel',
                    'channel_code': channel['channel_code'],
                    'channel_name': channel['channel_name'],
                    'count': channel['count'],
                    'page_visits': page_visit_count,
                    'booking_requests': booking_request_count
                }
                rows.append(row)
            
            # 요약 행 추가 (채널별 데이터가 없는 경우)
            if not channel_stats:
                row = {
                    'date': date,
                    'data_type': 'summary',
                    'channel_code': '',
                    'channel_name': '',
                    'count': 0,
                    'page_visits': page_visit_count,
                    'booking_requests': booking_request_count
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # 날짜 컬럼 추가
        df['date_obj'] = pd.to_datetime(df['date'])
        df['year'] = df['date_obj'].dt.year
        df['month'] = df['date_obj'].dt.month
        df['day'] = df['date_obj'].dt.day
        df['day_of_week'] = df['date_obj'].dt.day_name()
        
        # 컬럼 순서 정리
        column_order = [
            'date', 'year', 'month', 'day', 'day_of_week', 'data_type',
            'channel_code', 'channel_name', 'count', 'page_visits', 'booking_requests'
        ]
        
        df = df[column_order]
        
        logger.info(f"✅ DataFrame 생성 완료: {len(df)}행")
        return df
    
    def calculate_booking_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """예약 통계 계산"""
        logger.info("📊 예약 통계 계산 중...")
        
        # 일자별 요약 통계
        daily_stats = df.groupby('date').agg({
            'page_visits': 'first',
            'booking_requests': 'first'
        }).reset_index()
        
        # 채널별 통계
        channel_stats = df[df['data_type'] == 'channel'].groupby('channel_name').agg({
            'count': 'sum',
            'page_visits': 'sum',
            'booking_requests': 'sum'
        }).reset_index()
        
        # 전체 통계
        total_stats = {
            'total_page_visits': df['page_visits'].sum(),
            'total_booking_requests': df['booking_requests'].sum(),
            'avg_page_visits_per_day': df.groupby('date')['page_visits'].first().mean(),
            'avg_booking_requests_per_day': df.groupby('date')['booking_requests'].first().mean(),
            'total_channels': len(channel_stats),
            'total_days': len(daily_stats)
        }
        
        logger.info(f"✅ 통계 계산 완료 - 총 페이지 유입: {total_stats['total_page_visits']}, 총 예약 신청: {total_stats['total_booking_requests']}")
        
        return {
            'daily_stats': daily_stats,
            'channel_stats': channel_stats,
            'total_stats': total_stats
        }
    
    def save_booking_files(self, df: pd.DataFrame, stats: Dict[str, Any], output_dir: str = ".", client_name: str = "", year: int = 0, month: int = 0):
        """예약 데이터 파일 저장"""
        logger.info("💾 예약 데이터 파일 저장 중...")
        
        # 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 파일명 생성
        base_filename = f"{client_name}_{year}_{month:02d}_booking_statistics" if client_name and year and month else "booking_statistics"
        
        # CSV 파일 저장
        csv_file = os.path.join(output_dir, f"{base_filename}.csv")
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        # Excel 파일 저장
        excel_file = os.path.join(output_dir, f"{base_filename}.xlsx")
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # 메인 데이터 시트
            df.to_excel(writer, sheet_name='예약 통계', index=False)
            
            # 일자별 요약 시트
            stats['daily_stats'].to_excel(writer, sheet_name='일자별 요약', index=False)
            
            # 채널별 통계 시트
            stats['channel_stats'].to_excel(writer, sheet_name='채널별 통계', index=False)
            
            # 전체 통계 시트
            total_stats_df = pd.DataFrame([stats['total_stats']])
            total_stats_df.to_excel(writer, sheet_name='전체 통계', index=False)
        
        logger.info(f"✅ 파일 저장 완료: {csv_file}, {excel_file}")
        return csv_file, excel_file


def main():
    """메인 함수 - 테스트용"""
    print("🚀 네이버 예약 통계 크롤러 테스트")
    
    try:
        # 크롤러 생성
        crawler = NaverBookingStatCrawler()
        
        # 테스트 데이터 수집
        test_date = "2025-07-01"
        data = crawler.fetch_booking_data_for_date(test_date)
        
        print(f"✅ 테스트 완료: {test_date}")
        print(f"페이지 유입: {len(data['page_visits'])}개")
        print(f"예약 신청: {len(data['booking_requests'])}개")
        print(f"채널별 통계: {len(data['channel_stats'])}개")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")


if __name__ == "__main__":
    main()
