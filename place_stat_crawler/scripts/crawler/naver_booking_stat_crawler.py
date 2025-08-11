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
    from scripts.crawler.naver_place_pv_crawler_base import NaverCrawlerBase, ApiCallError
    from scripts.crawler.naver_place_pv_auth_manager import NaverAuthManager
    from scripts.util.logger import logger
    from scripts.util.config import ClientInfo, AuthConfig
except ImportError:
    # 직접 실행 시를 위한 절대 import
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent.parent
    sys.path.insert(0, str(parent_dir))
    sys.path.insert(0, str(current_dir))
    from scripts.crawler.naver_place_pv_crawler_base import NaverCrawlerBase, ApiCallError
    from scripts.crawler.naver_place_pv_auth_manager import NaverAuthManager
    from scripts.util.logger import logger
    from scripts.util.config import ClientInfo, AuthConfig

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NaverBookingStatCrawler(NaverCrawlerBase):
    """네이버 예약 통계 데이터 크롤러"""

    def __init__(self, client_info: ClientInfo, auth_config: AuthConfig):
        """
        Args:
            client_info: 사용할 클라이언트의 정보
            auth_config: 인증 관련 설정 정보 (여기서는 사용되지 않음)
        """
        # Booking 크롤러는 NaverAuthManager를 사용하지 않음
        self.client_info = client_info
        super().__init__(client_info=client_info, auth_manager=None) # 부모 클래스 초기화 (auth_manager 없이)
        
        self.booking_key = self.client_info.booking_key
        if not self.booking_key:
            raise ValueError(f"클라이언트 '{self.client_info.name}'의 BOOKING_KEY가 .env에 설정되지 않았습니다.")

        # 예약 통계 API URL (새로운 엔드포인트로 변경)
        self.booking_stat_url = f"https://partner.booking.naver.com/api/businesses/{self.booking_key}/reports"
        
        # 채널 코드 매핑
        self.channel_mapping = {
            'bee': '기타', 'bet': '외부서비스', 'bmp': '지도', 'bnb': '블로그',
            'bne': '네이버기타', 'ple': '플레이스상세', 'pll': '플레이스목록',
            'plt': 'PC플랫폼', 'psa': '플레이스광고'
        }
        
        # 클라이언트 ID (기본값, 실제로는 설정에서 가져옴)
        self.client_id = "563688"  # 예시 ID
    
    def set_client_id(self, client_id: str):
        """클라이언트 ID 설정"""
        if not client_id:
            raise ValueError("클라이언트 ID(booking_id)가 설정되지 않았습니다.")
        self.client_id = client_id
        logger.info(f"클라이언트 ID 설정: {client_id}")
    
    def get_cookies(self) -> Dict[str, str]:
        """Booking용 쿠키를 파싱하여 반환합니다. (인증 방식 유지)"""
        cookie_str = self.client_info.booking_cookie or ""
        cookies = {}
        for part in cookie_str.split(';'):
            part = part.strip()
            if not part:
                continue
            if '=' in part:
                name, value = part.split('=', 1)
                cookies[name.strip()] = value.strip()
        return cookies

    def get_auth_headers(self) -> Dict[str, str]:
        """
        Booking 크롤러는 별도의 인증 헤더가 필요 없습니다. 
        대신 new.smartplace.naver.com에 맞는 Referer와 User-Agent를 사용합니다.
        """
        return {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Referer': f'https://partner.booking.naver.com/reports/stats?businessId={self.booking_key}'
        }

    def _fetch_booking_requests(self, date: str) -> int:
        """특정 날짜의 예약 신청 건수를 가져옵니다."""
        params = {
            'bucket': 'bookingCount_sum,day_trend',
            'startDate': date,
            'endDate': date,
            'metric': 'REQUESTED'
        }
        api_data = self.make_request('GET', self.booking_stat_url, params=params)
        if not api_data or 'result' not in api_data or not api_data['result']:
            logger.warning(f"⚠️ {date} 예약 신청 건수 데이터 없음: {api_data}")
            return 0
        
        # 'bookingCount_sum' 값을 찾아 반환
        return api_data['result'][0].get('bookingCount_sum', 0)

    def _fetch_page_visits(self, date: str) -> (int, List[Dict[str, Any]]):
        """특정 날짜의 페이지 유입 수와 채널별 통계를 가져옵니다."""
        params = {
            'bucket': 'areaCode,sessionCount_sum,day_trend',
            'startDate': date,
            'endDate': date,
            'metric': 'UV'
        }
        api_data = self.make_request('GET', self.booking_stat_url, params=params)
        if not api_data or 'result' not in api_data:
            logger.warning(f"⚠️ {date} 페이지 유입 데이터 없음: {api_data}")
            return 0, []

        total_visits = 0
        channel_stats = []
        for item in api_data.get('result', []):
            visits = item.get('sessionCount_sum', 0)
            total_visits += visits
            
            area_code = item.get('areaCode', 'Unknown')
            channel_name = self.channel_mapping.get(area_code, area_code)
            channel_stats.append({
                'channel_name': channel_name,
                'count': visits,
                'channel_code': area_code
            })
            
        return total_visits, channel_stats

    def fetch_booking_data_for_date(self, date: str) -> Optional[Dict[str, Any]]:
        """특정 날짜의 예약 데이터를 새로운 API 엔드포인트에서 수집하고 파싱합니다."""
        logger.info(f"📊 {date} 예약 데이터 수집 중 (신규 API)...")
        
        try:
            # 1. 예약 신청 건수 수집
            total_requested = self._fetch_booking_requests(date)

            # 2. 페이지 유입 수 및 채널별 통계 수집
            total_uv, channel_stats_list = self._fetch_page_visits(date)

            logger.info(f"✅ {date} 완료 - 페이지 유입: {total_uv}, 예약 신청: {total_requested}, 채널: {len(channel_stats_list)}개")

            # 기존 데이터 구조와 호환되도록 반환
            return {
                'page_visits': [{'count': total_uv}],
                'booking_requests': [{'count': total_requested}],
                'channel_stats': channel_stats_list,
            }

        except ApiCallError as e:
            logger.error(f"❌ {date} 예약 데이터 수집 중 API 오류: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ {date} 예약 데이터 수집 중 예상치 못한 오류: {e}", exc_info=True)
            return None
    
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
                if daily_data: # fetch_booking_data_for_date가 None을 반환할 수 있으므로 체크
                    all_data.append(daily_data)
                else:
                    logger.warning(f"⚠️ {date_str} 데이터 수집 실패 또는 데이터 없음")
                    all_data.append({
                        'page_visits': [],
                        'booking_requests': [],
                        'channel_stats': []
                    })
            except Exception as e:
                logger.error(f"❌ {date_str} 데이터 수집 실패: {e}")
                # 빈 데이터로 추가
                all_data.append({
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
        # 테스트용 클라이언트 정보 생성 (실제 사용 시에는 설정에서 가져와야 함)
        client_info = ClientInfo(
            booking_id="563688",
            booking_secret="your_booking_secret",
            booking_name="test_booking_place"
        )
        auth_config = AuthConfig(
            client_id="your_client_id",
            client_secret="your_client_secret",
            redirect_uri="your_redirect_uri"
        )
        crawler = NaverBookingStatCrawler(client_info=client_info, auth_config=auth_config)
        
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
