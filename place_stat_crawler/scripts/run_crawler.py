import sys
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 경로를 sys.path에 추가하여 모듈 임포트가 가능하도록 함
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from scripts.util.config import get_config_manager
from scripts.crawler.naver_place_pv_monthly_crawler import MonthlyStatisticsCrawler
from scripts.crawler.naver_booking_monthly_crawler import MonthlyBookingCrawler
from scripts.util.logger import logger
from scripts.crawler.naver_place_pv_crawler_base import ApiCallError

def get_month_range(start_date_str: str, end_date_str: str) -> list[tuple[int, int]]:
    """시작 연월과 종료 연월 사이의 모든 (연, 월)을 리스트로 반환합니다."""
    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m')
        end_date = datetime.strptime(end_date_str, '%Y-%m')
    except ValueError:
        return []

    months = []
    current_date = start_date
    while current_date <= end_date:
        months.append((current_date.year, current_date.month))
        # 다음 달로 이동
        # 월이 12월을 넘어가면 다음 해 1월로 처리
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)
    return months

def main():
    """데이터 수집 프로세스를 총괄하는 메인 함수"""
    logger.info("=" * 60)
    logger.info("📊 네이버 플레이스 & 예약 월별 데이터 수집 스크립트")
    logger.info("=" * 60)

    # 1. 클라이언트 선택 (단 한번)
    config_manager = get_config_manager()
    client_info = config_manager.get_selected_client_config() # 대화형 프롬프트 호출
    if not client_info:
        logger.error("❌ 클라이언트가 선택되지 않았습니다. 스크립트를 종료합니다.")
        return
    logger.info(f"✅ '{client_info.name}' 클라이언트에 대한 작업을 시작합니다.")
    
    auth_config = config_manager.get_auth_config()

    # 2. 수집 기간 입력
    while True:
        start_str = input("▶️ 수집 시작 연월을 입력하세요 (예: 2024-09): ").strip()
        end_str = input("▶️ 수집 종료 연월을 입력하세요 (예: 2025-07): ").strip()
        
        months_to_collect = get_month_range(start_str, end_str)
        
        if not months_to_collect:
            logger.warning("❌ 잘못된 날짜 형식 또는 기간입니다. 'YYYY-MM' 형식으로 다시 입력해주세요.")
            continue
        
        if datetime.strptime(start_str, '%Y-%m') > datetime.strptime(end_str, '%Y-%m'):
            logger.warning("❌ 시작 연월은 종료 연월보다 나중일 수 없습니다.")
            continue
            
        break

    logger.info(f"총 {len(months_to_collect)}개월의 데이터를 수집합니다: [{start_str} ~ {end_str}]")

    # 3. 크롤러 인스턴스 생성 (선택된 정보를 인자로 전달)
    pv_crawler = MonthlyStatisticsCrawler(client_info, auth_config)
    booking_crawler = MonthlyBookingCrawler(client_info, auth_config)

    # 4. 월별 루프 실행
    for year, month in months_to_collect:
        logger.info("\n" + "=" * 25 + f" {year}년 {month:02d}월 데이터 수집 시작 " + "=" * 25)
        
        try:
            logger.info("--- [1/2] Place PV 통계 수집 중... ---")
            pv_crawler.run_monthly_analysis(year, month, client_info.name)

            logger.info("--- [2/2] Booking 통계 수집 중... ---")
            booking_crawler.run_monthly_analysis(year, month, client_info.name)

            logger.info(f"✅ {year}년 {month:02d}월 데이터 수집 완료")

        except ApiCallError:
            # ApiCallError가 발생하면 이미 하위 모듈에서 상세 로그를 출력했으므로
            # 여기서는 루프를 중단하고 스크립트를 종료하기만 함
            logger.error("API 인증 오류로 인해 전체 데이터 수집 프로세스를 중단합니다.")
            break
        except Exception as e:
            logger.critical(f"💥 {year}년 {month:02d}월 처리 중 심각한 오류 발생. 다음 달로 넘어갑니다.", exc_info=True)
            continue

    logger.info("\n" + "=" * 60)
    logger.info("🎉 모든 요청된 기간의 데이터 수집이 완료되었습니다.")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()