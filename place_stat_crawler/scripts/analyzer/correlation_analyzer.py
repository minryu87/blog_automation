import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

# 프로젝트 루트 경로를 sys.path에 추가
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from blog_automation.place_stat_crawler.scripts.util.logger import logger

class PerformanceAnalyzer:
    def __init__(self, client_name: str, start_year: int, start_month: int, end_year: int, end_month: int):
        self.client_name = client_name
        self.start_year = start_year
        self.start_month = start_month
        self.end_year = end_year
        self.end_month = end_month
        self.base_path = Path(__file__).resolve().parents[2]
        self.processed_data_path = self.base_path / 'data' / 'processed'
        self.analysis_results_path = self.base_path / 'analysis_results' / client_name
        self.analysis_results_path.mkdir(parents=True, exist_ok=True)
        
        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'AppleGothic'
        plt.rcParams['axes.unicode_minus'] = False


    def load_data(self) -> pd.DataFrame:
        """지정된 기간의 플레이스 PV 및 예약 데이터를 로드하고 병합합니다."""
        logger.info("데이터 로드를 시작합니다...")
        
        all_place_pv_df = []
        all_booking_df = []

        for year in range(self.start_year, self.end_year + 1):
            s_month = self.start_month if year == self.start_year else 1
            e_month = self.end_month if year == self.end_year else 12

            for month in range(s_month, e_month + 1):
                month_str = f"{month:02d}"
                
                # 플레이스 PV 데이터 로드
                pv_file = self.processed_data_path / f"{self.client_name}_{year}_{month_str}_integrated_statistics.csv"
                if pv_file.exists():
                    logger.info(f"로딩 (PV): {pv_file.name}")
                    all_place_pv_df.append(pd.read_csv(pv_file))
                else:
                    logger.warning(f"파일 없음 (PV): {pv_file.name}")

                # 예약 데이터 로드
                booking_file = self.processed_data_path / f"{self.client_name}_{year}_{month_str}_booking_integrated_statistics.csv"
                if booking_file.exists():
                    logger.info(f"로딩 (Booking): {booking_file.name}")
                    all_booking_df.append(pd.read_csv(booking_file))
                else:
                    logger.warning(f"파일 없음 (Booking): {booking_file.name}")

        if not all_place_pv_df or not all_booking_df:
            raise FileNotFoundError("분석에 필요한 데이터 파일이 충분하지 않습니다.")
            
        place_pv_df = pd.concat(all_place_pv_df, ignore_index=True)
        booking_df = pd.concat(all_booking_df, ignore_index=True)

        # 날짜 타입 변환
        place_pv_df['date'] = pd.to_datetime(place_pv_df['date'])
        booking_df['date'] = pd.to_datetime(booking_df['date'])

        # 일별 총 플레이스 조회수 집계
        daily_total_pv = place_pv_df.groupby('date')['total_count'].sum().reset_index()
        daily_total_pv.rename(columns={'total_count': 'total_place_pv'}, inplace=True)

        # 예약 데이터에서 필요한 컬럼만 선택
        booking_summary_df = booking_df.groupby('date')[['page_visits', 'booking_requests']].first().reset_index()
        
        # 데이터 병합
        merged_df = pd.merge(daily_total_pv, booking_summary_df, on='date', how='inner')
        logger.info("데이터 병합 완료. 최종 데이터프레임 Shape: %s", merged_df.shape)
        
        # 키워드 데이터 추가
        keyword_df = self.extract_keyword_data(place_pv_df)
        merged_df = pd.merge(merged_df, keyword_df, on='date', how='left').fillna(0)
        
        return merged_df

    def extract_keyword_data(self, place_pv_df: pd.DataFrame) -> pd.DataFrame:
        """플레이스 PV 데이터에서 브랜드/논브랜드 키워드 PV를 추출합니다."""
        keyword_df = place_pv_df[place_pv_df['data_type'] == 'keyword'].copy()
        
        # 브랜드 키워드 식별 (병원 이름의 일부가 포함된 경우)
        # client_name에서 영어 제외하고 한글만 추출하여 브랜드 키워드 리스트 생성
        brand_name = ''.join(filter(str.isalpha, self.client_name.replace("GOODMORNINGHANIGURO", "좋은아침한의원구로")))
        
        brand_keywords = [brand_name]
        # '좋은아침한의원' 과 같이 client_name의 일부를 포함하는 경우도 브랜드 키워드로 간주
        if "한의원" in brand_name:
            brand_keywords.append(brand_name.replace("한의원", ""))

        keyword_df['is_brand'] = keyword_df['keyword'].apply(lambda x: any(brand in str(x) for brand in brand_keywords))
        
        brand_pv = keyword_df[keyword_df['is_brand']].groupby('date')['total_count'].sum()
        non_brand_pv = keyword_df[~keyword_df['is_brand']].groupby('date')['total_count'].sum()
        
        keyword_summary = pd.DataFrame({
            'brand_keyword_pv': brand_pv,
            'non_brand_keyword_pv': non_brand_pv
        }).reset_index()

        return keyword_summary

    def analyze_correlation(self, df: pd.DataFrame):
        """주요 지표 간의 상관관계를 분석하고 히트맵으로 시각화합니다."""
        logger.info("상관관계 분석을 시작합니다...")
        
        metrics = ['total_place_pv', 'page_visits', 'booking_requests', 'brand_keyword_pv', 'non_brand_keyword_pv']
        corr_matrix = df[metrics].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('주요 지표 간 상관관계 히트맵')
        
        save_path = self.analysis_results_path / 'correlation_heatmap.png'
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"상관관계 히트맵 저장 완료: {save_path}")
        logger.info("\n상관계수 행렬:\n%s", corr_matrix)

    def analyze_conversion_rate(self, df: pd.DataFrame):
        """단계별 전환율을 시계열로 분석하고 그래프로 시각화합니다."""
        logger.info("전환율 분석을 시작합니다...")
        
        df['cvr_place_to_booking_page'] = (df['page_visits'] / df['total_place_pv']).replace([np.inf, -np.inf], 0) * 100
        df['cvr_booking_page_to_request'] = (df['booking_requests'] / df['page_visits']).replace([np.inf, -np.inf], 0) * 100
        
        df.set_index('date', inplace=True)
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        df['cvr_place_to_booking_page'].plot(title='전환율: 플레이스 조회 → 예약 페이지 유입 (%)')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        df['cvr_booking_page_to_request'].plot(title='전환율: 예약 페이지 유입 → 예약 신청 (%)', color='orange')
        plt.grid(True)
        
        plt.tight_layout()
        save_path = self.analysis_results_path / 'conversion_rate_timeseries.png'
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"전환율 시계열 그래프 저장 완료: {save_path}")
        logger.info("\n월별 평균 전환율:\n%s", df[['cvr_place_to_booking_page', 'cvr_booking_page_to_request']].resample('M').mean())


    def analyze_keyword_impact(self, df: pd.DataFrame):
        """브랜드/논브랜드 키워드 유입과 예약 신청의 관계를 분석합니다."""
        logger.info("키워드 영향력 분석을 시작합니다...")

        plt.figure(figsize=(15, 12))

        # 1. 브랜드/논브랜드 키워드 PV 시계열
        plt.subplot(3, 1, 1)
        df[['brand_keyword_pv', 'non_brand_keyword_pv']].plot(ax=plt.gca(), title='일별 브랜드/논브랜드 키워드 PV')
        plt.grid(True)

        # 2. 브랜드 키워드 PV와 예약 신청 수
        plt.subplot(3, 1, 2)
        sns.regplot(x='brand_keyword_pv', y='booking_requests', data=df, scatter_kws={'alpha':0.3})
        plt.title('브랜드 키워드 PV와 예약 신청 수의 관계')
        plt.grid(True)

        # 3. 논브랜드 키워드 PV와 예약 신청 수
        plt.subplot(3, 1, 3)
        sns.regplot(x='non_brand_keyword_pv', y='booking_requests', data=df, scatter_kws={'alpha':0.3}, color='g')
        plt.title('논브랜드 키워드 PV와 예약 신청 수의 관계')
        plt.grid(True)

        plt.tight_layout()
        save_path = self.analysis_results_path / 'keyword_impact_analysis.png'
        plt.savefig(save_path)
        plt.close()
        logger.info(f"키워드 영향력 분석 그래프 저장 완료: {save_path}")

    def run_analysis(self):
        """전체 분석 파이프라인을 실행합니다."""
        try:
            merged_df = self.load_data()
            self.analyze_correlation(merged_df.copy())
            self.analyze_conversion_rate(merged_df.copy())
            self.analyze_keyword_impact(merged_df.set_index('date').copy())
            logger.info("🎉 모든 분석이 성공적으로 완료되었습니다.")
        except FileNotFoundError as e:
            logger.error(e)
        except Exception as e:
            logger.error(f"분석 중 예기치 않은 오류 발생: {e}", exc_info=True)


if __name__ == '__main__':
    # 분석할 클라이언트와 기간 설정
    analyzer = PerformanceAnalyzer(
        client_name='GOODMORNINGHANIGURO',
        start_year=2024,
        start_month=7,
        end_year=2025,
        end_month=7
    )
    analyzer.run_analysis()
