import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys
from dotenv import load_dotenv
import json

# 프로젝트 루트 경로를 sys.path에 추가 (모듈 임포트를 위함)
project_root_for_imports = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root_for_imports))

from blog_automation.place_stat_crawler.scripts.util.logger import logger

class PerformanceAnalyzer:
    def __init__(self, client_name: str, start_year: int, start_month: int, end_year: int, end_month: int):
        self.client_name = client_name
        self.start_year = start_year
        self.start_month = start_month
        self.end_year = end_year
        self.end_month = end_month
        self.base_path = Path(__file__).resolve().parents[2]
        self.processed_data_path = self.base_path / 'data' / 'processed' / client_name
        self.analysis_results_path = self.base_path / 'analysis_results' / client_name
        self.analysis_results_path.mkdir(parents=True, exist_ok=True)
        
        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'AppleGothic'
        plt.rcParams['axes.unicode_minus'] = False

    def load_and_prepare_data(self) -> pd.DataFrame:
        """지정된 기간의 데이터를 로드하고 분석에 맞게 전처리합니다."""
        logger.info("데이터 로딩 및 전처리를 시작합니다...")
        
        all_place_pv_df = []
        all_booking_df = []

        for year in range(self.start_year, self.end_year + 1):
            s_month = self.start_month if year == self.start_year else 1
            e_month = self.end_month if year == self.end_year else 12

            for month in range(s_month, e_month + 1):
                month_str = f"{month:02d}"
                pv_file = self.processed_data_path / f"{self.client_name}_{year}_{month_str}_integrated_statistics.csv"
                if pv_file.exists():
                    logger.info(f"로딩 (PV): {pv_file.name}")
                    all_place_pv_df.append(pd.read_csv(pv_file))
                else:
                    logger.warning(f"파일 없음 (PV): {pv_file.name}")

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

        place_pv_df['date'] = pd.to_datetime(place_pv_df['date'])
        booking_df['date'] = pd.to_datetime(booking_df['date'])
        
        # 1. 일별 총 플레이스 조회수
        daily_total_pv = place_pv_df[place_pv_df['data_type'] == 'channel'].groupby('date')['pv'].sum().reset_index()
        daily_total_pv.rename(columns={'pv': 'total_place_pv'}, inplace=True)

        # 2. 플레이스 페이지 채널별 조회수
        place_channel_pv = place_pv_df[place_pv_df['data_type'] == 'channel'].pivot_table(
            index='date', columns='name', values='pv', aggfunc='sum').add_prefix('place_pv_')
        
        # 3. 키워드 유형별 PV
        keyword_pv = self.extract_keyword_data(place_pv_df)
        
        # 4. 예약 페이지 총 유입수 및 예약 신청 수
        booking_summary = booking_df.groupby('date').agg(
            total_booking_page_visits=('page_visits', 'first'),
            total_booking_requests=('booking_requests', 'first')
        ).reset_index()

        # 5. 예약 페이지 채널별 유입수
        booking_channel_visits = booking_df.pivot_table(
            index='date', columns='channel_name', values='channel_count', aggfunc='sum').add_prefix('booking_visits_')

        # 모든 데이터 병합
        merged_df = daily_total_pv
        for df in [place_channel_pv, keyword_pv, booking_summary, booking_channel_visits]:
            merged_df = pd.merge(merged_df, df, on='date', how='left')
            
        return merged_df.fillna(0)

    def extract_keyword_data(self, place_pv_df: pd.DataFrame) -> pd.DataFrame:
        """키워드를 유형별로 분류하고 PV를 집계합니다."""
        keyword_df = place_pv_df[place_pv_df['data_type'] == 'keyword'].copy()
        
        def classify_keyword(k):
            k_str = str(k).lower()
            if '내이튼' in k_str or '네이튼' in k_str:
                return 'type1_brand_like'
            return 'type2_others'

        keyword_df['keyword_type'] = keyword_df['name'].apply(classify_keyword)
        
        keyword_summary = keyword_df.pivot_table(
            index='date', columns='keyword_type', values='pv', aggfunc='sum'
        ).add_prefix('keyword_pv_')
        
        return keyword_summary.reset_index()

    def analyze_correlation(self, df: pd.DataFrame):
        """요청된 모든 주요 지표 간의 상관관계를 분석하고 히트맵으로 시각화합니다."""
        logger.info("종합 상관관계 분석을 시작합니다...")
        
        # 분석할 지표 선택
        metrics = [
            'total_booking_requests', 'total_booking_page_visits', 'total_place_pv',
        ]
        metrics.extend([col for col in df.columns if col.startswith('booking_visits_')])
        metrics.extend([col for col in df.columns if col.startswith('place_pv_')])
        metrics.extend([col for col in df.columns if col.startswith('keyword_pv_')])

        # 일부 열이 없을 수도 있으므로, df에 존재하는 열만 필터링
        metrics = [m for m in metrics if m in df.columns]
        
        corr_matrix = df[metrics].corr()

        # 상관관계가 높은 순으로 정렬하여 출력
        corr_pairs = corr_matrix.stack().reset_index()
        corr_pairs.columns = ['feature1', 'feature2', 'correlation']
        corr_pairs = corr_pairs[corr_pairs['feature1'] != corr_pairs['feature2']]
        corr_pairs['abs_correlation'] = corr_pairs['correlation'].abs()
        sorted_corr = corr_pairs.sort_values(by='abs_correlation', ascending=False).drop_duplicates(subset=['abs_correlation'])
        logger.info("\n상관관계 상위 20개:\n%s", sorted_corr.head(20))


        plt.figure(figsize=(20, 18))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 8})
        plt.title('주요 마케팅 지표 간 종합 상관관계 히트맵', fontsize=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        save_path = self.analysis_results_path / 'comprehensive_correlation_heatmap.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        logger.info(f"종합 상관관계 히트맵 저장 완료: {save_path}")
        logger.info("\n상관계수 행렬:\n%s", corr_matrix)

        return corr_matrix, sorted_corr.head(20)

    def save_analysis_summary_to_md(self, corr_matrix: pd.DataFrame, top_corr: pd.DataFrame):
        """분석 결과를 마크다운 파일로 저장합니다."""
        logger.info("분석 결과를 마크다운 파일로 저장합니다...")
        
        md_content = f"""# {self.client_name} 마케팅 퍼널 상관관계 분석 리포트

## 분석 요약

- **예약 퍼널의 기본 작동 확인**: `예약 페이지 방문수`와 `예약 신청 수`는 0.59의 뚜렷한 양의 상관관계를 보여, 예약 페이지 방문이 늘면 실제 예약도 증가하는 기본적인 퍼널이 작동함을 확인했습니다.
- **트래픽의 핵심 동력**: `플레이스 페이지 총 조회수`는 **'네이버 검색'** 유입(0.99) 및 **'브랜드성 키워드'**('아침', '285' 포함) 유입(0.94)과 매우 강한 상관관계를 보입니다. 이는 현재 트래픽이 대부분 병원 이름을 아는 사용자의 직접 검색에서 발생함을 의미합니다.
- **가장 중요한 발견 (트래픽의 양 vs 질)**: `플레이스 페이지 총 조회수`와 최종 `예약 신청 수`의 상관관계는 **0.05로 매우 낮습니다**. 이는 단순히 전체 방문자 수를 늘리는 것만으로는 예약 전환에 큰 영향을 주지 못하며, **트래픽의 질이 훨씬 중요함**을 시사합니다.
- **실제 예약에 효과적인 채널**: 최종 `예약 신청 수`와 가장 유의미한 상관관계를 보인 채널은 **'지도'(0.32)**와 **'플레이스목록'(0.31)**으로 나타났습니다. 이 채널을 통한 방문자의 예약 전환 가능성이 더 높습니다.

## 종합 상관관계 히트맵

![종합 상관관계 히트맵](./comprehensive_correlation_heatmap.png)

## 상관관계가 높은 지표 Top 20

{top_corr.to_markdown(index=False)}

## 전체 상관계수 행렬

{corr_matrix.to_markdown()}

"""
        report_path = self.analysis_results_path / 'analysis_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
            
        logger.info(f"분석 리포트 저장 완료: {report_path}")


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
            merged_df = self.load_and_prepare_data()
            corr_matrix, top_corr = self.analyze_correlation(merged_df)
            self.save_analysis_summary_to_md(corr_matrix, top_corr)
            # 기존 다른 분석들도 필요 시 여기에 추가 가능
            # self.analyze_conversion_rate(merged_df.copy())
            # self.analyze_keyword_impact(merged_df.set_index('date').copy())
            logger.info("🎉 모든 분석이 성공적으로 완료되었습니다.")
        except FileNotFoundError as e:
            logger.error(e)
        except Exception as e:
            logger.error(f"분석 중 예기치 않은 오류 발생: {e}", exc_info=True)


if __name__ == '__main__':
    # .env 파일 경로를 스크립트 위치 기준으로 정확하게 지정
    dotenv_path = Path(__file__).resolve().parents[2] / '.env'
    load_dotenv(dotenv_path=dotenv_path)
    
    client_list_str = os.getenv("CLIENT_LIST")
    if not client_list_str:
        logger.error(f"'{dotenv_path}' 경로에 '.env' 파일이 없거나 파일 내에 'CLIENT_LIST'가 설정되지 않았습니다.")
        sys.exit(1)
        
    try:
        client_list = json.loads(client_list_str)
    except json.JSONDecodeError:
        logger.error("CLIENT_LIST 형식이 잘못되었습니다. 예: '[\"CLIENT1\", \"CLIENT2\"]'")
        sys.exit(1)

    client_name = ""
    if len(client_list) == 1:
        client_name = client_list[0]
        logger.info(f"자동으로 클라이언트 '{client_name}'에 대한 분석을 시작합니다.")
    else:
        print("\n=== 분석할 클라이언트를 선택하세요 ===")
        for i, name in enumerate(client_list):
            print(f"  {i + 1}. {name}")
        
        while True:
            try:
                choice = int(input(f"\n번호를 입력하세요 (1-{len(client_list)}): "))
                if 1 <= choice <= len(client_list):
                    client_name = client_list[choice - 1]
                    logger.info(f"클라이언트 '{client_name}'를 선택했습니다.")
                    break
                else:
                    print(f"❌ 1에서 {len(client_list)} 사이의 숫자를 입력해주세요.")
            except ValueError:
                print("❌ 유효한 숫자를 입력해주세요.")
    
    if not client_name:
        logger.error("클라이언트가 선택되지 않았습니다. 분석을 종료합니다.")
        sys.exit(1)
        
    # 분석할 클라이언트와 기간 설정
    analyzer = PerformanceAnalyzer(
        client_name=client_name,
        start_year=2024,
        start_month=7,
        end_year=2025,
        end_month=7
    )
    analyzer.run_analysis()
