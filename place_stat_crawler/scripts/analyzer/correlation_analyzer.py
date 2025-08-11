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
        self.analysis_results_path = self.base_path / 'data' / 'analyzed' / client_name
        self.analysis_results_path.mkdir(parents=True, exist_ok=True)
        
        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'AppleGothic'
        plt.rcParams['axes.unicode_minus'] = False

        # 분석할 주요 지표 쌍 정의
        self.focused_pairs = [
            ("1. 예약 전환 분석", "total_booking_requests", "total_booking_page_visits"),
            ("1. 예약 전환 분석", "total_booking_requests", "booking_visits_지도"),
            ("1. 예약 전환 분석", "total_booking_requests", "booking_visits_플레이스목록"),
            ("1. 예약 전환 분석", "total_booking_page_visits", "total_place_pv"),
            ("2. 플레이스 트래픽 분석", "total_place_pv", "place_pv_네이버검색"),
            ("2. 플레이스 트래픽 분석", "total_place_pv", "place_pv_네이버지도"),
            ("2. 플레이스 트래픽 분석", "total_place_pv", "keyword_pv_type1_brand_like"),
            ("2. 플레이스 트래픽 분석", "total_place_pv", "keyword_pv_type2_others"),
        ]

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
            # 클라이언트별 브랜드 키워드를 여기서 동적으로 설정할 수 있습니다.
            # 지금은 하드코딩된 예시를 사용합니다.
            if self.client_name == 'GOODMORNINGHANIGURO':
                brand_keywords = ['아침', '285']
            elif self.client_name == 'NATENCLINIC':
                brand_keywords = ['내이튼', '네이튼']
            else:
                brand_keywords = []

            if any(kw in k_str for kw in brand_keywords):
                return 'type1_brand_like'
            return 'type2_others'

        keyword_df['keyword_type'] = keyword_df['name'].apply(classify_keyword)
        
        keyword_summary = keyword_df.pivot_table(
            index='date', columns='keyword_type', values='pv', aggfunc='sum'
        ).add_prefix('keyword_pv_')
        
        return keyword_summary.reset_index()
        
    def _calculate_correlations(self, df: pd.DataFrame, pairs: list) -> dict:
        """정의된 쌍에 대한 상관계수를 계산합니다."""
        results = {}
        for category, col1, col2 in pairs:
            pair_key = f"{col1} vs {col2}"
            # 데이터프레임에 두 컬럼이 모두 존재할 경우에만 계산
            if col1 in df.columns and col2 in df.columns and df[col1].nunique() > 1 and df[col2].nunique() > 1:
                correlation = df[col1].corr(df[col2])
                results[pair_key] = {'category': category, 'correlation': correlation}
            else:
                results[pair_key] = {'category': category, 'correlation': np.nan}
        return results

    def run_focused_analysis(self, df: pd.DataFrame):
        """요청된 형식의 리포트를 위한 분석을 수행합니다."""
        logger.info("포커스 분석(월별, 안정성)을 시작합니다...")
        
        # 전체 기간 상관관계
        overall_corr_results = self._calculate_correlations(df, self.focused_pairs)
        overall_corr_df = pd.DataFrame.from_dict(overall_corr_results, orient='index').reset_index().rename(columns={'index': 'pair'})
        
        # 월별 상관관계
        df_monthly = df.set_index('date').groupby(pd.Grouper(freq='M'))
        monthly_corr_list = []

        for month, group in df_monthly:
            if len(group) < 2: continue # 데이터가 2개 미만이면 상관관계 계산 불가
            month_str = month.strftime('%Y-%m')
            monthly_results = self._calculate_correlations(group, self.focused_pairs)
            for pair, values in monthly_results.items():
                monthly_corr_list.append({
                    'month': month_str,
                    'category': values['category'],
                    'pair': pair,
                    'correlation': values['correlation']
                })
        
        monthly_corr_df = pd.DataFrame(monthly_corr_list)

        # 안정성 분석 (표준편차)
        stability_df = monthly_corr_df.groupby('pair')['correlation'].std().reset_index()
        stability_df.rename(columns={'correlation': 'std_dev'}, inplace=True)
        stability_df = stability_df.sort_values(by='std_dev').dropna()

        self.plot_monthly_correlation_trends(monthly_corr_df)

        return overall_corr_df, stability_df, monthly_corr_df

    def plot_monthly_correlation_trends(self, monthly_corr_df: pd.DataFrame):
        """월별 상관관계 트렌드를 시각화합니다."""
        logger.info("월별 상관관계 트렌드 그래프를 생성합니다...")
        
        pivot_df = monthly_corr_df.pivot(index='month', columns='pair', values='correlation')
        
        plt.figure(figsize=(20, 12))
        for column in pivot_df.columns:
            plt.plot(pivot_df.index, pivot_df[column], marker='o', linestyle='-', label=column)
            
        plt.title('월별 주요 지표 상관관계 트렌드', fontsize=20)
        plt.xlabel('월')
        plt.ylabel('상관계수')
        plt.xticks(rotation=45)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        
        save_path = self.analysis_results_path / 'monthly_correlation_trends.png'
        plt.savefig(save_path)
        plt.close()
        logger.info(f"월별 트렌드 그래프 저장 완료: {save_path}")

    def save_analysis_summary_to_md(self, overall_corr, stability_df, monthly_corr_df):
        """요청된 형식에 맞춰 분석 결과를 마크다운 파일로 저장합니다."""
        logger.info("새로운 형식의 분석 리포트를 마크다운 파일로 저장합니다...")

        md_content = f"""# {self.client_name} 마케팅 퍼널 상관관계 분석 리포트

## 1. 전체 기간 주요 퍼널 단계별 상관관계

전체 분석 기간동안의 평균적인 관계입니다.

{overall_corr.to_markdown(index=False)}

## 2. 상관관계 안정성 분석 (변동성 낮은 순)

월별 상관관계의 표준편차입니다. 값이 낮을수록 기간에 상관없이 꾸준하고 안정적인 관계임을 의미합니다.

{stability_df.to_markdown(index=False)}

## 3. 월별 상관관계 트렌드

주요 관계들이 월별로 어떻게 변하는지 보여줍니다. 특정 마케팅 활동이나 이벤트와의 연관성을 파악하는 데 유용합니다.

![월별 상관관계 트렌드](monthly_correlation_trends.png)

## 4. 월별 상관관계 상세 데이터

{monthly_corr_df.sort_values(by=['month', 'category']).to_markdown(index=False)}
"""
        report_path = self.analysis_results_path / f'{self.client_name}_focused_analysis_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
            
        logger.info(f"분석 리포트 저장 완료: {report_path}")

    def run_analysis(self):
        """전체 분석 파이프라인을 실행합니다."""
        try:
            merged_df = self.load_and_prepare_data()
            overall_corr, stability_df, monthly_corr_df = self.run_focused_analysis(merged_df)
            self.save_analysis_summary_to_md(overall_corr, stability_df, monthly_corr_df)

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
        start_month=9,
        end_year=2025,
        end_month=7
    )
    analyzer.run_analysis()