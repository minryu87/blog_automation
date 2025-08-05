import pandas as pd
import numpy as np
import os
import glob
import warnings
import json
import re
from dotenv import load_dotenv, find_dotenv
from agno.agent import Agent
from agno.models.google import Gemini

warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

# --- Configuration ---
RAW_DATA_DIR = 'blog_automation/place_analysis/data/raw_data/동탄치과'
RESULT_PATH = 'blog_automation/place_analysis/analysis_result'
REPORT_OUTPUT_PATH = os.path.join(RESULT_PATH, 'llm_enhanced_analysis_v8_report.md')
TARGET_CLINIC = '내이튼치과의원'

class RankAnalyzerV8:
    def __init__(self, data_dir, result_path):
        self.data_dir = data_dir
        self.result_path = result_path
        self.agent = self._initialize_llm_agent()
        self.time_series_df = self._load_and_prepare_time_series_data()
        os.makedirs(self.result_path, exist_ok=True)

    def _initialize_llm_agent(self):
        print("LLM 에이전트 초기화 중...")
        load_dotenv(find_dotenv())
        try:
            llm = Gemini(id=os.getenv("GEMINI_MODEL", "gemini/gemini-1.5-pro-latest"), api_key=os.getenv("GEMINI_API_KEY"))
            return Agent(model=llm)
        except Exception as e:
            print(f"LLM 에이전트 초기화 실패: {e}.")
            return None

    def _load_and_prepare_time_series_data(self):
        print("시계열 데이터 로딩 및 통합 시작...")
        csv_files = glob.glob(os.path.join(self.data_dir, 'vs*.csv'))
        if not csv_files:
            print(f"오류: CSV 파일을 찾을 수 없습니다.")
            return pd.DataFrame()

        all_data = []
        # First, establish T=0 (current) data from the most recent file available.
        # Let's use vs1.csv as the baseline for "current" values.
        baseline_file = os.path.join(self.data_dir, 'vs1.csv')
        if not os.path.exists(baseline_file):
            print("오류: 기준 파일(vs1.csv)이 없습니다.")
            return pd.DataFrame()

        df_t0 = pd.read_csv(baseline_file)
        # Clean company name
        df_t0['company_name'] = df_t0['company_name_category'].apply(lambda x: x.split(',')[0] if isinstance(x, str) else None)
        df_t0.dropna(subset=['company_name'], inplace=True)
        
        # Add T=0 data
        t0_data = df_t0[['company_name', 'rank', 'visitor_reviews', 'blog_reviews']].copy()
        t0_data['days_ago'] = 0
        all_data.append(t0_data)

        for f in csv_files:
            try:
                days_ago = int(re.search(r'vs(\d+)\.csv', f).group(1))
                if days_ago == 0: continue

                df = pd.read_csv(f)
                df['company_name'] = df['company_name_category'].apply(lambda x: x.split(',')[0] if isinstance(x, str) else None)
                df.dropna(subset=['company_name'], inplace=True)

                # Calculate past data based on the change from the T=0 baseline
                temp_df = df_t0[['company_name', 'visitor_reviews', 'blog_reviews']].copy()
                
                # Clean up change columns
                for col in ['visitor_reviews_change', 'blog_reviews_change']:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
                merged = pd.merge(temp_df, df[['company_name', 'rank', 'visitor_reviews_change', 'blog_reviews_change']], on='company_name', how='inner')
                
                merged['visitor_reviews'] = merged['visitor_reviews'] - merged['visitor_reviews_change']
                merged['blog_reviews'] = merged['blog_reviews'] - merged['blog_reviews_change']
                
                historical_data = merged[['company_name', 'rank', 'visitor_reviews', 'blog_reviews']].copy()
                historical_data['days_ago'] = days_ago
                all_data.append(historical_data)

            except Exception as e:
                print(f"'{f}' 파일 처리 중 오류 발생: {e}")
        
        if not all_data: return pd.DataFrame()

        full_df = pd.concat(all_data, ignore_index=True)
        full_df = full_df.sort_values(by=['company_name', 'days_ago']).reset_index(drop=True)
        print(f"시계열 데이터 통합 완료. 총 {len(full_df)}개 레코드 생성.")
        return full_df

    def analyze_review_acceleration(self):
        print("1. 리뷰 증가 '가속도' 분석 중...")
        df_unique = self.time_series_df.drop_duplicates(subset=['company_name', 'days_ago'], keep='last')
        df = df_unique.pivot(index='company_name', columns='days_ago', values=['visitor_reviews', 'blog_reviews'])
        df.columns = [f"{val}_{days}" for val, days in df.columns]

        # Calculate increases for two 30-day periods
        for review_type in ['visitor_reviews', 'blog_reviews']:
            # Ensure columns exist before calculation
            required_cols = [f"{review_type}_{d}" for d in [0, 30, 60]]
            if not all(col in df.columns for col in required_cols):
                print(f"경고: 가속도 분석에 필요한 데이터(0, 30, 60일 전)가 부족합니다. ({review_type})")
                continue

            df[f'increase_last_30d_{review_type}'] = df[f'{review_type}_0'] - df[f'{review_type}_30']
            df[f'increase_first_30d_{review_type}'] = df[f'{review_type}_30'] - df[f'{review_type}_60']
            df[f'acceleration_{review_type}'] = df[f'increase_last_30d_{review_type}'] - df[f'increase_first_30d_{review_type}']

        # Get top 5 accelerators for each review type
        top_blog_accel = df.sort_values(by='acceleration_blog_reviews', ascending=False).head(5)[['increase_first_30d_blog_reviews', 'increase_last_30d_blog_reviews', 'acceleration_blog_reviews']]
        top_visitor_accel = df.sort_values(by='acceleration_visitor_reviews', ascending=False).head(5)[['increase_first_30d_visitor_reviews', 'increase_last_30d_visitor_reviews', 'acceleration_visitor_reviews']]
        
        return top_blog_accel.reset_index(), top_visitor_accel.reset_index()

    def analyze_competitor_roles(self):
        print("2. '상위권 수문장' vs '급상승 도전자' 식별 중...")
        df_0 = self.time_series_df[self.time_series_df['days_ago'] == 0]
        df_60 = self.time_series_df[self.time_series_df['days_ago'] == 60]

        # Gatekeepers: Top 10 at T-60 and T-0
        top_60 = df_60[df_60['rank'] <= 10]['company_name']
        top_0 = df_0[df_0['rank'] <= 10]['company_name']
        gatekeepers = pd.merge(top_60, top_0, on='company_name')

        # Challengers: Outside Top 20 at T-60, inside Top 15 at T-0
        outside_20_at_60 = df_60[df_60['rank'] > 20]['company_name']
        inside_15_at_0 = df_0[df_0['rank'] <= 15][['company_name', 'rank']]
        challengers = pd.merge(outside_20_at_60, inside_15_at_0, on='company_name')
        
        # Add original rank for challengers
        challengers = pd.merge(challengers, df_60[['company_name', 'rank']], on='company_name', suffixes=('_current', '_60_days_ago'))

        return gatekeepers, challengers

    def analyze_marketing_timelag(self):
        print("3. 마케팅 활동 '시기'와 '효과 발생 시점' 추정 중...")
        df_unique = self.time_series_df.drop_duplicates(subset=['company_name', 'days_ago'], keep='last')
        df_pivoted = df_unique.pivot(index='company_name', columns='days_ago', values=['blog_reviews', 'rank'])
        df_pivoted.columns = [f"{val}_{days}" for val, days in df_pivoted.columns]
        
        # Find who had the biggest blog increase in the first 30 days (T-60 to T-30)
        df_pivoted['blog_increase_first_30d'] = df_pivoted['blog_reviews_30'] - df_pivoted['blog_reviews_60']
        
        # Find their rank change in the last 30 days (T-30 to T-0)
        df_pivoted['rank_change_last_30d'] = df_pivoted['rank_30'] - df_pivoted['rank_0']
        
        # Filter for those with significant blog increase and positive rank change
        top_investors = df_pivoted[df_pivoted['blog_increase_first_30d'] > 0].sort_values(by='blog_increase_first_30d', ascending=False).head(5)
        
        result = top_investors[['blog_increase_first_30d', 'rank_change_last_30d', 'rank_60', 'rank_30', 'rank_0']].copy()
        result.rename(columns={'rank_0': '현재 순위', 'rank_30': '30일 전 순위', 'rank_60': '60일 전 순위'}, inplace=True)
        return result.reset_index()
        
    def analyze_review_efficiency(self):
        print("4. '리뷰 효율성' 분석 (마케팅 ROI) 중...")
        df_0 = self.time_series_df[self.time_series_df['days_ago'] == 0]
        df_60 = self.time_series_df[self.time_series_df['days_ago'] == 60]

        merged = pd.merge(df_0, df_60, on='company_name', suffixes=('_0', '_60'))
        
        merged['rank_improvement'] = merged['rank_60'] - merged['rank_0']
        merged['total_review_change'] = (merged['visitor_reviews_0'] - merged['visitor_reviews_60']) + \
                                        (merged['blog_reviews_0'] - merged['blog_reviews_60'])
        
        # Analyze only those who improved their rank and reviews
        improved_clinics = merged[(merged['rank_improvement'] > 0) & (merged['total_review_change'] > 0)].copy()
        
        if improved_clinics.empty:
            return "순위 상승 업체 데이터 없음"

        improved_clinics['reviews_per_rank'] = improved_clinics['total_review_change'] / improved_clinics['rank_improvement']
        
        market_avg_efficiency = improved_clinics['reviews_per_rank'].mean()
        
        return f"순위 1칸 상승에 필요한 평균 리뷰 수: {market_avg_efficiency:.1f}개"

    def generate_report(self, all_data):
        print("\nLLM 기반 최종 보고서 생성 (v8)...")
        if not self.agent: return

        # Convert dataframes to markdown for the prompt
        for key, value in all_data.items():
            if isinstance(value, pd.DataFrame):
                all_data[key] = value.to_markdown(index=False)

        prompt = f"""
        ## 시스템 지침: 당신의 역할과 분석의 맥락 (v8) ##
        당신은 '동탄치과' 키워드에 대한 60일간의 시계열 데이터를 분석하여, 시장의 동적인 '흐름'과 '역동성'을 파악하고, 이를 바탕으로 실행 가능한 경쟁 전략을 제안하는 최고의 데이터 분석 컨설턴트입니다. 이전 v7 분석이 시장의 '정적인 스냅샷'이었다면, 이번 v8 분석은 시장의 '동적인 영상'을 분석하는 것입니다.

        - **v8 분석의 핵심:**
            1.  **가속도 분석:** 경쟁사의 마케팅 투자 의지를 파악합니다.
            2.  **경쟁 구도 입체화:** 시장의 '수문장'과 '도전자'를 식별합니다.
            3.  **마케팅 시차 분석:** 마케팅 활동과 성과 발생 간의 시간 차이를 추정합니다.
            4.  **리뷰 효율성 분석:** 마케팅 ROI를 정량화하여 구체적인 목표 설정을 지원합니다.

        ---

        ## 사용자 요청: '동탄치과' 시장 동적 분석 보고서 작성 ##
        이제 아래 주어진 **4가지 핵심 인사이트 분석 결과**를 종합적으로 해석하고, 명확하고 논리적인 구조에 따라 **하나의 완성된 보고서**를 작성해주세요. 특히, 이 모든 분석이 최종적으로 '{TARGET_CLINIC}'에게 어떤 의미가 있는지, 어떤 전략을 취해야 하는지에 대한 제언으로 귀결되어야 합니다.

        ### 종합 분석 데이터 ###

        #### 1. 리뷰 증가 '가속도' 분석: 최근 마케팅을 강화한 경쟁사는? ####
        *최근 30일(T-30~T-0)과 이전 30일(T-60~T-30)의 리뷰 증가량을 비교하여, 마케팅 투자를 확대하고 있는 경쟁사를 식별합니다.*
        
        **블로그 리뷰 가속도 TOP 5**
        {all_data['blog_accel']}

        **방문자 리뷰 가속도 TOP 5**
        {all_data['visitor_accel']}

        #### 2. 경쟁 구도 분석: '수문장' vs '도전자' ####
        *지난 60일간의 순위 변동을 추적하여 시장의 구조를 입체적으로 분석합니다.*

        **굳건한 상위권 (수문장 그룹)**
        {all_data['gatekeepers']}

        **최근 순위 급상승 (도전자 그룹)**
        {all_data['challengers']}

        #### 3. 마케팅 시차 분석: 블로그 마케팅, 효과는 언제부터? ####
        *초기 30일간 블로그 리뷰에 가장 많이 투자한 병원들이, 이후 30일간 어떤 순위 변화를 겪었는지 추적합니다.*
        {all_data['timelag']}
        
        #### 4. 리뷰 효율성 분석: 순위 1칸 올리려면 리뷰 몇 개가 필요할까? ####
        *지난 60일간 순위가 상승한 모든 병원의 데이터를 분석하여, 순위 1칸을 올리는 데 필요한 평균 리뷰 수를 계산합니다.*
        
        **시장 평균 효율성:** {all_data['efficiency']}
        """

        response = self.agent.run(prompt)
        md_content = f"# [V8] '동탄치과' 시장 동적 분석 및 경쟁 전략 보고서\n\n{response.content if response and hasattr(response, 'content') else '보고서 생성에 실패했습니다.'}"

        with open(REPORT_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"\n분석 보고서(v8)가 다음 경로에 저장되었습니다: {REPORT_OUTPUT_PATH}")

def main():
    analyzer = RankAnalyzerV8(data_dir=RAW_DATA_DIR, result_path=RESULT_PATH)
    if analyzer.time_series_df.empty:
        print("분석을 위한 데이터가 부족합니다.")
        return

    blog_accel, visitor_accel = analyzer.analyze_review_acceleration()
    gatekeepers, challengers = analyzer.analyze_competitor_roles()
    timelag_analysis = analyzer.analyze_marketing_timelag()
    review_efficiency = analyzer.analyze_review_efficiency()
    
    all_data_for_report = {
        "blog_accel": blog_accel,
        "visitor_accel": visitor_accel,
        "gatekeepers": gatekeepers,
        "challengers": challengers,
        "timelag": timelag_analysis,
        "efficiency": review_efficiency,
    }

    analyzer.generate_report(all_data_for_report)

if __name__ == "__main__":
    main()
