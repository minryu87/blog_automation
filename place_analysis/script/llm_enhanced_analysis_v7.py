import pandas as pd
import numpy as np
import os
import glob
import warnings
import json
from dotenv import load_dotenv, find_dotenv
from agno.agent import Agent
from agno.models.google import Gemini
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

# --- Configuration ---
PROCESSED_DATA_DIR = 'blog_automation/place_analysis/data/processed_data/지역별_검색순위'
RESULT_PATH = 'blog_automation/place_analysis/analysis_result'
REPORT_OUTPUT_PATH = os.path.join(RESULT_PATH, 'llm_enhanced_analysis_v7_report.md')
TARGET_KEYWORD = '동탄치과' # The single keyword we want to analyze
TARGET_CLINIC = '내이튼치과의원'

class RankAnalyzerV7:
    def __init__(self, data_dir, result_path, target_keyword):
        self.data_dir = data_dir
        self.result_path = result_path
        self.target_keyword = target_keyword
        # Load data specifically for the target keyword
        self.keyword_df = self._load_and_prepare_data()
        self.agent = self._initialize_llm_agent()
        os.makedirs(self.result_path, exist_ok=True)

    def _load_and_prepare_data(self):
        print(f"'{self.target_keyword}' 키워드에 대한 데이터 로딩 시작...")
        keyword_csv_path = os.path.join(self.data_dir, f'{self.target_keyword}.csv')
        
        if not os.path.exists(keyword_csv_path):
            print(f"오류: '{keyword_csv_path}' 파일을 찾을 수 없습니다.")
            return pd.DataFrame()
            
        try:
            df = pd.read_csv(keyword_csv_path)
            df['search_keyword'] = self.target_keyword
        except Exception as e:
            print(f"'{keyword_csv_path}' 파일 로드 중 오류 발생: {e}")
            return pd.DataFrame()
            
        # Basic data cleaning
        for col in ['place_rank', 'place_visit_cnt', 'place_blog_cnt', 'place_rank_compare', 'place_visit_cnt_compare', 'place_blog_cnt_compare']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows where essential data is missing
        df.dropna(subset=['place_name', 'place_rank', 'place_visit_cnt', 'place_blog_cnt'], inplace=True)
        print(f"'{self.target_keyword}' 키워드 데이터 로딩 및 전처리 완료. 총 {len(df)}개 업체 데이터.")
        return df

    def _initialize_llm_agent(self):
        print("LLM 에이전트 초기화 중...")
        load_dotenv(find_dotenv())
        try:
            llm = Gemini(id=os.getenv("GEMINI_MODEL", "gemini/gemini-1.5-pro-latest"), api_key=os.getenv("GEMINI_API_KEY"))
            return Agent(model=llm)
        except Exception as e:
            print(f"LLM 에이전트 초기화 실패: {e}.")
            return None

    def _create_dynamic_tiers_kmeans(self, df, column_name, n_clusters=5):
        print(f"'{self.target_keyword}' 시장 내 '{column_name}'에 대한 K-평균 클러스터링 기반 동적 등급 생성 중...")
        data = df[[column_name]].dropna()
        
        # Adjust n_clusters if there are fewer data points than clusters
        if len(data) < n_clusters:
            print(f"데이터 포인트({len(data)}개)가 클러스터 수({n_clusters}개)보다 적어 클러스터 수를 {len(data)}개로 조정합니다.")
            n_clusters = len(data)

        if n_clusters == 0:
            print("분석할 데이터가 없습니다.")
            return df

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(scaled_data)
        
        df.loc[data.index, f'{column_name}_cluster'] = clusters
        
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        cluster_order = np.argsort(cluster_centers.flatten())[::-1]
        
        tier_names = [f'Tier {i+1}' for i in range(n_clusters)]
        tier_map = {cluster_order[i]: tier_names[i] for i in range(n_clusters)}
        
        tier_col_name = f'{column_name}_tier'
        df[tier_col_name] = df[f'{column_name}_cluster'].map(tier_map)
        
        return df.drop(columns=[f'{column_name}_cluster'])

    def analyze_tier_performance(self, df, tier_col, value_cols):
        agg_funcs = ['size', 'mean', 'std']
        analysis = df.groupby(tier_col)[value_cols].agg(agg_funcs).reset_index()
        
        analysis.columns = ['_'.join(col).strip() for col in analysis.columns.values]
        analysis.rename(columns={
            f'{tier_col}_': 'tier',
            f'{value_cols[0]}_size': '업체 수'
        }, inplace=True)
        
        for col in analysis.columns[2:]:
             analysis[col] = analysis[col].fillna(0).round(1)
        return analysis.sort_values(by='tier', ascending=True)

    def simulate_clinic_performance(self, df, clinic_name, visit_tier_data, blog_tier_data):
        print(f"'{clinic_name}' 맞춤형 시뮬레이션 중...")
        target_clinic = df[df['place_name'].str.contains(clinic_name, na=False)]
        if target_clinic.empty: 
            print(f"'{clinic_name}'을(를) '{self.target_keyword}' 검색 결과에서 찾을 수 없습니다.")
            return None

        clinic_info = target_clinic.iloc[0]
        
        current_visit_tier = clinic_info['place_visit_cnt_tier']
        current_blog_tier = clinic_info['place_blog_cnt_tier']

        def get_levelup_target(tier_data, current_tier, value_col):
            if current_tier == "Tier 1": return "이미 최상위 등급"
            
            current_tier_num = int(current_tier.split(' ')[1])
            next_tier_name = f'Tier {current_tier_num - 1}'
            
            next_tier_info = tier_data[tier_data['tier'] == next_tier_name]
            if next_tier_info.empty: return "데이터 없음"
            
            return int(round(next_tier_info[f'{value_col}_mean'].iloc[0]))

        sim_data = {
            'clinic_name': clinic_info['place_name'],
            'current_rank': clinic_info['place_rank'],
            'current_visit_reviews': clinic_info['place_visit_cnt'],
            'current_blog_reviews': clinic_info['place_blog_cnt'],
            'current_visit_tier': current_visit_tier,
            'current_blog_tier': current_blog_tier,
            'visit_levelup_target': get_levelup_target(visit_tier_data, current_visit_tier, 'place_visit_cnt'),
            'blog_levelup_target': get_levelup_target(blog_tier_data, current_blog_tier, 'place_blog_cnt'),
        }
        return sim_data

    def generate_report(self, all_data):
        print("\nLLM 기반 최종 보고서 생성 (v7)...")
        if not self.agent: return

        if all_data.get('simulation'):
            sim_data = all_data['simulation']
            for key, value in sim_data.items():
                if isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
                    sim_data[key] = int(value)
                elif isinstance(value, (np.float64, np.float32)):
                    sim_data[key] = float(value)

        base_prompt = f"""
        ## 시스템 지침: 당신의 역할과 분석의 맥락 (v7) ##
        당신은 특정 지역 키워드의 '네이버 플레이스' 순위 데이터를 분석하여, 해당 지역 병원 마케팅 담당자에게 실행 가능한 초정밀 로컬 마케팅 전략을 제안하는 최고의 데이터 분석 컨설턴트입니다. 당신의 답변은 반드시 아래의 맥락 정보를 완벽하게 숙지하고, 전문성과 신뢰성을 담아 한국어 마크다운 형식으로 작성되어야 합니다.
        
        - **v7 분석의 핵심 (가장 중요한 차별점):**
            1.  **'단일 키워드' 초정밀 분석:** 이전 분석과 달리, 전국 데이터가 아닌 오직 **'{self.target_keyword}'** 라는 단 하나의 검색어 결과에만 100% 집중합니다. 이 보고서의 모든 내용은 '동탄치과' 시장 내에서의 경쟁 구도만을 다룹니다.
            2.  **모든 경쟁자 포함:** '{self.target_keyword}' 검색 시 노출되는 모든 병원을 분석에 포함시켜, 실제 로컬 경쟁 환경을 있는 그대로 반영했습니다. (이상치 제거 없음)
            3.  **K-평균 클러스터링:** '{self.target_keyword}' 시장 내에서, 컴퓨터가 데이터의 실제 분포를 보고 '자연스러운 그룹(군집)'을 찾아내어 통계적으로 유의미한 **'동적 등급(Tier)'**을 생성했습니다.
            4.  **Tier 1 = 최상위:** 모든 데이터에서 'Tier 1'이 가장 높은 등급(리뷰 수가 가장 많은 그룹)을 의미합니다.
        
        - **분석의 논리 흐름:**
            1.  **'로컬 시장 현실 진단'**: '{self.target_keyword}' 시장에서, 방문자/블로그 리뷰 각각의 기준으로 병원들이 어떻게 통계적 그룹(Tier)으로 나뉘는지 '업체 수'와 함께 보여줍니다.
            2.  **'로컬 시장 성장 속도 제시'**: 각 Tier에 속한 병원들의 '10일간 평균 리뷰 증가량'을 보여주며, 이 시장의 경쟁 속도를 인지시킨다.
            3.  **'맞춤형 전략 제안'**: 타겟 병원의 현재 Tier를 진단하고, '{self.target_keyword}' 시장의 다음 상위 Tier로 올라서기 위해 필요한 '리뷰 목표'를 구체적으로 계산하여 제시한다.
        
        ---
        
        ## 사용자 요청: '{self.target_keyword}' 시장 분석 보고서 작성 ##
        이제 아래 주어진 **모든 분석 데이터**를 종합적으로 해석하고, 위에서 학습한 모든 맥락을 총동원하여 **'{self.target_keyword}' 시장에 대한 완성된 하나의 보고서**를 작성해주세요.
        """
        
        full_prompt = base_prompt + f"\n\n### '{self.target_keyword}' 시장 종합 분석 데이터 ###\n\n" + \
                      f"#### 1-1. 방문자 리뷰 동적 등급(Tier)별 현황 ####\n{all_data['visit_tier'].to_markdown(index=False)}\n\n" + \
                      f"#### 1-2. 블로그 리뷰 동적 등급(Tier)별 현황 ####\n{all_data['blog_tier'].to_markdown(index=False)}\n\n" + \
                      f"#### 2-1. 방문자 리뷰 등급별 '10일' 성장 속도 ####\n{all_data['visit_change'].to_markdown(index=False)}\n\n" + \
                      f"#### 2-2. 블로그 리뷰 등급별 '10일' 성장 속도 ####\n{all_data['blog_change'].to_markdown(index=False)}\n\n" + \
                      f"#### 3. '{all_data.get('simulation', {}).get('clinic_name', 'N/A')}' 맞춤형 전략 시뮬레이션 ####\n" + \
                      f"```json\n{json.dumps(all_data.get('simulation'), indent=2, ensure_ascii=False)}\n```"

        response = self.agent.run(full_prompt)
        md_content = f"# [V7] '{self.target_keyword}' 시장 정밀 분석 보고서\n\n{response.content if response and hasattr(response, 'content') else '보고서 생성에 실패했습니다.'}"

        with open(REPORT_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"\n분석 보고서(v7)가 다음 경로에 저장되었습니다: {REPORT_OUTPUT_PATH}")

def main():
    analyzer = RankAnalyzerV7(data_dir=PROCESSED_DATA_DIR, result_path=RESULT_PATH, target_keyword=TARGET_KEYWORD)
    if analyzer.keyword_df.empty: return

    # No outlier removal in v7, we analyze all competitors in the local market.
    
    # Create dynamic tiers using K-Means on the local market data
    df_with_tiers = analyzer._create_dynamic_tiers_kmeans(analyzer.keyword_df.copy(), 'place_visit_cnt')
    df_with_tiers = analyzer._create_dynamic_tiers_kmeans(df_with_tiers, 'place_blog_cnt')

    visit_tier_data = analyzer.analyze_tier_performance(df_with_tiers, 'place_visit_cnt_tier', ['place_visit_cnt'])
    blog_tier_data = analyzer.analyze_tier_performance(df_with_tiers, 'place_blog_cnt_tier', ['place_blog_cnt'])

    visit_change_data = analyzer.analyze_tier_performance(df_with_tiers, 'place_visit_cnt_tier', ['place_visit_cnt_compare'])
    blog_change_data = analyzer.analyze_tier_performance(df_with_tiers, 'place_blog_cnt_tier', ['place_blog_cnt_compare'])

    sim_data = analyzer.simulate_clinic_performance(df_with_tiers, TARGET_CLINIC, visit_tier_data, blog_tier_data)

    all_data_for_report = {
        'visit_tier': visit_tier_data,
        'blog_tier': blog_tier_data,
        'visit_change': visit_change_data,
        'blog_change': blog_change_data,
        'simulation': sim_data
    }
    analyzer.generate_report(all_data_for_report)

if __name__ == "__main__":
    main()
