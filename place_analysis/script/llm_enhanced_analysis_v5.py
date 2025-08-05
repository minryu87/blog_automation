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
REPORT_OUTPUT_PATH = os.path.join(RESULT_PATH, 'llm_enhanced_analysis_v5_report.md')
TARGET_KEYWORD = '동탄치과'
TARGET_CLINIC = '내이튼치과의원'

class RankAnalyzerV5:
    def __init__(self, data_dir, result_path):
        self.data_dir = data_dir
        self.result_path = result_path
        self.full_df = self._load_and_prepare_data()
        self.agent = self._initialize_llm_agent()
        os.makedirs(self.result_path, exist_ok=True)

    def _load_and_prepare_data(self):
        print(f"데이터 로딩 시작: '{self.data_dir}'")
        csv_files = glob.glob(os.path.join(self.data_dir, '*.csv'))
        if not csv_files:
            print(f"오류: CSV 파일을 찾을 수 없습니다.")
            return pd.DataFrame()
        
        df_list = []
        for f in csv_files:
            try:
                df = pd.read_csv(f)
                df['search_keyword'] = os.path.basename(f).replace('.csv', '')
                df_list.append(df)
            except Exception as e:
                print(f"'{f}' 파일 로드 중 오류 발생: {e}")
                
        if not df_list: return pd.DataFrame()
            
        full_df = pd.concat(df_list, ignore_index=True)
        
        for col in ['place_rank', 'place_visit_cnt', 'place_blog_cnt', 'place_rank_compare', 'place_visit_cnt_compare', 'place_blog_cnt_compare']:
            if col in full_df.columns:
                full_df[col] = pd.to_numeric(full_df[col], errors='coerce')
        full_df.dropna(subset=['place_name', 'place_rank', 'place_visit_cnt', 'place_blog_cnt'], inplace=True)
        print("데이터 로딩 및 전처리 완료.")
        return full_df

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
        """Creates dynamic tiers using K-Means clustering. Tier 1 is the highest."""
        print(f"'{column_name}'에 대한 K-평균 클러스터링 기반 동적 등급 생성 중...")
        
        data = df[[column_name]].dropna()
        if data.empty:
            return df
            
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_data)
        
        df.loc[data.index, f'{column_name}_cluster'] = clusters
        
        # V5.1 Change: Order clusters by their center value descending (Tier 1 is highest)
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        cluster_order = np.argsort(cluster_centers.flatten())[::-1] # Note the [::-1] for descending
        
        tier_names = [f'Tier {i+1} (Avg: {int(round(cluster_centers[cluster_order[i]][0]))})' for i in range(n_clusters)]
        tier_map = {cluster_order[i]: tier_names[i] for i in range(n_clusters)}
        
        tier_col_name = f'{column_name}_tier'
        df[tier_col_name] = df[f'{column_name}_cluster'].map(tier_map)
        
        return df.drop(columns=[f'{column_name}_cluster'])

    def analyze_tier_performance(self, df, tier_col, value_cols):
        """Analyzes performance for a given tier type."""
        agg_funcs = ['mean', 'std']
        analysis = df.groupby(tier_col)[value_cols].agg(agg_funcs).reset_index()
        
        analysis.columns = ['_'.join(col).strip() for col in analysis.columns.values]
        analysis.rename(columns={f'{tier_col}_': 'tier'}, inplace=True)
        
        for col in analysis.columns[1:]:
             analysis[col] = analysis[col].fillna(0).round(1)
        # V5.1 Change: Sort by Tier name to ensure Tier 1 is at the top
        return analysis.sort_values(by='tier', ascending=True)

    def simulate_clinic_performance(self, df, keyword, clinic_name, visit_tier_data, blog_tier_data):
        print(f"'{clinic_name}' 맞춤형 시뮬레이션 중...")
        target_clinic = df[df['place_name'].str.contains(clinic_name, na=False) & (df['search_keyword'] == keyword)]
        if target_clinic.empty: return None

        clinic_info = target_clinic.iloc[0]
        
        current_visit_tier = clinic_info['place_visit_cnt_tier']
        current_blog_tier = clinic_info['place_blog_cnt_tier']

        def get_levelup_target(tier_data, current_tier, value_col):
            # V5.1 Change: Logic to find the next higher tier
            if current_tier == tier_data['tier'].iloc[0]: return "이미 최상위 등급"
            
            # Since tiers are sorted alphabetically (Tier 1, Tier 10...), we need to be careful
            # We will sort by the mean value to get the correct order
            sorted_tiers = tier_data.sort_values(by=f'{value_col}_mean', ascending=False).reset_index(drop=True)
            current_tier_index = sorted_tiers[sorted_tiers['tier'] == current_tier].index[0]

            if current_tier_index == 0: return "이미 최상위 등급" # Should not happen due to first check, but as a safeguard
            
            next_tier_info = sorted_tiers.iloc[current_tier_index - 1]
            return int(round(next_tier_info[f'{value_col}_mean']))

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
        print("\nLLM 기반 최종 보고서 생성 (v5)...")
        if not self.agent: return

        base_prompt = """
        ## 시스템 지침: 당신의 역할과 분석의 맥락 (v5.1) ##
        당신은 대한민국 '네이버 플레이스' 순위 데이터를 분석하여 병원 마케터에게 실행 가능한 전략을 제안하는 최고의 데이터 분석 컨설턴트입니다. 당신의 답변은 반드시 아래의 맥락 정보를 완벽하게 숙지하고, 전문성과 신뢰성을 담아 한국어 마크다운 형식으로 작성되어야 합니다.
        
        - **v5.1 핵심 변경사항:** 모든 데이터에서 **'Tier 1'이 가장 높은 등급(가장 리뷰 수가 많은 그룹)**을 의미합니다. 보고서의 모든 서술은 이 기준에 맞춰 일관성 있게 작성되어야 합니다.
        - **핵심 목표:** K-평균 클러스터링 기반의 '동적 등급' 분석 결과를 바탕으로, 고객사가 '리뷰 관리 서비스'를 구매하도록 설득하는 것이 최종 목표입니다.
        - **분석의 논리 흐름:**
            1.  **'현실 진단'**: 방문자/블로그 리뷰 각각의 기준으로, 병원들이 어떻게 통계적 그룹(Tier 1~5)으로 나뉘는지 보여줍니다. (Tier 1이 최상위 그룹)
            2.  **'성장 속도 제시'**: 각 Tier에 속한 병원들의 '10일간 평균 리뷰 증가량'을 보여주며, Tier별 경쟁 속도를 인지시킨다.
            3.  **'맞춤형 전략 제안'**: 타겟 병원의 현재 Tier를 진단하고, 다음 상위 Tier로 올라서기 위해 필요한 '리뷰 목표'를 구체적으로 계산하여 제시한다.
        - **핵심 키워드:** 'Tier 1 (최상위 그룹)', 'K-평균 클러스터링', '동적 등급', '통계적 유의미성', '맞춤형 목표', 'Tier 상승 전략'
        
        ---
        
        ## 사용자 요청: 보고서 작성 ##
        이제 아래 주어진 **모든 분석 데이터**를 종합적으로 해석하고, 위에서 학습한 모든 맥락을 총동원하여 **완성된 하나의 보고서**를 작성해주세요.
        """
        
        # Convert numpy types to native Python types for JSON serialization
        if all_data.get('simulation'):
            sim_data = all_data['simulation']
            for key, value in sim_data.items():
                if isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
                    sim_data[key] = int(value)
                elif isinstance(value, (np.float64, np.float32)):
                    sim_data[key] = float(value)

        full_prompt = base_prompt + f"\n\n### 종합 분석 데이터 ###\n\n" + \
                      f"#### 1-1. 방문자 리뷰 동적 등급(Tier)별 현황 (Tier 1이 최상위) ####\n{all_data['visit_tier'].to_markdown(index=False)}\n\n" + \
                      f"#### 1-2. 블로그 리뷰 동적 등급(Tier)별 현황 (Tier 1이 최상위) ####\n{all_data['blog_tier'].to_markdown(index=False)}\n\n" + \
                      f"#### 2-1. 방문자 리뷰 등급별 '10일' 성장 속도 (Tier 1이 최상위) ####\n{all_data['visit_change'].to_markdown(index=False)}\n\n" + \
                      f"#### 2-2. 블로그 리뷰 등급별 '10일' 성장 속도 (Tier 1이 최상위) ####\n{all_data['blog_change'].to_markdown(index=False)}\n\n" + \
                      f"#### 3. '{all_data['simulation']['clinic_name']}' 맞춤형 전략 시뮬레이션 ####\n" + \
                      f"```json\n{json.dumps(all_data['simulation'], indent=2, ensure_ascii=False)}\n```"

        response = self.agent.run(full_prompt)
        md_content = f"# [V5.1] 네이버 플레이스 순위 분석 보고서: K-Means 클러스터링 기반 (Tier 1 = 최상위)\n\n{response.content if response and hasattr(response, 'content') else '보고서 생성에 실패했습니다.'}"

        with open(REPORT_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"\n분석 보고서(v5.1)가 다음 경로에 저장되었습니다: {REPORT_OUTPUT_PATH}")


def main():
    analyzer = RankAnalyzerV5(data_dir=PROCESSED_DATA_DIR, result_path=RESULT_PATH)
    if analyzer.full_df.empty: return

    df_with_tiers = analyzer._create_dynamic_tiers_kmeans(analyzer.full_df, 'place_visit_cnt')
    df_with_tiers = analyzer._create_dynamic_tiers_kmeans(df_with_tiers, 'place_blog_cnt')
    
    visit_tier_data = analyzer.analyze_tier_performance(df_with_tiers, 'place_visit_cnt_tier', ['place_visit_cnt'])
    blog_tier_data = analyzer.analyze_tier_performance(df_with_tiers, 'place_blog_cnt_tier', ['place_blog_cnt'])

    visit_change_data = analyzer.analyze_tier_performance(df_with_tiers, 'place_visit_cnt_tier', ['place_visit_cnt_compare'])
    blog_change_data = analyzer.analyze_tier_performance(df_with_tiers, 'place_blog_cnt_tier', ['place_blog_cnt_compare'])

    sim_data = analyzer.simulate_clinic_performance(df_with_tiers, TARGET_KEYWORD, TARGET_CLINIC, visit_tier_data, blog_tier_data)

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
