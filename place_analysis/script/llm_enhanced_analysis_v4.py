import pandas as pd
import numpy as np
import os
import glob
import warnings
import json
from dotenv import load_dotenv, find_dotenv
from agno.agent import Agent
from agno.models.google import Gemini

warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

# --- Configuration ---
PROCESSED_DATA_DIR = 'blog_automation/place_analysis/data/processed_data/지역별_검색순위'
RESULT_PATH = 'blog_automation/place_analysis/analysis_result'
REPORT_OUTPUT_PATH = os.path.join(RESULT_PATH, 'llm_enhanced_analysis_v4_report.md')
TARGET_KEYWORD = '동탄치과'
TARGET_CLINIC = '내이튼치과의원'

class RankAnalyzerV4:
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
        full_df.dropna(subset=['place_name', 'place_rank'], inplace=True)
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

    def _get_dynamic_tiers(self, df):
        """Creates dynamic tiers based on review count quantiles."""
        print("동적 등급(Dynamic Tiers) 설정 중...")
        # Define quantiles for tier creation
        quantiles = [0, 0.5, 0.7, 0.85, 0.95, 1.0]
        labels = ['Tier 5 (하위 50%)', 'Tier 4 (50-70%)', 'Tier 3 (70-85%)', 'Tier 2 (85-95%)', 'Tier 1 (상위 5%)']
        
        # Visitor Review Tiers
        df['visit_tier'] = pd.qcut(df['place_visit_cnt'], q=quantiles, labels=labels, duplicates='drop')
        
        # Blog Review Tiers
        df['blog_tier'] = pd.qcut(df['place_blog_cnt'], q=quantiles, labels=labels, duplicates='drop')
        
        return df

    def analyze_tier_performance(self, df_with_tiers, tier_col, value_cols):
        """Analyzes performance for a given tier type (visit or blog)."""
        agg_funcs = ['mean', 'std']
        analysis = df_with_tiers.groupby(tier_col)[value_cols].agg(agg_funcs).reset_index()
        
        # Flatten multi-index columns
        analysis.columns = ['_'.join(col).strip() for col in analysis.columns.values]
        analysis.rename(columns={f'{tier_col}_': 'tier'}, inplace=True)
        
        # Clean up and format
        for col in analysis.columns[1:]:
             analysis[col] = analysis[col].fillna(0).round(1)
        return analysis

    def simulate_clinic_performance(self, df_with_tiers, keyword, clinic_name, visit_tier_data, blog_tier_data):
        print(f"'{clinic_name}' 맞춤형 시뮬레이션 중...")
        target_clinic = df_with_tiers[df_with_tiers['place_name'].str.contains(clinic_name, na=False) & (df_with_tiers['search_keyword'] == keyword)]
        if target_clinic.empty: return None

        clinic_info = target_clinic.iloc[0]
        
        # Find current tiers
        current_visit_tier = clinic_info['visit_tier']
        current_blog_tier = clinic_info['blog_tier']

        # Find target review counts to level up
        def get_levelup_target(tier_data, current_tier, value_col):
            if current_tier == tier_data['tier'].iloc[-1]: return "이미 최상위 등급" # Already at top tier
            current_tier_index = tier_data[tier_data['tier'] == current_tier].index[0]
            next_tier_info = tier_data.iloc[current_tier_index + 1]
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

    def generate_report(self, visit_tier_data, blog_tier_data, visit_change_data, blog_change_data, sim_data):
        print("\nLLM 기반 최종 보고서 생성 (v4)...")
        if not self.agent: return

        # ... (LLM Base Prompt would be here, more complex to handle different sections)
        
        md_content = "# [V4] 네이버 플레이스 순위 분석 보고서: Dynamic Tier 기반 전략\n\n"

        # ... (LLM calls for each section, feeding the specific data)
        # This part requires more complex prompt engineering to generate a cohesive report from multiple data pieces.
        # For brevity, I'll just show the structure.

        md_content += "## 1. 방문자 리뷰 등급(Tier) 분석: 고객 신뢰도의 현주소\n"
        md_content += f"{visit_tier_data.to_markdown(index=False)}\n\n"
        md_content += "## 2. 블로그 리뷰 등급(Tier) 분석: 온라인 인지도의 현주소\n"
        md_content += f"{blog_tier_data.to_markdown(index=False)}\n\n"
        md_content += "## 3. 등급별 성장 속도 분석: 경쟁의 현재 속도\n"
        md_content += "### 방문자 리뷰 성장 속도\n"
        md_content += f"{visit_change_data.to_markdown(index=False)}\n\n"
        md_content += "### 블로그 리뷰 성장 속도\n"
        md_content += f"{blog_change_data.to_markdown(index=False)}\n\n"
        md_content += "## 4. 맞춤형 전략 시뮬레이션\n"
                # Convert numpy types to native Python types for JSON serialization
        if sim_data:
            for key, value in sim_data.items():
                if isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
                    sim_data[key] = int(value)
                elif isinstance(value, (np.float64, np.float32)):
                    sim_data[key] = float(value)
        
        md_content += f"```json\n{json.dumps(sim_data, indent=2, ensure_ascii=False)}\n```\n"

        with open(REPORT_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"\n분석 보고서(v4)가 다음 경로에 저장되었습니다: {REPORT_OUTPUT_PATH}")


def main():
    analyzer = RankAnalyzerV4(data_dir=PROCESSED_DATA_DIR, result_path=RESULT_PATH)
    if analyzer.full_df.empty: return

    # Create dynamic tiers
    df_with_tiers = analyzer._get_dynamic_tiers(analyzer.full_df)

    # 1. Analyze Tier Performance
    visit_tier_data = analyzer.analyze_tier_performance(df_with_tiers, 'visit_tier', ['place_visit_cnt'])
    blog_tier_data = analyzer.analyze_tier_performance(df_with_tiers, 'blog_tier', ['place_blog_cnt'])

    # 2. Analyze Tier Growth
    visit_change_data = analyzer.analyze_tier_performance(df_with_tiers, 'visit_tier', ['place_visit_cnt_compare'])
    blog_change_data = analyzer.analyze_tier_performance(df_with_tiers, 'blog_tier', ['place_blog_cnt_compare'])

    # 3. Simulation
    sim_data = analyzer.simulate_clinic_performance(df_with_tiers, TARGET_KEYWORD, TARGET_CLINIC, visit_tier_data, blog_tier_data)

    # 4. Generate Report
    analyzer.generate_report(visit_tier_data, blog_tier_data, visit_change_data, blog_change_data, sim_data)

if __name__ == "__main__":
    main()
