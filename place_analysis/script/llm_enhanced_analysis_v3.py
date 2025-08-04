import pandas as pd
import numpy as np
import os
import glob
import warnings
from dotenv import load_dotenv, find_dotenv
from agno.agent import Agent
from agno.models.google import Gemini

warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

# --- Configuration ---
PROCESSED_DATA_DIR = 'blog_automation/place_analysis/data/processed_data/지역별_검색순위'
RESULT_PATH = 'blog_automation/place_analysis/analysis_result'
REPORT_OUTPUT_PATH = os.path.join(RESULT_PATH, 'llm_enhanced_analysis_v3_report.md')
TARGET_KEYWORD = '동탄치과'
TARGET_CLINIC = '내이튼치과의원'

# --- Data Loading and Analysis Functions ---

class RankAnalyzerV3:
    def __init__(self, data_dir, result_path):
        self.data_dir = data_dir
        self.result_path = result_path
        self.full_df = self._load_and_prepare_data()
        self.agent = self._initialize_llm_agent()
        os.makedirs(self.result_path, exist_ok=True)

    def _load_and_prepare_data(self):
        """Load all CSV files and concatenate them."""
        print(f"데이터 로딩 시작: '{self.data_dir}'")
        csv_files = glob.glob(os.path.join(self.data_dir, '*.csv'))
        if not csv_files:
            print(f"오류: '{self.data_dir}' 디렉토리에서 CSV 파일을 찾을 수 없습니다.")
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
        
        for col in ['place_visit_cnt', 'place_blog_cnt', 'place_rank_compare', 'place_visit_cnt_compare', 'place_blog_cnt_compare']:
            if col in full_df.columns:
                full_df[col] = pd.to_numeric(full_df[col], errors='coerce').fillna(0)

        full_df.dropna(subset=['place_name'], inplace=True)
        full_df['place_name'] = full_df['place_name'].astype(str)
        print("데이터 로딩 및 전처리 완료.")
        return full_df

    def _initialize_llm_agent(self):
        print("LLM 에이전트 초기화 중...")
        load_dotenv(find_dotenv())
        try:
            # v2 스크립트의 초기화 방식과 동일하게 수정
            llm = Gemini(id=os.getenv("GEMINI_MODEL", "gemini/gemini-1.5-pro-latest"), api_key=os.getenv("GEMINI_API_KEY"))
            return Agent(model=llm)
        except Exception as e:
            print(f"LLM 에이전트 초기화 실패: {e}. 보고서 생성 시 해석 부분이 제외됩니다.")
            return None

    def analyze_rank_tier_performance(self):
        """Analyzes the average review counts for different rank tiers."""
        print("분석 1: 등급별 목표치 분석 중...")
        if self.full_df.empty or 'place_rank' not in self.full_df.columns: return None
        
        df = self.full_df.copy()
        df['rank_tier'] = pd.cut(df['place_rank'], 
                                 bins=[0, 10, 20, 30, 40, 50, 100, 200, 500], 
                                 labels=['1-10위', '11-20위', '21-30위', '31-40위', '41-50위', '51-100위', '101-200위', '201-500위'],
                                 right=True)
        
        tier_analysis = df.groupby('rank_tier')[['place_visit_cnt', 'place_blog_cnt']].mean().reset_index()
        tier_analysis[['place_visit_cnt', 'place_blog_cnt']] = tier_analysis[['place_visit_cnt', 'place_blog_cnt']].fillna(0).round(0).astype(int)
        return tier_analysis

    def analyze_rank_change_drivers(self):
        """Compares review metrics for clinics with positive vs. negative/stagnant rank changes."""
        print("분석 2: 순위 상승/하락 그룹 비교 분석 중...")
        if self.full_df.empty or 'place_rank_compare' not in self.full_df.columns: return None

        df = self.full_df[self.full_df['place_rank_compare'] != 0].copy()
        if df.empty: return None

        df['rank_improved'] = df['place_rank_compare'] < 0

        df['prev_visit_cnt'] = df['place_visit_cnt'] - df['place_visit_cnt_compare']
        df['prev_blog_cnt'] = df['place_blog_cnt'] - df['place_blog_cnt_compare']
        
        df['visit_change_rate'] = (df['place_visit_cnt_compare'] / df['prev_visit_cnt'].replace(0, np.nan) * 100).fillna(0)
        df['blog_change_rate'] = (df['place_blog_cnt_compare'] / df['prev_blog_cnt'].replace(0, np.nan) * 100).fillna(0)

        change_analysis = df.groupby('rank_improved')[['place_visit_cnt_compare', 'place_blog_cnt_compare', 'visit_change_rate', 'blog_change_rate']].mean().reset_index()
        change_analysis.rename(columns={
            'place_visit_cnt_compare': '평균 방문자 리뷰 증감',
            'place_blog_cnt_compare': '평균 블로그 리뷰 증감',
            'visit_change_rate': '평균 방문자 리뷰 증가율(%)',
            'blog_change_rate': '평균 블로그 리뷰 증가율(%)'
        }, inplace=True)
        
        change_analysis['rank_improved'] = change_analysis['rank_improved'].map({True: '순위 상승 그룹', False: '순위 하락/유지 그룹'})
        return change_analysis

    def simulate_top_10_entry(self, keyword, clinic_name):
        """Calculates the required review increases for a target clinic to match the Top 10 average."""
        print(f"분석 3: '{clinic_name}' TOP 10 진입 시뮬레이션 중...")
        df_keyword = self.full_df[self.full_df['search_keyword'] == keyword]
        if df_keyword.empty: return None

        top_10_df = df_keyword[df_keyword['place_rank'] <= 10]
        target_clinic_df = df_keyword[df_keyword['place_name'].str.contains(clinic_name, na=False)]

        if top_10_df.empty or target_clinic_df.empty: return None
            
        top_10_avg = top_10_df[['place_visit_cnt', 'place_blog_cnt']].mean()
        clinic_current_stats = target_clinic_df[['place_name', 'place_rank', 'place_visit_cnt', 'place_blog_cnt']].iloc[0]

        return {
            'target_clinic_name': clinic_current_stats['place_name'],
            'current_rank': clinic_current_stats['place_rank'],
            'current_visit_reviews': clinic_current_stats['place_visit_cnt'],
            'current_blog_reviews': clinic_current_stats['place_blog_cnt'],
            'top_10_avg_visit_reviews': int(round(top_10_avg['place_visit_cnt'])),
            'top_10_avg_blog_reviews': int(round(top_10_avg['place_blog_cnt'])),
            'needed_visit_reviews': int(max(0, round(top_10_avg['place_visit_cnt'] - clinic_current_stats['place_visit_cnt']))),
            'needed_blog_reviews': int(max(0, round(top_10_avg['place_blog_cnt'] - clinic_current_stats['place_blog_cnt']))),
        }

    def generate_report(self, tier_data, change_data, sim_data):
        """Generates the final v3 report using the LLM agent."""
        print("\nLLM 기반 최종 보고서 생성 (v3)...")
        if not self.agent:
            print("LLM 에이전트가 없어 보고서 생성을 건너뜁니다."); return

        base_prompt = """
        ## 시스템 지침: 당신의 역할과 분석의 맥락 ##
        당신은 대한민국 '네이버 플레이스' 순위 데이터를 분석하여 병원 마케팅 담당자에게 실행 가능한 전략을 제안하는 최고의 데이터 분석 컨설턴트입니다. 당신의 답변은 반드시 아래의 맥락 정보를 완벽하게 숙지하고, 전문성과 신뢰성을 담아 한국어 마크다운 형식으로 작성되어야 합니다.
        
        - **핵심 목표:** 분석 결과를 바탕으로 고객사가 '리뷰 관리 서비스'를 구매하도록 설득하는 것이 최종 목표입니다.
        - **분석의 논리 흐름:**
            1.  **'목표 제시'**: 상위권 병원의 평균 리뷰 수를 보여주며 성공의 기준점을 제시한다.
            2.  **'필요성 증명'**: 순위 상승 그룹의 리뷰 증가율이 훨씬 높다는 것을 보여주며 리뷰 관리의 중요성을 데이터로 증명한다.
            3.  **'맞춤형 전략 제안'**: 타겟 병원이 경쟁사를 제치고 목표 순위에 도달하기 위해 필요한 구체적인 리뷰 증가량을 계산하여 제시한다.
        - **핵심 키워드:** '목표 설정', '격차 확인', '필요성 증명', '액션 플랜', '기대 효과'
        
        ---
        
        ## 사용자 요청: 보고서 섹션 작성 ##
        이제 아래 주어진 **분석 데이터**와 **작성 주제**에 맞춰, 위에서 학습한 모든 맥락을 총동원하여 보고서의 한 섹션을 작성해주세요.
        """
        
        md_content = "# [V3] 네이버 플레이스 순위 분석 보고서: 실행 중심 전략 제안\n\n"

        # --- Section 1: Rank Tier ---
        if tier_data is not None:
            print("리포트 섹션 1 생성 중...")
            prompt = base_prompt + f"""
            **분석 데이터:**\n{tier_data.to_markdown(index=False)}
            
            **작성할 보고서 섹션 주제:** '1. [목표 제시] 순위 등급별 현황 분석: 성공의 기준점'
            
            **가이드라인:** 전국 200개 지역의 네이버 지도 검색 1-50위 치과 데이터(총 10,000개 치과)를 분석했습니다. 이 데이터를 통해 "성공하는 병원들은 이 정도의 리뷰를 가지고 있다"는 메시지를 명확히 전달하며, 상위 등급으로 가기 위한 목표치를 설정하는 데 도움이 되는 전문적인 분석 보고서 섹션을 작성해주세요.
            """
            response = self.agent.run(prompt)
            md_content += response.content + "\n\n---\n\n" if response else ""

        # --- Section 2: Change Drivers ---
        if change_data is not None:
            print("리포트 섹션 2 생성 중...")
            prompt = base_prompt + f"""
            **분석 데이터:**\n{change_data.to_markdown(index=False)}
            
            **작성할 보고서 섹션 주제:** '2. [필요성 증명] 순위 변동 그룹 비교: 무엇이 차이를 만드는가?'

            **가이드라인:** "순위가 오른 곳은 다 이유가 있다. 바로 '리뷰의 꾸준한 증가'이다." 라는 메시지를 강력하게 전달하며, 리뷰 관리의 중요성을 데이터로 증명하는 설득력 있는 보고서 섹션을 작성해주세요.
            """
            response = self.agent.run(prompt)
            md_content += response.content + "\n\n---\n\n" if response else ""

        # --- Section 3: Simulation ---
        if sim_data is not None:
            print("리포트 섹션 3 생성 중...")
            sim_text = "\n".join([f"- {key}: {value}" for key, value in sim_data.items()])
            prompt = base_prompt + f"""
            **분석 데이터:**\n{sim_text}

            **작성할 보고서 섹션 주제:** "3. [전략 제안] '{sim_data['target_clinic_name']}' TOP 10 진입을 위한 맞춤형 액션 플랜"

            **가이드라인:** 일반적인 분석이 아닌, 오직 이 병원만을 위한 구체적인 액션 플랜을 제시해야 합니다. "TOP 10에 가려면, 방문자 리뷰 X개, 블로그 리뷰 Y개가 더 필요합니다." 라는 결론을 명확히 내려주고, 서비스 구매를 유도하는 강력한 마무리 멘트를 포함해주세요.
            """
            response = self.agent.run(prompt)
            md_content += response.content if response else ""

        with open(REPORT_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"\n분석 보고서(v3)가 다음 경로에 저장되었습니다: {REPORT_OUTPUT_PATH}")


# --- Main Execution Block ---
def main():
    analyzer = RankAnalyzerV3(data_dir=PROCESSED_DATA_DIR, result_path=RESULT_PATH)
    if analyzer.full_df.empty:
        return

    tier_data = analyzer.analyze_rank_tier_performance()
    change_data = analyzer.analyze_rank_change_drivers()
    sim_data = analyzer.simulate_top_10_entry(keyword=TARGET_KEYWORD, clinic_name=TARGET_CLINIC)
    
    analyzer.generate_report(tier_data, change_data, sim_data)

if __name__ == "__main__":
    main()
