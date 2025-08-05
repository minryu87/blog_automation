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

class RankAnalyzerV3:
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
        
        for col in ['place_rank', 'place_visit_cnt', 'place_blog_cnt', 'place_rank_compare', 'place_visit_cnt_compare', 'place_blog_cnt_compare']:
            if col in full_df.columns:
                full_df[col] = pd.to_numeric(full_df[col], errors='coerce')

        full_df.dropna(subset=['place_name', 'place_rank'], inplace=True)
        full_df['place_name'] = full_df['place_name'].astype(str)
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

    def analyze_rank_tier_performance(self):
        """
        [V3-1] Analyzes the average and std dev of review counts for different rank tiers.
        """
        print("분석 1: 등급별 현황 분석 (평균 및 표준편차) 중...")
        if self.full_df.empty: return None
        
        df = self.full_df.copy()
        df['rank_tier'] = pd.cut(df['place_rank'], 
                                 bins=[0, 10, 20, 30, 40, 50, 100, 200, 500], 
                                 labels=['1-10위', '11-20위', '21-30위', '31-40위', '41-50위', '51-100위', '101-200위', '201-500위'],
                                 right=True)
        
        agg_funcs = ['mean', 'std']
        tier_analysis = df.groupby('rank_tier')[['place_visit_cnt', 'place_blog_cnt']].agg(agg_funcs).reset_index()
        tier_analysis.columns = ['rank_tier', 'visit_cnt_mean', 'visit_cnt_std', 'blog_cnt_mean', 'blog_cnt_std']
        
        for col in ['visit_cnt_mean', 'visit_cnt_std', 'blog_cnt_mean', 'blog_cnt_std']:
            tier_analysis[col] = tier_analysis[col].fillna(0).round(0).astype(int)
            
        return tier_analysis

    def analyze_rank_tier_change(self):
        """
        [V3-2] Analyzes the average and std dev of 10-day review changes for different rank tiers.
        """
        print("분석 2: 등급별 '10일간 리뷰 증감' 분석 (평균 및 표준편차) 중...")
        if self.full_df.empty: return None

        df = self.full_df.copy()
        df['rank_tier'] = pd.cut(df['place_rank'], 
                                 bins=[0, 10, 20, 30, 40, 50, 100, 200, 500], 
                                 labels=['1-10위', '11-20위', '21-30위', '31-40위', '41-50위', '51-100위', '101-200위', '201-500위'],
                                 right=True)

        agg_funcs = ['mean', 'std']
        change_analysis = df.groupby('rank_tier')[['place_visit_cnt_compare', 'place_blog_cnt_compare']].agg(agg_funcs).reset_index()
        change_analysis.columns = ['rank_tier', 'visit_change_mean', 'visit_change_std', 'blog_change_mean', 'blog_change_std']
        
        for col in ['visit_change_mean', 'visit_change_std', 'blog_change_mean', 'blog_change_std']:
            change_analysis[col] = change_analysis[col].fillna(0).round(1)

        return change_analysis


    def simulate_top_10_entry(self, keyword, clinic_name, tier_change_data):
        """
        [V3-3] Calculates required review increases and 10-day goals for a target clinic.
        """
        print(f"분석 3: '{clinic_name}' TOP 10 진입 시뮬레이션 중...")
        if self.full_df.empty: return None
        df_keyword = self.full_df[self.full_df['search_keyword'] == keyword]
        if df_keyword.empty: return None

        top_10_df = df_keyword[df_keyword['place_rank'] <= 10]
        target_clinic_df = df_keyword[df_keyword['place_name'].str.contains(clinic_name, na=False)]

        if top_10_df.empty or target_clinic_df.empty: return None
            
        top_10_avg = top_10_df[['place_visit_cnt', 'place_blog_cnt']].mean()
        clinic_current_stats = target_clinic_df[['place_name', 'place_rank', 'place_visit_cnt', 'place_blog_cnt']].iloc[0]

        # Get 10-day goal from tier_change_data
        ten_day_goal = tier_change_data[tier_change_data['rank_tier'] == '1-10위'] if tier_change_data is not None else None
        
        return {
            'target_clinic_name': clinic_current_stats['place_name'],
            'current_rank': clinic_current_stats['place_rank'],
            'current_visit_reviews': clinic_current_stats['place_visit_cnt'],
            'current_blog_reviews': clinic_current_stats['place_blog_cnt'],
            'top_10_avg_visit_reviews': int(round(top_10_avg['place_visit_cnt'])),
            'top_10_avg_blog_reviews': int(round(top_10_avg['place_blog_cnt'])),
            'needed_visit_reviews': int(max(0, round(top_10_avg['place_visit_cnt'] - clinic_current_stats['place_visit_cnt']))),
            'needed_blog_reviews': int(max(0, round(top_10_avg['place_blog_cnt'] - clinic_current_stats['place_blog_cnt']))),
            '10_day_visit_goal': ten_day_goal['visit_change_mean'].iloc[0] if ten_day_goal is not None and not ten_day_goal.empty else 'N/A',
            '10_day_blog_goal': ten_day_goal['blog_change_mean'].iloc[0] if ten_day_goal is not None and not ten_day_goal.empty else 'N/A'
        }

    def generate_report(self, tier_data, change_data, sim_data):
        print("\nLLM 기반 최종 보고서 생성 (v3)...")
        if not self.agent:
            print("LLM 에이전트가 없어 보고서 생성을 건너뜁니다."); return

        base_prompt = """
        ## 시스템 지침: 당신의 역할과 분석의 맥락 ##
        당신은 대한민국 '네이버 플레이스' 순위 데이터를 분석하여 병원 마케팅 담당자에게 실행 가능한 전략을 제안하는 최고의 데이터 분석 컨설턴트입니다. 당신의 답변은 반드시 아래의 맥락 정보를 완벽하게 숙지하고, 전문성과 신뢰성을 담아 한국어 마크다운 형식으로 작성되어야 합니다.
        
        - **핵심 목표:** 분석 결과를 바탕으로 고객사가 '리뷰 관리 서비스'를 구매하도록 설득하는 것이 최종 목표입니다.
        - **분석의 논리 흐름:**
            1.  **'목표 제시'**: 상위권 병원의 평균 리뷰 수를 보여주며 성공의 기준점을 제시한다. (표준편차를 언급하며 데이터의 분포도 함께 설명)
            2.  **'성장 속도 제시'**: 상위권 병원들의 '10일간 평균 리뷰 증가량'을 보여주며, 현재의 경쟁 속도를 인지시킨다.
            3.  **'맞춤형 전략 제안'**: 타겟 병원이 목표 순위에 도달하기 위해 필요한 '전체 리뷰 목표'와 '단기 성장 목표'를 구체적으로 계산하여 제시한다.
        - **핵심 키워드:** '목표 설정', '경쟁 강도', '성장 속도', '단기 목표', '액션 플랜', '기대 효과'
        
        ---
        
        ## 사용자 요청: 보고서 섹션 작성 ##
        이제 아래 주어진 **분석 데이터**와 **작성 주제**에 맞춰, 위에서 학습한 모든 맥락을 총동원하여 보고서의 한 섹션을 작성해주세요.
        """
        
        md_content = "# [V3] 네이버 플레이스 순위 분석 보고서: 실행 중심 전략 제안\n\n"

        # Section 1: Rank Tier
        if tier_data is not None:
            print("리포트 섹션 1 (등급별 현황) 생성 중...")
            prompt = base_prompt + f"""
            **분석 데이터:**\n{tier_data.rename(columns={'visit_cnt_mean': '방문자리뷰 평균', 'visit_cnt_std': '방문자리뷰 표준편차', 'blog_cnt_mean': '블로그리뷰 평균', 'blog_cnt_std': '블로그리뷰 표준편차'}).to_markdown(index=False)}
            
            **작성할 보고서 섹션 주제:** '1. [목표 제시] 순위 등급별 현황 분석: 성공의 기준점'
            
            **가이드라인:** "성공하는 병원들은 평균적으로 이 정도의 리뷰를 가지고 있다"는 메시지를 명확히 전달해주세요. 특히 '표준편차' 데이터를 활용하여, "경쟁이 치열한 상위권일수록 편차가 적어, 일정 수준 이상의 리뷰는 필수"라는 점과 "하위권은 편차가 커서 기회와 위기가 공존한다"는 점을 함께 해석하여 전문성을 더해주세요.
            """
            response = self.agent.run(prompt)
            md_content += response.content + "\n\n---\n\n" if response and hasattr(response, 'content') else ""

        # Section 2: Tier Change
        if change_data is not None:
            print("리포트 섹션 2 (등급별 성장률) 생성 중...")
            prompt = base_prompt + f"""
            **분석 데이터:**\n{change_data.rename(columns={'visit_change_mean': '10일간 방문자리뷰 평균 증감', 'visit_change_std': '증감 표준편차', 'blog_change_mean': '10일간 블로그리뷰 평균 증감', 'blog_change_std': '증감 표준편차'}).to_markdown(index=False)}
            
            **작성할 보고서 섹션 주제:** '2. [성장 속도 인지] 순위 등급별 '10일' 리뷰 증감 분석: 경쟁의 현재 속도'

            **가이드라인:** "상위권 병원들은 10일마다 이만큼 성장하고 있다. 이것이 현재 시장의 경쟁 속도다." 라는 메시지를 전달해야 합니다. '평균 증감' 데이터를 통해 목표로 해야 할 단기 성장률을 제시하고, '표준편차'를 통해 이 경쟁이 얼마나 치열하고 꾸준한지를 설명해주세요.
            """
            response = self.agent.run(prompt)
            md_content += response.content + "\n\n---\n\n" if response and hasattr(response, 'content') else ""

        # Section 3: Simulation
        if sim_data is not None:
            print("리포트 섹션 3 (맞춤형 시뮬레이션) 생성 중...")
            sim_text = "\n".join([f"- {key}: {value}" for key, value in sim_data.items()])
            prompt = base_prompt + f"""
            **분석 데이터:**\n{sim_text}

            **작성할 보고서 섹션 주제:** "3. [전략 제안] '{sim_data['target_clinic_name']}' TOP 10 진입을 위한 맞춤형 액션 플랜"

            **가이드라인:** '내이튼치과의원'만을 위한 구체적인 액션 플랜을 제시해야 합니다.
            1. **장기 목표:** TOP 10 진입에 필요한 '총 리뷰 증가량' 제시.
            2. **단기 목표:** 경쟁 속도에 뒤처지지 않기 위한 '**10일간의 리뷰 증가 목표**' 제시.
            3. 이 두 가지 목표를 달성하기 위한 저희 서비스의 필요성을 강력하게 어필하며 마무리해주세요.
            """
            response = self.agent.run(prompt)
            md_content += response.content if response and hasattr(response, 'content') else ""

        with open(REPORT_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"\n분석 보고서(v3)가 다음 경로에 저장되었습니다: {REPORT_OUTPUT_PATH}")

def main():
    analyzer = RankAnalyzerV3(data_dir=PROCESSED_DATA_DIR, result_path=RESULT_PATH)
    if analyzer.full_df.empty: return

    tier_data = analyzer.analyze_rank_tier_performance()
    change_data = analyzer.analyze_rank_tier_change()
    sim_data = analyzer.simulate_top_10_entry(keyword=TARGET_KEYWORD, clinic_name=TARGET_CLINIC, tier_change_data=change_data)
    
    analyzer.generate_report(tier_data, change_data, sim_data)

if __name__ == "__main__":
    main()
