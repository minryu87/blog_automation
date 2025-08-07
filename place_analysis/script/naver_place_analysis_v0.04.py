import pandas as pd
import numpy as np
import os
import glob
import warnings
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from markdown2 import Markdown
from dotenv import load_dotenv, find_dotenv
from agno.agent import Agent
from agno.models.google import Gemini
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

# --- Matplotlib Configuration for Korean Fonts ---
import matplotlib.font_manager as fm
import platform

def find_korean_fonts():
    korean_fonts = []
    if platform.system() == 'Darwin':
        font_paths = ['/System/Library/Fonts/Supplemental/AppleGothic.ttf']
        for font_path in font_paths:
            if os.path.exists(font_path):
                korean_fonts.append(font_path)
    return korean_fonts

font_found = False
korean_font_paths = find_korean_fonts()
if korean_font_paths:
    for font_path in korean_font_paths:
        try:
            fm.fontManager.addfont(font_path)
            font_name = fm.FontProperties(fname=font_path).get_name()
            plt.rc('font', family=font_name)
            plt.rcParams['axes.unicode_minus'] = False
            print(f"한글 폰트 '{font_name}' 설정 완료. (경로: {font_path})")
            font_found = True
            break
        except Exception as e:
            print(f"폰트 '{font_path}' 설정 실패: {e}")
            continue

if not font_found:
    print("경고: 적절한 한글 폰트를 찾을 수 없습니다. 그래프의 한글이 깨질 수 있습니다.")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

# --- Configuration ---
TARGET_QUERY = '동탄치과'  # 검색 키워드
RAW_DATA_DIR = f'blog_automation/place_analysis/data/raw_data/{TARGET_QUERY}'
RESULT_PATH = 'blog_automation/place_analysis/analysis_result'
os.makedirs(RESULT_PATH, exist_ok=True)
REPORT_OUTPUT_PATH_HTML = os.path.join(RESULT_PATH, 'naver_place_analysis_v0.04_report.html')
TARGET_CLINIC = '내이튼치과의원'
ANALYSIS_DATE = '2025-08-04'

class NaverPlaceAnalyzerV0_04:
    def __init__(self, data_dir, result_path):
        self.data_dir = data_dir
        self.result_path = result_path
        self.agent = self._initialize_llm_agent()
        self.time_series_df = self._load_and_prepare_time_series_data()
        self.markdown_converter = Markdown(extras=["tables"])

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
        print("시계열 데이터 로딩 및 전처리 중...")
        csv_files = glob.glob(os.path.join(self.data_dir, 'vs*.csv'))
        if not csv_files: return pd.DataFrame()
        
        all_data = []
        for f in csv_files:
            try:
                days_ago_match = re.search(r'vs(\d+)\.csv', os.path.basename(f))
                if not days_ago_match: continue
                days_ago = int(days_ago_match.group(1))
                
                df = pd.read_csv(f)
                df['company_name'] = df['company_name_category'].apply(lambda x: x.split(',')[0] if isinstance(x, str) else None)
                df.dropna(subset=['company_name'], inplace=True)
                df['days_ago'] = 0 if days_ago == 1 else days_ago

                numeric_cols = ['rank', 'rank_change', 'rank_trend', 'visitor_reviews', 'blog_reviews', 'visitor_reviews_change', 'blog_reviews_change']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

                all_data.append(df)
            except Exception as e:
                print(f"'{f}' 파일 처리 중 오류 발생: {e}")

        if not all_data: return pd.DataFrame()
        
        full_df = pd.concat(all_data, ignore_index=True)
        
        def calculate_rank(row):
            if row['days_ago'] == 0: return row['rank']
            if row['rank_trend'] == 1: return row['rank'] + row['rank_change']
            elif row['rank_trend'] == 2: return row['rank']
            elif row['rank_trend'] == 3: return row['rank'] - row['rank_change']
            else: return row['rank']
        full_df['rank'] = full_df.apply(calculate_rank, axis=1)

        current_reviews = full_df[full_df['days_ago'] == 0][['company_name', 'visitor_reviews', 'blog_reviews']].set_index('company_name')
        
        def get_past_reviews(row, review_type):
            company = row['company_name']
            if company not in current_reviews.index: return 0
            if row['days_ago'] == 0:
                return current_reviews.loc[company][review_type]
            else:
                change_col = f"{review_type}_change"
                change = row.get(change_col, 0)
                return current_reviews.loc[company][review_type] - change
        
        full_df['visitor_reviews'] = full_df.apply(get_past_reviews, review_type='visitor_reviews', axis=1)
        full_df['blog_reviews'] = full_df.apply(get_past_reviews, review_type='blog_reviews', axis=1)

        return full_df.drop_duplicates(subset=['company_name', 'days_ago']).sort_values(by=['company_name', 'days_ago']).reset_index(drop=True)

    def _assign_tiers_by_rule(self, df):
        df_sorted = df.sort_values(by=['rank', 'rank_std_60d'], ascending=[True, True]).reset_index(drop=True)
        total_companies = len(df_sorted)
        tier1_count = int(total_companies * 0.025)
        tier2_count = int(total_companies * 0.05)
        tier3_count = int(total_companies * 0.10)
        df_sorted['tier'] = ''
        df_sorted.loc[:tier1_count-1, 'tier'] = 'Tier 1'
        df_sorted.loc[tier1_count:tier1_count + tier2_count-1, 'tier'] = 'Tier 2'
        df_sorted.loc[tier1_count + tier2_count : tier1_count + tier2_count + tier3_count-1, 'tier'] = 'Tier 3'
        remaining_df = df_sorted[df_sorted['tier'] == ''].copy()
        if not remaining_df.empty:
            features = ['rank', 'rank_std_60d']
            X_scaled = StandardScaler().fit_transform(remaining_df[features])
            n_clusters = 2
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            clusters = kmeans.fit_predict(X_scaled)
            remaining_df['cluster'] = clusters
            cluster_rank_means = remaining_df.groupby('cluster')['rank'].mean().sort_values()
            tier_map = {cluster_id: f'Tier {i+4}' for i, cluster_id in enumerate(cluster_rank_means.index)}
            df_sorted.loc[remaining_df.index, 'tier'] = remaining_df['cluster'].map(tier_map)
        return df_sorted

    def create_and_analyze_tiers(self):
        print("\n[단계 1] Tier 시스템 생성 및 시장 분석")
        if self.time_series_df.empty: return None, None
        
        current_data = self.time_series_df[self.time_series_df['days_ago'] == 0].copy()
        rank_stability = self.time_series_df.groupby('company_name')['rank'].std().reset_index(name='rank_std_60d').fillna(0)
        df_for_current_tiering = pd.merge(current_data, rank_stability, on='company_name', how='left').fillna(0)
        df_with_current_tiers = self._assign_tiers_by_rule(df_for_current_tiering)
        df_with_current_tiers.rename(columns={'tier': 'unified_tier'}, inplace=True)

        past_data = self.time_series_df[self.time_series_df['days_ago'] == 60].copy()
        df_for_past_tiering = pd.merge(past_data, rank_stability, on='company_name', how='left').fillna(0)
        df_with_past_tiers = self._assign_tiers_by_rule(df_for_past_tiering)
        df_with_past_tiers.rename(columns={'tier': 'unified_tier_past'}, inplace=True)
        
        final_df = pd.merge(df_with_current_tiers, df_with_past_tiers[['company_name', 'rank', 'unified_tier_past']], on='company_name', suffixes=('_current', '_past'), how='left')
        
        past_reviews = self.time_series_df[self.time_series_df['days_ago'] == 60][['company_name', 'visitor_reviews', 'blog_reviews']].set_index('company_name')
        
        def get_change(row, review_type):
            if row['company_name'] in past_reviews.index:
                return row[review_type] - past_reviews.loc[row['company_name']][review_type]
            return 0
            
        final_df['visitor_reviews_change'] = final_df.apply(get_change, review_type='visitor_reviews', axis=1)
        final_df['blog_reviews_change'] = final_df.apply(get_change, review_type='blog_reviews', axis=1)

        return final_df, df_with_current_tiers

    def analyze_market_overview(self, df_with_tiers, final_df):
        print("\n[단계 2] 시장 경쟁 현황 분석")
        
        # 컬럼명 변경
        display_df = final_df.rename(columns={
            'company_name': '병원명', 'unified_tier': '현재 Tier', 'rank_current': '현재 순위', 
            'unified_tier_past': '60일 전 Tier', 'rank_past': '60일 전 순위',
            'visitor_reviews': '방문자 리뷰 수', 'visitor_reviews_change': '방문자 리뷰 수 증가량',
            'blog_reviews': '블로그 리뷰 수', 'blog_reviews_change': '블로그 리뷰 수 증가량'
        })
        
        # 요청된 컬럼만 선택
        final_columns = ['병원명', '현재 Tier', '현재 순위', '60일 전 Tier', '60일 전 순위', '방문자 리뷰 수', '방문자 리뷰 수 증가량', '블로그 리뷰 수', '블로그 리뷰 수 증가량']
        
        # 숫자형 컬럼 변환
        for col in final_columns:
            if col not in ['병원명', '현재 Tier', '60일 전 Tier']:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce').fillna(0)
        
        # 정수형으로 변환
        display_df['방문자 리뷰 수'] = display_df['방문자 리뷰 수'].astype(int)
        display_df['블로그 리뷰 수'] = display_df['블로그 리뷰 수'].astype(int)
        display_df = display_df[final_columns].astype({'현재 순위': 'int', '60일 전 순위': 'Int64', '방문자 리뷰 수 증가량': 'Int64', '블로그 리뷰 수 증가량': 'Int64'})
        
        # Tier 변동 계산
        tier_changes = self._calculate_tier_changes(final_df)
        
        # 순위 변동성 계산 (60일간 표준편차)
        rank_volatility = final_df.groupby('unified_tier')['rank_std_60d'].mean().round(1)
        
        # 리뷰 증가량 평균 계산
        # 먼저 숫자형으로 변환
        final_df['visitor_reviews_change'] = pd.to_numeric(final_df['visitor_reviews_change'], errors='coerce').fillna(0)
        final_df['blog_reviews_change'] = pd.to_numeric(final_df['blog_reviews_change'], errors='coerce').fillna(0)
        
        review_changes = final_df.groupby('unified_tier').agg({
            'visitor_reviews_change': 'mean',
            'blog_reviews_change': 'mean'
        }).round(0)
        
        # 기본 통계 계산
        for col in ['rank', 'visitor_reviews', 'blog_reviews']:
            df_with_tiers[col] = pd.to_numeric(df_with_tiers[col], errors='coerce').fillna(0)

        tier_summary = df_with_tiers.groupby('unified_tier').agg(
            업체_수=('company_name', 'count'), 
            평균_순위=('rank', 'mean'), 
            평균_방문자_리뷰=('visitor_reviews', 'mean'), 
            평균_블로그_리뷰=('blog_reviews', 'mean')
        ).round(0)
        
        # 추가 컬럼 병합
        tier_summary['평균_순위변동성'] = rank_volatility
        tier_summary['평균_방문자리뷰_증가량'] = review_changes['visitor_reviews_change']
        tier_summary['평균_블로그리뷰_증가량'] = review_changes['blog_reviews_change']
        tier_summary = tier_summary.merge(tier_changes, left_index=True, right_index=True)
        
        # 컬럼 순서 정리
        tier_summary = tier_summary[['업체_수', '평균_순위', '평균_순위변동성', '평균_방문자_리뷰', 
                                     '평균_방문자리뷰_증가량', '평균_블로그_리뷰', '평균_블로그리뷰_증가량',
                                     'Tier_IN_개수', 'Tier_OUT_개수']]
        tier_summary = tier_summary.reset_index().sort_values(by='unified_tier')
        
        # 차트 생성 (한글화 전에 생성)
        chart_path = self._create_tier_analysis_chart(tier_summary.copy())
        
        # 컬럼명 한글화
        tier_summary.columns = ['Tier', '업체 수', '평균 순위', '평균 순위변동성', '평균 방문자리뷰', 
                               '평균 방문자리뷰 증가량', '평균 블로그리뷰', '평균 블로그리뷰 증가량',
                               '기간 내 Tier IN(상승) 개수', '기간 내 Tier OUT(하락) 개수']
        
        # Tier 요약 테이블 HTML 생성
        tier_summary_html = tier_summary.to_html(index=False, classes='table table-striped')
        
        prompt = f"""## '{TARGET_QUERY}' 네이버 플레이스 경쟁 현황 분석
        ### 분석 요청사항
        1. **Tier별 현황표 제시**: 아래 'Tier별 요약' 데이터를 마크다운 테이블 형식으로 그대로 제시해주세요.
        2. **Tier별 특징 분석**: 각 Tier를 데이터에 기반하여 직관적인 이름(예: 절대 강자 그룹)을 부여하고, 각 그룹의 특징과 전략을 설명해주세요.
        3. **Tier 변동 인사이트**: Tier IN/OUT 데이터를 바탕으로 시장의 역동성을 분석해주세요.
        4. **순위 변동성 분석**: 각 Tier별 순위 변동성의 의미와 전략적 시사점을 제시해주세요.
        ### 데이터
        [Tier별 요약]
        [Tier별 종합 분석]
        {tier_summary.to_markdown(index=False)}
        """
        response = self.agent.run(prompt, max_tokens=16384)
        return {
            "title": f"{TARGET_QUERY} 플레이스 경쟁 현황 (기준 일자: {ANALYSIS_DATE})", 
            "content": response.content if response else "LLM 분석 실패", 
            "full_data_table": display_df.to_html(index=False, classes='table table-striped'),
            "tier_summary_table": tier_summary_html,
            "chart_filename": chart_path
        }

    def analyze_success_factors(self, final_df):
        print("\n[단계 3] 순위 상승 핵심 요인 분석")
        final_df['rank_change'] = pd.to_numeric(final_df['rank_past'], errors='coerce').fillna(0) - pd.to_numeric(final_df['rank_current'], errors='coerce').fillna(0)
        climbers = final_df[final_df['rank_change'] > 10].sort_values(by='rank_change', ascending=False).head(10)
        prompt = f"""## 순위 상승의 핵심 요인 분석 (성공 방정식)
        ### 분석 요청사항
        '{TARGET_QUERY}' 시장에서 최근 60일간 순위가 10계단 이상 급등한 '도전자' 그룹의 성공 요인을 분석하고, 이를 바탕으로 '성공 방정식'을 정의해주세요.
        ### 데이터 (순위 급상승 업체)
        {climbers.to_markdown(index=False)}
        """
        response = self.agent.run(prompt, max_tokens=16384)
        return {"title": "순위 상승의 핵심 요인 분석", "content": response.content if response else "LLM 분석 실패"}

    def analyze_target_clinic(self, df_with_tiers):
        print(f"\n[단계 4] '{TARGET_CLINIC}' 맞춤 성장 전략")
        target_info = df_with_tiers[df_with_tiers['company_name'] == TARGET_CLINIC]
        if target_info.empty: return None
        target_tier = target_info['unified_tier'].iloc[0]
        target_rank = target_info['rank'].iloc[0]
        upper_tier_num = int(target_tier.split(' ')[1]) - 1
        if upper_tier_num < 1: upper_tier_num = 1
        upper_tier = f"Tier {upper_tier_num}"
        target_tier_avg = df_with_tiers[df_with_tiers['unified_tier'] == target_tier].mean(numeric_only=True)
        upper_tier_avg = df_with_tiers[df_with_tiers['unified_tier'] == upper_tier].mean(numeric_only=True)
        prompt = f"""## '{TARGET_CLINIC}' 맞춤 성장 전략 제안
        ### 분석 요청사항
        '{TARGET_CLINIC}'의 현재 데이터와 목표 Tier와의 격차를 바탕으로, '리뷰부스터'와 '메디컨텐츠' 서비스 도입을 제안하는 구체적인 액션 플랜을 제시해주세요.
        ### 데이터
        - 현재 소속: {target_tier} (현재 순위: {target_rank}위)
        - 현재 Tier 평균: 방문자 리뷰 {target_tier_avg['visitor_reviews']:.0f}개, 블로그 리뷰 {target_tier_avg['blog_reviews']:.0f}개
        - 목표 Tier ({upper_tier}) 평균: 방문자 리뷰 {upper_tier_avg['visitor_reviews']:.0f}개, 블로그 리뷰 {upper_tier_avg['blog_reviews']:.0f}개
        """
        response = self.agent.run(prompt, max_tokens=16384)
        return {"title": f"'{TARGET_CLINIC}' 맞춤 성장 전략", "content": response.content if response else "LLM 분석 실패"}

    def _calculate_tier_changes(self, final_df):
        """Tier 변동 (IN/OUT) 계산"""
        tier_in_out = {}
        for tier in final_df['unified_tier'].unique():
            # 현재 Tier에 속한 업체들
            current_tier_companies = set(final_df[final_df['unified_tier'] == tier]['company_name'])
            # 60일 전 같은 Tier에 속했던 업체들
            past_tier_companies = set(final_df[final_df['unified_tier_past'] == tier]['company_name'])
            
            # Tier IN: 60일 전엔 이 Tier가 아니었지만 현재는 이 Tier인 업체
            tier_in = len(current_tier_companies - past_tier_companies)
            # Tier OUT: 60일 전엔 이 Tier였지만 현재는 아닌 업체
            tier_out = len(past_tier_companies - current_tier_companies)
            
            tier_in_out[tier] = {'Tier_IN_개수': tier_in, 'Tier_OUT_개수': tier_out}
        
        return pd.DataFrame(tier_in_out).T
    
    def _create_tier_analysis_chart(self, tier_summary):
        """Tier 분석 차트 생성"""
        import matplotlib
        matplotlib.use('Agg')
        
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # X축 설정
        x = range(len(tier_summary))
        tiers = tier_summary['unified_tier'].values
        
        # 막대 그래프 (방문자/블로그 리뷰)
        width = 0.35
        ax1.bar([i - width/2 for i in x], tier_summary['평균_방문자_리뷰'], width, 
                label='평균 방문자리뷰', color='#3498db', alpha=0.8)
        ax1.bar([i + width/2 for i in x], tier_summary['평균_블로그_리뷰'], width, 
                label='평균 블로그리뷰', color='#2ecc71', alpha=0.8)
        
        ax1.set_xlabel('Tier', fontsize=12)
        ax1.set_ylabel('리뷰 수', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(tiers)
        ax1.legend(loc='upper left')
        
        # 두 번째 Y축 (순위)
        ax2 = ax1.twinx()
        ax2.plot(x, tier_summary['평균_순위'], 'ro-', linewidth=2, markersize=8, label='평균 순위')
        ax2.set_ylabel('순위', fontsize=12)
        ax2.set_ylim(0, 200)  # Y축 범위를 0~200으로 설정 (아래가 0, 위가 200)
        
        # 세 번째 Y축 (순위 변동성)
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(x, tier_summary['평균_순위변동성'], 'go--', linewidth=2, markersize=8, label='평균 순위변동성')
        ax3.set_ylabel('순위 변동성', fontsize=12)
        
        # 범례 통합
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper center', 
                   bbox_to_anchor=(0.5, -0.1), ncol=5)
        
        plt.title('Tier별 종합 분석', fontsize=16, pad=20)
        plt.tight_layout()
        
        # 차트 저장
        chart_filename = 'tier_analysis_chart_v0.04.png'
        chart_path = os.path.join(self.result_path, chart_filename)
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_filename
    
    def generate_html_report_v0_04(self, sections):
        print("\nHTML 보고서 생성 (v0.04)...")
        introduction_html = """
        <h2>1. 분석의 배경 및 목적</h2>
        <h3>네이버 플레이스, 병원 마케팅의 가장 중요한 전쟁터</h3>
        <p>대한민국에서 잠재 환자들이 병원을 찾는 주된 경로는 '네이버 검색'입니다. 네이버에 '""" + TARGET_QUERY + """'와 같이 지역명과 진료 분야를 검색하면, 네이버 지도와 연동된 '네이버 플레이스' 서비스가 검색 목록을 보여줍니다. 이 목록의 상위에 위치하는 것은 신규 환자 유입에 지대한 영향을 미치므로, 모든 병원은 이 순위 경쟁에 많은 노력을 기울이고 있습니다.</p>
        <p>네이버는 공식적으로 순위 결정에 <strong>유사도, 인기도, 거리, 정보의 충실성</strong> 등 복합적인 요소가 작용한다고 밝히고 있습니다. 특히 최근 알고리즘은 실제 방문 경험이 인증된 <strong>'방문자 리뷰'</strong>의 중요성을 높이고, 사용자의 <strong>'저장하기', '공유하기'</strong>와 같은 실제 행동 데이터를 더욱 중시하는 방향으로 변화하고 있습니다. 이는 단순한 마케팅 물량 공세가 아닌, 실제 고객 만족도에 기반한 '진정성' 있는 운영이 순위 상승의 핵심이 되었음을 의미합니다.</p>
        <h3>이 보고서의 목적</h3>
        <p>본 보고서는 네이버 플레이스 시장의 경쟁 구도를 심층적으로 분석하여, 귀사(""" + TARGET_CLINIC + """)의 현재 위치를 객관적으로 진단하고, 플레이스 순위 상승을 위한 최적의 맞춤 전략을 제안하는 것을 목표로 합니다.</p>
        <p>궁극적으로, 이 분석 결과를 바탕으로 실제 방문 고객의 리뷰를 효과적으로 확보하는 '리뷰부스터' 서비스와, 온라인상의 정보 경쟁력을 강화하는 전문적인 '메디컨텐츠' 서비스 도입이 왜 필요한지, 그리고 어떻게 귀사의 성장에 기여할 수 있는지를 명확하게 제시하고자 합니다.</p>
        <p>(네이버 지도에서 '""" + TARGET_QUERY + """'를 검색하였을 때의 순위를 기준으로 분석되었으며, 기준일은 """ + ANALYSIS_DATE + """입니다.)</p>
        """
        content_html = ""
        market_overview = sections.get('market_overview', {})
        if market_overview:
            content_html += f"<h2>2. {market_overview.get('title', '')}</h2>"
            # 전체 업체 리스트를 먼저 표시
            content_html += "<h3>전체 업체 리스트 및 통합 Tier 현황</h3>"
            content_html += market_overview.get('full_data_table', '')
            
            content_html += f"<h3>1. Tier 종합 분석(기간: 60일)</h3>"
            # Tier 요약 테이블 추가
            if 'tier_summary_table' in market_overview:
                content_html += market_overview['tier_summary_table']
            # 차트 이미지 추가
            if 'chart_filename' in market_overview:
                content_html += f'<div style="text-align: center; margin: 20px 0;">'
                content_html += f'<img src="{market_overview["chart_filename"]}" alt="Tier 종합 분석 차트" style="max-width: 100%; height: auto;">'
                content_html += f'</div>'
            content_html += self.markdown_converter.convert(market_overview.get('content', ''))
        
        success_factors = sections.get('success_factors', {})
        if success_factors:
            content_html += f"<h2>3. {success_factors.get('title', '')}</h2>"
            content_html += self.markdown_converter.convert(success_factors.get('content', ''))

        target_strategy = sections.get('target_strategy', {})
        if target_strategy:
            content_html += f"<h2>4. {target_strategy.get('title', '')}</h2>"
            content_html += self.markdown_converter.convert(target_strategy.get('content', ''))

        html_template = f"""
        <!DOCTYPE html><html><head><meta charset="UTF-8"><title>'{TARGET_QUERY}' 네이버 플레이스 순위 분석 보고서 v0.04</title>
        <style>body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;line-height:1.6;padding:20px;max-width:1000px;margin:auto;color:#333;}}h1,h2,h3{{color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:10px;}}table{{border-collapse:collapse;width:100%;margin:20px 0;box-shadow:0 2px 3px rgba(0,0,0,0.1);}}th,td{{border:1px solid #ddd;padding:12px;text-align:left;}}th{{background-color:#3498db;color:white;}}tr:nth-child(even){{background-color:#f2f9fd;}}</style>
        </head><body><h1>[V0.04] '{TARGET_QUERY}' 네이버 플레이스 경쟁력 분석 및 성장 전략 제안</h1>
        {introduction_html}
        {content_html}
        </body></html>
        """
        with open(REPORT_OUTPUT_PATH_HTML, 'w', encoding='utf-8') as f:
            f.write(html_template)
        print(f"HTML 보고서가 다음 경로에 저장되었습니다: {REPORT_OUTPUT_PATH_HTML}")


def main():
    analyzer = NaverPlaceAnalyzerV0_04(data_dir=RAW_DATA_DIR, result_path=RESULT_PATH)
    if analyzer.time_series_df.empty: return
    final_df, df_with_current_tiers = analyzer.create_and_analyze_tiers()
    if final_df is None: return
    
    sections = {
        'market_overview': analyzer.analyze_market_overview(df_with_current_tiers, final_df),
        'success_factors': analyzer.analyze_success_factors(final_df),
        'target_strategy': analyzer.analyze_target_clinic(df_with_current_tiers)
    }
    analyzer.generate_html_report_v0_04(sections)

if __name__ == "__main__":
    main()
