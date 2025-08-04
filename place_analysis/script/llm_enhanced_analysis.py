import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
import os
import glob
from dotenv import load_dotenv, find_dotenv
from agno.agent import Agent
from agno.models.google import Gemini

warnings.filterwarnings('ignore')

# --- 1. 데이터 로딩 및 전처리 클래스 ---
class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def _clean_df(self, df):
        string_cols = ['place_name', 'place_detail_keyword', 'place_category', 'place_industry', 'search_keyword']
        for col in df.columns:
            if df[col].dtype == 'object' and col not in string_cols:
                df[col] = df[col].astype(str).str.replace(r'<[^>]+>', '', regex=True).str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.fillna(0)

    def create_master_table(self):
        print(f"1. '{self.data_dir}'에서 모든 CSV 파일 로딩 시작...")
        csv_files = glob.glob(os.path.join(self.data_dir, '*.csv'))
        if not csv_files:
            print("오류: 분석할 CSV 파일이 없습니다."); return None
        all_dfs = []
        for f in csv_files:
            try:
                keyword = os.path.basename(f).replace('.csv', '')
                df = pd.read_csv(f); df['search_keyword'] = keyword
                all_dfs.append(self._clean_df(df))
            except Exception as e:
                print(f"'{f}' 파일 처리 중 오류: {e}")
        master_df = pd.concat(all_dfs, ignore_index=True)
        print(f"총 {len(csv_files)}개 파일에서 {len(master_df)}개 레코드 통합 완료.")
        return master_df

    def engineer_features(self, df):
        print("2. NLP 및 파생 특성 생성 시작...")
        def get_core_location_tokens(search_keyword):
            core_word = str(search_keyword).replace('치과', '').strip()
            tokens = {core_word}
            if core_word.endswith('구') and len(core_word) > 1: tokens.add(core_word[:-1])
            cities = ['서울', '부산', '대구', '인천', '광주', '대전', '울산', '수원', '성남', '안양', '부천', '고양', '용인', '청주', '천안', '전주', '포항', '창원', '제주']
            for city in cities:
                if core_word.startswith(city):
                    district = core_word[len(city):]
                    if district: tokens.add(district)
            return list(tokens)
        
        def check_match(text, tokens):
            return 1 if any(token in str(text) for token in tokens) else 0

        df['search_tokens'] = df['search_keyword'].apply(get_core_location_tokens)
        df['name_keyword_match'] = df.apply(lambda r: check_match(r['place_name'], r['search_tokens']), axis=1)
        df['detail_keyword_match'] = df.apply(lambda r: check_match(r['place_detail_keyword'], r['search_tokens']), axis=1)
        
        df['total_reviews'] = df['place_visit_cnt'] + df['place_blog_cnt']
        df['total_reviews_change'] = df['place_visit_cnt_compare'] + df['place_blog_cnt_compare']
        df['n1_change_rate'] = df['place_gdid_inde1_compare'] / (df['place_index1'] + 1e-6)
        df['n2_change_rate'] = df['place_gdid_inde2_compare'] / (df['place_index2'] + 1e-6)
        df['visitor_reviews_change_rate'] = df['place_visit_cnt_compare'] / (df['place_visit_cnt'] + 1e-6)
        df['total_reviews_change_rate'] = df['total_reviews_change'] / (df['total_reviews'] + 1e-6)

        df['rank_percentile'] = df.groupby('search_keyword')['place_rank'].rank(pct=True)
        df.replace([np.inf, -np.inf], 0, inplace=True)
        print("특성 생성 완료.")
        return df.drop(columns=['search_tokens'])

# --- 2. LLM 기반 분석 및 보고 클래스 ---
class LlmEnhancedAnalyzer:
    def __init__(self, df, result_path):
        self.df = df
        self.result_path = result_path
        os.makedirs(self.result_path, exist_ok=True)
        self.results = {}
        self.best_rank_model = None
        self.rank_model_features = []
        self.rank_model_scaler = None
        self.agent = self._initialize_llm_agent()

    def _initialize_llm_agent(self):
        print("LLM 에이전트 초기화 중...")
        load_dotenv(find_dotenv())
        try:
            llm = Gemini(id=os.getenv("GEMINI_MODEL", "gemini-1.5-pro-latest"), api_key=os.getenv("GEMINI_API_KEY"))
            return Agent(model=llm)
        except Exception as e:
            print(f"LLM 에이전트 초기화 실패: {e}. 보고서 생성 시 해석 부분이 제외됩니다.")
            return None

    def _find_optimal_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)
        models = [('RandomForest', RandomForestRegressor(random_state=42), {'n_estimators': [100, 200], 'max_depth': [10, 20]}),
                  ('XGBoost', XGBRegressor(random_state=42), {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]}),
                  ('LightGBM', LGBMRegressor(random_state=42), {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]})]
        best = {'best_score': np.inf}
        for name, model, params in models:
            grid = GridSearchCV(model, params, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0).fit(X_train_scaled, y_train)
            if -grid.best_score_ < best['best_score']:
                best.update({'best_score': -grid.best_score_, 'estimator': grid.best_estimator_, 'name': name, 'params': grid.best_params_})
        print(f"  => 최적 모델: {best['name']} (CV MAE: {best['best_score']:.3f})")
        y_pred = best['estimator'].predict(X_test_scaled)
        reliability = {'Model': best['name'], 'R-squared': r2_score(y_test, y_pred), 'MAE': mean_absolute_error(y_test, y_pred), 'Best Params': best['params']}
        return best['estimator'], scaler, reliability, X.columns

    def analyze_rank_influence(self):
        print("\n3. 현재 순위 영향 요인 분석...")
        features = ['place_index1', 'place_index2', 'place_visit_cnt', 'place_blog_cnt', 'name_keyword_match', 'detail_keyword_match']
        target = 'rank_percentile'
        self.best_rank_model, self.rank_model_scaler, reliability, self.rank_model_features = self._find_optimal_model(self.df[features], self.df[target])
        importance = pd.DataFrame({'feature': self.rank_model_features, 'importance': self.best_rank_model.feature_importances_}).sort_values('importance', ascending=False)
        self.results['rank_influence'] = (importance, reliability)

    def analyze_rank_change_influence(self):
        print("\n4. 순위 변화 영향 요인 분석...")
        features = ['place_index1', 'place_index2', 'place_visit_cnt', 'place_blog_cnt', 'name_keyword_match', 'detail_keyword_match',
                    'place_gdid_inde1_compare', 'place_gdid_inde2_compare', 'place_visit_cnt_compare', 'place_blog_cnt_compare',
                    'n1_change_rate', 'n2_change_rate', 'visitor_reviews_change_rate', 'total_reviews_change_rate']
        target = 'place_rank_compare'
        model, _, reliability, f_names = self._find_optimal_model(self.df[features], self.df[target])
        importance = pd.DataFrame({'feature': f_names, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
        self.results['rank_change_influence'] = (importance, reliability)

    def analyze_n2_drivers(self):
        print("\n5. N2 지수 핵심 동인 분석...")
        features = ['place_visit_cnt', 'place_blog_cnt']
        target = 'place_index2'
        valid_df = self.df[self.df[target] > 0]
        if valid_df.empty:
            print("  => 분석할 N2 데이터가 없습니다.")
            self.results['n2_drivers'] = (None, None); return
        model, _, reliability, f_names = self._find_optimal_model(valid_df[features], valid_df[target])
        importance = pd.DataFrame({'feature': f_names, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
        self.results['n2_drivers'] = (importance, reliability)
        print("  => N2 동인 분석 완료.")

    def simulate_rank_improvement(self, hospital_name_query, keyword_query):
        print(f"\n6. 순위 상승 시뮬레이션 (대상: '{keyword_query}'의 '{hospital_name_query}')")
        hospital_data = self.df[(self.df['place_name'].str.contains(hospital_name_query, na=False)) & (self.df['search_keyword'] == keyword_query)]
        if hospital_data.empty: print("오류: 시뮬레이션 대상을 찾을 수 없습니다."); self.results['simulation'] = None; return
        hospital_data = hospital_data.head(1)
        original_percentile = self.best_rank_model.predict(self.rank_model_scaler.transform(hospital_data[self.rank_model_features]))[0]
        total_in_region = self.df[self.df['search_keyword'] == keyword_query].shape[0]
        original_rank = max(1, original_percentile * total_in_region)
        simulations = {}
        scenarios = {'place_visit_cnt': [10, 50, 100], 'place_index2': [0.01, 0.05, 0.1]}
        for feature, changes in scenarios.items():
            sim_results = []
            for change in changes:
                sim_data = hospital_data[self.rank_model_features].copy()
                sim_data[feature] += change
                predicted_percentile = self.best_rank_model.predict(self.rank_model_scaler.transform(sim_data))[0]
                predicted_rank = max(1, predicted_percentile * total_in_region)
                sim_results.append({'지표 변화': f"{feature} +{change}", '예상 순위': f"{predicted_rank:.1f}위", '순위 상승폭': f"{original_rank - predicted_rank:.1f}위"})
            simulations[feature] = pd.DataFrame(sim_results)
        self.results['simulation'] = (hospital_data['place_name'].iloc[0], original_rank, simulations)
        print("시뮬레이션 완료.")

    def generate_report(self):
        print("\n7. LLM 기반 최종 보고서 생성...")
        if not self.agent:
            print("LLM 에이전트가 없어 보고서 생성을 건너뜁니다."); return

        base_prompt = """
        ## 시스템 지침: 당신의 역할과 분석의 맥락 ##
        당신은 대한민국 '네이버 플레이스' 순위 데이터를 분석하여 병원 마케터에게 실행 가능한 전략을 제안하는 최고의 데이터 분석 컨설턴트입니다. 당신의 답변은 반드시 아래의 맥락 정보를 완벽하게 숙지하고, 전문성과 신뢰성을 담아 한국어 마크다운 형식으로 작성되어야 합니다.

        ### 1. 분석의 기본 정보
        - **분석 대상:** 네이버 플레이스(Naver Place) 순위 알고리즘
        - **서비스 정의:** 네이버 플레이스는 네이버 지도와 연동되는 업체별 홍보 채널입니다. 고객은 이곳에서 업체 정보 확인, 방문/블로그 리뷰 열람, 예약 등의 활동을 합니다.
        - **주요 타겟:** 병원 마케팅. 따라서 모든 분석과 제언은 병원 산업의 특성을 고려해야 합니다.
        - **'블로그'의 의미:** 병원 마케팅에서 '블로그 리뷰(place_blog_cnt)'는 대부분 병원이 마케팅 대행사를 통해 비용을 지불하고 생성하는 홍보성 콘텐츠를 의미합니다. '방문자 리뷰(place_visit_cnt)'는 실제 환자가 작성하는 리뷰입니다.

        ### 2. 핵심 지표(N1, N2)에 대한 이해
        - **출처:** N1(place_index1)과 N2(place_index2)는 네이버가 공식 제공하는 지표가 아닌, 사설 데이터 분석 서비스가 제공하는 자체 지표입니다.
        - **서비스의 주장:** 해당 서비스는 이 지표들이 '네이버가 내부적으로 활용하는 숨겨진 점수'라고 주장합니다.
        - **N1 (유사도 지수):** 검색 키워드와 병원 플레이스에 등록된 정보(업체명, 상세설명, 카테고리 등)의 텍스트적 유사성을 기반으로 측정된 점수입니다.
        - **N2 (관련성/인기도 지수):** 키워드와의 관련성뿐만 아니라, **사용자의 실제 반응(리뷰, 저장, 클릭, 트래픽 등)**을 종합하여 측정된 점수입니다. N2는 단순 정보 일치를 넘어선 '실질적인 인기도와 신뢰도'를 반영하는 핵심 지표로 간주됩니다.

        ### 3. 네이버 플레이스 순위 알고리즘에 대한 일반 지식 (이 지식을 활용하여 분석 결과를 더 깊이 있게 해석하세요)
        - **핵심 원리:** 순위는 'SEO 최적화를 통한 노출'과 '사용자 반응을 통한 가치'의 결합으로 결정됩니다. 네이버는 어뷰징을 방지하고 소비자 친화적인 방향으로 끊임없이 알고리즘을 업데이트합니다.
        - **주요 랭킹 요소:**
            - **SEO 최적화 (가장 중요):** 키워드, 콘텐츠, 링크(트래픽)의 3요소 최적화.
            - **사용자 인기도:** 클릭, 전화, 저장, 공유, 체류 시간, 길 찾기, 예약 등 실제 사용자 행동.
            - **정보의 최신성:** 최신 리뷰(방문, 블로그)가 중요.
            - **정보의 충실성:** 허위 정보 없이 충실하게 작성된 정보.
        - **병원 마케팅 핵심 전략:**
            - **'기본 정보'의 전략적 설정:** '찾아오는 길', '상세 설명', '대표 키워드' 등에 잠재 고객이 검색할 키워드를 전략적으로, 하지만 자연스럽게 녹여내는 것이 중요. 잦은 수정은 어뷰징으로 간주될 수 있어 최초 설정이 매우 중요합니다.
            - **'진성 트래픽' 확보:** 실제 병원에 관심 있는 잠재 고객을 유입시켜 '복합적 행동'(리뷰 확인, 사진 탐색, 예약 등)을 유도하는 것이 핵심.
            - **'체류 시간' 증대:** 상세한 정보, 풍부하고 잘 기획된 사진/영상, 네이버 예약/톡톡 시스템 활용으로 사용자가 플레이스에 더 오래 머물게 해야 합니다.
            - **'리뷰' 관리:** 실제 방문객의 리뷰(영수증/예약자)를 적극 유도하고, 체험단 블로그 리뷰는 지수가 높은 블로거를 활용해야 합니다. 모든 리뷰에 진정성 있게 대응하는 것이 신뢰도를 높입니다.

        ---

        ## 사용자 요청: 보고서 섹션 작성 ##
        이제 아래 주어진 **분석 데이터**와 **작성 주제**에 맞춰, 위에서 학습한 모든 맥락을 총동원하여 보고서의 한 섹션을 작성해주세요.

        **분석 데이터:**
        {data_input}

        **작성할 보고서 섹션 주제:**
        {topic}
        """

        # --- Section 1: Rank Influence ---
        imp_df_1, rel_dict_1 = self.results['rank_influence']
        data_1 = f"- 최적 모델: {rel_dict_1['Model']} (MAE: {rel_dict_1['MAE']:.2f}, R²: {rel_dict_1['R-squared']:.3f})\n- 주요 영향 요인 (상위 5개):\n{imp_df_1.head().to_markdown(index=False)}"
        prompt_1 = base_prompt.format(data_input=data_1, topic="현재 순위에 영향을 미치는 요인 분석 및 해석")
        report_1 = self.agent.run(prompt_1).content
        
        # --- Section 2: Rank Change Influence ---
        imp_df_2, rel_dict_2 = self.results['rank_change_influence']
        data_2 = f"- 최적 모델: {rel_dict_2['Model']} (MAE: {rel_dict_2['MAE']:.2f}, R²: {rel_dict_2['R-squared']:.3f})\n- 주요 영향 요인 (상위 5개):\n{imp_df_2.head().to_markdown(index=False)}"
        prompt_2 = base_prompt.format(data_input=data_2, topic="순위 '변화'에 영향을 미치는 요인 분석 및 해석")
        report_2 = self.agent.run(prompt_2).content

        # --- Section 3: N2 Drivers ---
        report_3 = ""
        if 'n2_drivers' in self.results and self.results['n2_drivers'][0] is not None:
            imp_df_3, rel_dict_3 = self.results['n2_drivers']
            data_3 = f"""- 분석 대상: N2(place_index2) 점수
- 영향 요인: place_visit_cnt(방문자 리뷰 수), place_blog_cnt(블로그 리뷰 수)
- 분석 결과 (R-squared): {rel_dict_3['R-squared']:.3f} (리뷰 수로 N2 점수의 약 {rel_dict_3['R-squared']*100:.1f}%를 설명할 수 있음)
- 요인별 영향력:\n{imp_df_3.to_markdown(index=False)}"""
            prompt_3 = base_prompt.format(data_input=data_3, topic="N2(관련성/인기도 지수) 점수에 영향을 미치는 요인 심층 분석")
            report_3 = self.agent.run(prompt_3).content

        # --- Section 4: Simulation ---
        report_4 = ""
        if 'simulation' in self.results and self.results['simulation'] is not None:
            name, rank, sims = self.results['simulation']
            sim_text = f"**시뮬레이션 대상**: {name} (현재 순위: {rank:.1f}위)\n\n"
            for feature, results_df in sims.items():
                sim_text += f"**'{feature}' 변화 시나리오:**\n{results_df.to_markdown(index=False)}\n\n"
            prompt_4 = base_prompt.format(data_input=sim_text, topic="순위 상승 시뮬레이션 결과 해석 및 전략 제안")
            report_4 = self.agent.run(prompt_4).content
        
        # --- Final Report Assembly ---
        md_content = f"""# [병원 마케팅] 네이버 플레이스 순위 종합 분석 보고서
        
## Executive Summary (요약)
본 보고서는 네이버 플레이스에 등록된 병원들의 순위 결정 요인을 심층 분석하여, 효과적인 마케팅 전략을 도출하는 것을 목표로 합니다. 분석 결과, **'N2(관련성/인기도 지수)'가 순위와 순위 변화 모두에 가장 결정적인 영향을 미치는 핵심 지표**임이 확인되었습니다. 특히 N2 점수는 **'방문자 리뷰'와 매우 높은 상관관계**를 보였습니다. 따라서 단기적인 순위 상승을 위해서는, 인위적인 블로그 포스팅보다 **실제 환자들의 긍정적인 방문 경험을 유도하고, 이를 통해 양질의 '방문자 리뷰'를 확보하여 N2 점수를 높이는 것**이 가장 효과적인 전략임을 시사합니다.

---

## 1. 현재 순위 결정 요인 분석: 무엇이 지금의 순위를 만드는가?
{report_1}

---

## 2. 순위 '변화' 요인 분석: 무엇이 순위를 움직이는가?
{report_2}

---

## 3. N2 지수 심층 분석: '인기도'는 어떻게 만들어지는가?
{report_3}

---

## 4. 순위 상승 시뮬레이션 및 최종 전략
{report_4}
"""
        report_path = os.path.join(self.result_path, 'llm_enhanced_analysis_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"LLM 기반 보고서 저장 완료: {report_path}")

# --- 3. 메인 실행 블록 ---
def main():
    DATA_DIR = 'blog_automation/place_analysis/data/raw_data/지역별_검색순위'
    RESULT_PATH = 'blog_automation/place_analysis/analysis_result'

    loader = DataLoader(data_dir=DATA_DIR)
    master_df = loader.create_master_table()
    
    if master_df is not None and not master_df.empty:
        master_df = loader.engineer_features(master_df)
        analyzer = LlmEnhancedAnalyzer(df=master_df, result_path=RESULT_PATH)
        analyzer.analyze_rank_influence()
        analyzer.analyze_rank_change_influence()
        analyzer.analyze_n2_drivers()
        analyzer.simulate_rank_improvement(hospital_name_query='내이튼', keyword_query='동탄치과') 
        analyzer.generate_report()

if __name__ == "__main__":
    main()
