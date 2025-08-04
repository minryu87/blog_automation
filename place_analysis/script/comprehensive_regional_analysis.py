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

warnings.filterwarnings('ignore')

# --- 1. 데이터 로딩 및 전처리 클래스 ---
class DataLoader:
    """
    여러 지역의 CSV 파일을 불러오고, 통합하고, NLP 특성을 포함한 전처리를 수행합니다.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def _clean_df(self, df):
        """개별 데이터프레임을 정리하는 내부 함수"""
        string_cols = ['place_name', 'place_detail_keyword', 'place_category', 'place_industry']
        for col in df.columns:
            if df[col].dtype == 'object' and col not in string_cols:
                df[col] = df[col].astype(str).str.replace(r'<[^>]+>', '', regex=True).str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.fillna(0)

    def create_master_table(self):
        """폴더 내 모든 CSV를 읽어 하나의 마스터 데이터프레임으로 만듭니다."""
        print(f"1. '{self.data_dir}'에서 모든 CSV 파일 로딩 시작...")
        csv_files = glob.glob(os.path.join(self.data_dir, '*.csv'))
        if not csv_files:
            print("오류: 분석할 CSV 파일이 없습니다.")
            return None

        all_dfs = []
        for f in csv_files:
            try:
                keyword = os.path.basename(f).replace('.csv', '')
                df = pd.read_csv(f)
                df = self._clean_df(df)
                df['search_keyword'] = keyword
                all_dfs.append(df)
            except Exception as e:
                print(f"'{f}' 파일 처리 중 오류 발생: {e}")

        master_df = pd.concat(all_dfs, ignore_index=True)
        print(f"총 {len(csv_files)}개 파일에서 {len(master_df)}개 레코드 통합 완료.")
        return master_df

    def engineer_features(self, df):
        """NLP 및 파생 특성을 생성합니다."""
        print("2. NLP 및 파생 특성 생성 시작...")
        
        def get_core_location_tokens(search_keyword):
            core_word = search_keyword.replace('치과', '').strip()
            tokens = {core_word}
            if core_word.endswith('구') and len(core_word) > 1:
                tokens.add(core_word[:-1])
            cities = ['서울', '부산', '대구', '인천', '광주', '대전', '울산', '수원', '성남', '안양', '부천', '고양', '용인', '청주', '천안', '전주', '포항', '창원', '제주']
            for city in cities:
                if core_word.startswith(city):
                    district = core_word[len(city):]
                    if district: tokens.add(district)
            return list(tokens)

        def check_match(text, tokens):
            text = str(text)
            return 1 if any(token in text for token in tokens) else 0

        df['search_tokens'] = df['search_keyword'].apply(get_core_location_tokens)
        df['name_keyword_match'] = df.apply(lambda row: check_match(row['place_name'], row['search_tokens']), axis=1)
        df['detail_keyword_match'] = df.apply(lambda row: check_match(row['place_detail_keyword'], row['search_tokens']), axis=1)

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


# --- 2. 머신러닝 분석 및 시뮬레이션 클래스 ---
class RegionalRankAnalyzer:
    def __init__(self, df, result_path):
        self.df = df
        self.result_path = result_path
        os.makedirs(self.result_path, exist_ok=True)
        self.results = {}
        self.best_rank_model = None
        self.rank_model_features = []
        self.rank_model_scaler = None

    def _find_optimal_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models_and_params = [
            ('RandomForest', RandomForestRegressor(random_state=42), {'n_estimators': [100, 200], 'max_depth': [10, 20]}),
            ('XGBoost', XGBRegressor(random_state=42), {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]}),
            ('LightGBM', LGBMRegressor(random_state=42), {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]})
        ]
        best_model_info = {'best_score': np.inf, 'best_estimator': None, 'model_name': ''}

        for name, model, params in models_and_params:
            print(f"  - {name} 모델 최적화 중...")
            grid_search = GridSearchCV(model, params, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0)
            grid_search.fit(X_train_scaled, y_train)
            if -grid_search.best_score_ < best_model_info['best_score']:
                best_model_info.update({
                    'best_score': -grid_search.best_score_,
                    'best_estimator': grid_search.best_estimator_,
                    'model_name': name,
                    'best_params': grid_search.best_params_
                })
        
        print(f"  => 최적 모델: {best_model_info['model_name']} (CV MAE: {best_model_info['best_score']:.3f})")
        best_model = best_model_info['best_estimator']
        y_pred = best_model.predict(X_test_scaled)
        
        reliability = {
            'Model': best_model_info['model_name'],
            'R-squared': r2_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'Best Params': best_model_info['best_params']
        }
        return best_model, scaler, reliability, X.columns

    def analyze_rank_influence(self):
        print("\n3. 현재 순위 영향 요인 분석 (모델 최적화 중...)")
        features = ['place_index1', 'place_index2', 'place_visit_cnt', 'place_blog_cnt', 'name_keyword_match', 'detail_keyword_match']
        target = 'rank_percentile'
        self.best_rank_model, self.rank_model_scaler, reliability, self.rank_model_features = self._find_optimal_model(self.df[features], self.df[target])
        importance = pd.DataFrame({'feature': self.rank_model_features, 'importance': self.best_rank_model.feature_importances_}).sort_values(by='importance', ascending=False)
        self.results['rank_influence'] = (importance, reliability)
        print("분석 완료.")

    def analyze_rank_change_influence(self):
        print("\n4. 순위 변화 영향 요인 분석 (모델 최적화 중...)")
        features = [
            'place_index1', 'place_index2', 'place_visit_cnt', 'place_blog_cnt', 'name_keyword_match', 'detail_keyword_match',
            'place_gdid_inde1_compare', 'place_gdid_inde2_compare', 'place_visit_cnt_compare', 'place_blog_cnt_compare',
            'n1_change_rate', 'n2_change_rate', 'visitor_reviews_change_rate', 'total_reviews_change_rate'
        ]
        target = 'place_rank_compare'
        best_model, _, reliability, feature_names = self._find_optimal_model(self.df[features], self.df[target])
        importance = pd.DataFrame({'feature': feature_names, 'importance': best_model.feature_importances_}).sort_values(by='importance', ascending=False)
        self.results['rank_change_influence'] = (importance, reliability)
        print("분석 완료.")

    def analyze_n2_drivers(self):
        """추가 분석: N2 점수 및 변화량의 동인 분석"""
        print("\n5. [심층 분석] N2(관련성 지수) 동인 분석...")
        # 분석 A: 현재 N2 점수
        features_n2 = ['place_visit_cnt', 'place_blog_cnt', 'total_reviews']
        target_n2 = 'place_index2'
        best_model_n2, _, reliability_n2, feature_names_n2 = self._find_optimal_model(self.df[features_n2], self.df[target_n2])
        importance_n2 = pd.DataFrame({'feature': feature_names_n2, 'importance': best_model_n2.feature_importances_}).sort_values(by='importance', ascending=False)
        self.results['n2_drivers'] = (importance_n2, reliability_n2)

        # 분석 B: N2 변화량
        features_n2_change = ['place_visit_cnt_compare', 'place_blog_cnt_compare', 'total_reviews_change', 'visitor_reviews_change_rate', 'total_reviews_change_rate']
        target_n2_change = 'place_gdid_inde2_compare'
        best_model_n2_change, _, reliability_n2_change, feature_names_n2_change = self._find_optimal_model(self.df[features_n2_change], self.df[target_n2_change])
        importance_n2_change = pd.DataFrame({'feature': feature_names_n2_change, 'importance': best_model_n2_change.feature_importances_}).sort_values(by='importance', ascending=False)
        self.results['n2_change_drivers'] = (importance_n2_change, reliability_n2_change)
        print("분석 완료.")

    def simulate_rank_improvement(self, hospital_name_query, keyword_query):
        print(f"\n6. 순위 상승 시뮬레이션 (대상: '{keyword_query}'의 '{hospital_name_query}')")
        hospital_data = self.df[(self.df['place_name'].str.contains(hospital_name_query, na=False)) & (self.df['search_keyword'] == keyword_query)]
        if hospital_data.empty:
            print("오류: 시뮬레이션 대상을 찾을 수 없습니다."); self.results['simulation'] = None; return
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
        print("\n7. 최종 보고서 생성...")
        
        def format_reliability(rel_dict):
            return (f"- **최적 모델**: {rel_dict['Model']} (파라미터: `{rel_dict['Best Params']}`)\n"
                    f"- **모델 설명력 (R-squared)**: {rel_dict['R-squared']:.3f}\n"
                    f"- **예측 오차 (MAE)**: 약 {rel_dict['MAE']:.2f}\n")

        # --- MD CONTENT ---
        md_content = "# 지역별 치과 순위 종합 분석 보고서\n\n"
        
        # 분석 1: 순위 영향 요인
        md_content += "## 1. 현재 순위에 영향을 미치는 요인\n"
        imp_df, rel_dict = self.results['rank_influence']
        md_content += imp_df.head().to_markdown(index=False) + "\n\n"
        md_content += "### 분석 모델 신뢰도\n" + format_reliability(rel_dict)
        md_content += f"**해석**: 이 모델은 순위의 상대적 위치를 약 {rel_dict['R-squared']:.1%} 정도 설명할 수 있으며, 예측 오차는 백분위 기준 약 {rel_dict['MAE']:.2f} 입니다.\n\n---\n\n"

        # 분석 2: 순위 변화 영향 요인
        md_content += "## 2. 순위 '변화'에 영향을 미치는 요인\n"
        imp_df, rel_dict = self.results['rank_change_influence']
        md_content += imp_df.head().to_markdown(index=False) + "\n\n"
        md_content += "### 분석 모델 신뢰도\n" + format_reliability(rel_dict)
        md_content += f"**해석**: 이 모델은 지난 기간의 순위 변동폭을 약 {rel_dict['R-squared']:.1%} 정도 설명할 수 있으며, 평균 예측 오차는 약 {rel_dict['MAE']:.1f}칸 입니다.\n\n---\n\n"
        
        # 분석 3: N2 동인 분석
        md_content += "## 3. 심층 분석: N2(관련성 지수)는 무엇에 영향을 받는가?\n"
        md_content += "N2가 순위에 가장 중요한 요인으로 확인되어, 어떤 활동이 N2 점수를 높이는지 추가 분석했습니다.\n\n"
        # 3.1
        imp_df, rel_dict = self.results['n2_drivers']
        md_content += "### 3.1. 현재 N2 점수와 리뷰 수의 관련성\n"
        md_content += imp_df.head().to_markdown(index=False) + "\n\n"
        md_content += "#### 분석 모델 신뢰도\n" + format_reliability(rel_dict)
        md_content += f"**해석**: 리뷰 관련 지표(특히 블로그 리뷰 수)는 현재 N2 점수의 변동을 약 {rel_dict['R-squared']:.1%}나 설명할 수 있습니다. 이는 **리뷰 수가 많은 병원이 N2 점수가 높을 확률이 매우 높다**는 것을 의미하며, 리뷰 활동이 N2 점수의 핵심 구성 요소임을 강력하게 시사합니다.\n\n"
        # 3.2
        imp_df, rel_dict = self.results['n2_change_drivers']
        md_content += "### 3.2. N2 점수 '변화'와 리뷰 변화의 관련성\n"
        md_content += imp_df.head().to_markdown(index=False) + "\n\n"
        md_content += "#### 분석 모델 신뢰도\n" + format_reliability(rel_dict)
        md_content += f"**해석**: 리뷰 수의 '변화'는 N2 점수 '변화'의 약 {rel_dict['R-squared']:.1%}를 설명합니다. 즉, 리뷰 수가 늘어나는 것이 N2 점수 상승에 긍정적인 영향을 주지만, 이것이 전부는 아님을 의미합니다.\n\n---\n\n"

        # 분석 4: 시뮬레이션
        md_content += "## 4. 순위 상승 시뮬레이션\n"
        if self.results.get('simulation'):
            name, rank, sims = self.results['simulation']
            md_content += f"**시뮬레이션 대상**: {name} (현재 순위: {rank:.1f}위)\n\n"
            for feature, results_df in sims.items():
                md_content += f"**'{feature}' 변화 시나리오:**\n{results_df.to_markdown(index=False)}\n\n"
        
        report_path = os.path.join(self.result_path, 'comprehensive_regional_analysis_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"보고서 저장 완료: {report_path}")

# --- 3. 메인 실행 블록 ---
def main():
    DATA_DIR = 'blog_automation/place_analysis/data/raw_data/지역별_검색순위'
    RESULT_PATH = 'blog_automation/place_analysis/analysis_result'

    loader = DataLoader(data_dir=DATA_DIR)
    master_df = loader.create_master_table()
    
    if master_df is not None and not master_df.empty:
        master_df = loader.engineer_features(master_df)
        analyzer = RegionalRankAnalyzer(df=master_df, result_path=RESULT_PATH)
        analyzer.analyze_rank_influence()
        analyzer.analyze_rank_change_influence()
        analyzer.analyze_n2_drivers()
        analyzer.simulate_rank_improvement(hospital_name_query='내이튼', keyword_query='동탄치과') 
        analyzer.generate_report()

if __name__ == "__main__":
    main()
