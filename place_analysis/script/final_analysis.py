import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
import os

warnings.filterwarnings('ignore')

class FinalRankAnalyzer:
    def __init__(self, data_path, result_path):
        self.data_path = data_path
        self.result_path = result_path
        os.makedirs(self.result_path, exist_ok=True)
        
        # 분석 결과를 저장할 속성 초기화
        self.df = None
        self.results = {}
        self.rank_model = None
        self.rank_model_features = []
        self.rank_model_scaler = None

    def _clean_df(self, df):
        """데이터프레임 정리 및 타입 변환"""
        if 'company_name_category' in df.columns:
            df['company_name_category'] = df['company_name_category'].astype(str).str.strip()

        for col in df.columns:
            if col != 'company_name_category':
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace('-', '0', regex=False), 
                    errors='coerce'
                ).fillna(0)
        return df

    def load_and_preprocess(self):
        """데이터 로드 및 필요한 파생 변수 생성"""
        print("1. 데이터 로드 및 전처리...")
        try:
            self.df = self._clean_df(pd.read_csv(self.data_path))
            
            # 기본 파생 변수
            self.df['total_reviews'] = self.df['visitor_reviews'] + self.df['blog_reviews']
            self.df['total_reviews_change'] = self.df['visitor_reviews_change'] + self.df['blog_reviews_change']

            # 변화율 변수 (분모가 0인 경우 방지)
            self.df['n1_change_rate'] = self.df['n1_change'] / (self.df['n1'] + 1e-6)
            self.df['n2_change_rate'] = self.df['n2_change'] / (self.df['n2'] + 1e-6)
            self.df['visitor_reviews_change_rate'] = self.df['visitor_reviews_change'] / (self.df['visitor_reviews'] + 1e-6)
            self.df['total_reviews_change_rate'] = self.df['total_reviews_change'] / (self.df['total_reviews'] + 1e-6)
            
            self.df.replace([np.inf, -np.inf], 0, inplace=True) # 혹시 모를 inf 값 처리

            print("데이터 준비 완료.")
            return True
        except FileNotFoundError:
            print(f"오류: 데이터 파일을 찾을 수 없습니다 - {self.data_path}")
            return False

    def analyze_rank_influence(self):
        """분석 1: 현재 순위에 영향을 미치는 요인 분석"""
        print("\n2. 현재 순위 영향 요인 분석...")
        
        self.rank_model_features = ['n1', 'n2', 'visitor_reviews', 'total_reviews']
        X = self.df[self.rank_model_features].copy()
        y = self.df['rank'].copy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.rank_model_scaler = StandardScaler()
        X_train_scaled = self.rank_model_scaler.fit_transform(X_train)
        X_test_scaled = self.rank_model_scaler.transform(X_test)

        self.rank_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rank_model.fit(X_train_scaled, y_train)
        
        y_pred = self.rank_model.predict(X_test_scaled)
        
        reliability = {
            'R-squared': r2_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred)
        }
        
        importance = pd.DataFrame({
            'feature': self.rank_model_features,
            'importance': self.rank_model.feature_importances_
        }).sort_values(by='importance', ascending=False)
        
        self.results['rank_influence'] = (importance, reliability)
        print("분석 완료.")

    def analyze_rank_change_influence(self):
        """분석 3: 순위 '변화'에 영향을 미치는 요인 분석"""
        print("\n3. 순위 변화 영향 요인 분석...")
        
        change_features = [
            'n1', 'n2', 'visitor_reviews', 'total_reviews',
            'n1_change', 'n2_change', 'visitor_reviews_change', 'total_reviews_change',
            'n1_change_rate', 'n2_change_rate', 'visitor_reviews_change_rate', 'total_reviews_change_rate'
        ]
        
        X = self.df[change_features].copy()
        y = self.df['rank_change'].copy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        
        reliability = {
            'R-squared': r2_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred)
        }
        
        importance = pd.DataFrame({
            'feature': change_features,
            'importance': model.feature_importances_
        }).sort_values(by='importance', ascending=False)
        
        self.results['rank_change_influence'] = (importance, reliability)
        print("분석 완료.")

    def simulate_rank_improvement(self, hospital_index=20):
        """분석 2: 순위 상승 시뮬레이션"""
        print(f"\n4. 순위 상승 시뮬레이션 (대상: {hospital_index+1}위 병원)...")
        if self.rank_model is None:
            print("오류: 순위 예측 모델이 먼저 훈련되어야 합니다.")
            return

        hospital_data = self.df.iloc[[hospital_index]]
        original_data_scaled = self.rank_model_scaler.transform(hospital_data[self.rank_model_features])
        original_rank = self.rank_model.predict(original_data_scaled)[0]

        simulations = {}
        # 시뮬레이션할 변수와 변화량
        scenarios = {
            'visitor_reviews': [10, 50, 100],
            'n2': [0.01, 0.05, 0.1]
        }

        for feature, changes in scenarios.items():
            sim_results = []
            for change in changes:
                sim_data = hospital_data[self.rank_model_features].copy()
                sim_data[feature] += change
                
                # total_reviews 업데이트
                if feature == 'visitor_reviews':
                    sim_data['total_reviews'] = sim_data['visitor_reviews'] + hospital_data['blog_reviews']

                sim_data_scaled = self.rank_model_scaler.transform(sim_data)
                predicted_rank = self.rank_model.predict(sim_data_scaled)[0]
                sim_results.append({
                    'change': f"+{change}",
                    'predicted_rank': f"{predicted_rank:.1f}위",
                    'improvement': f"{original_rank - predicted_rank:.1f}위 상승"
                })
            simulations[feature] = sim_results
        
        self.results['simulation'] = (hospital_data['company_name_category'].iloc[0], original_rank, simulations)
        print("시뮬레이션 완료.")

    def generate_report(self):
        """최종 분석 보고서 생성"""
        print("\n5. 최종 보고서 생성...")

        # 신뢰도 포맷팅 함수
        def format_reliability(rel_dict):
            report = f"- **분석 정확도 (R-squared)**: {rel_dict['R-squared']:.3f} (모델이 데이터의 약 {rel_dict['R-squared']:.1%}를 설명합니다.)\n"
            report += f"- **분석 신뢰도 (평균 오차)**: 약 ±{rel_dict['MAE']:.1f} 순위\n"
            return report

        # 중요도 포맷팅 함수
        def format_importance(importance_df):
            return importance_df.head(5).to_markdown(index=False)

        # 시뮬레이션 결과 포맷팅
        sim_report = ""
        if 'simulation' in self.results:
            name, rank, sims = self.results['simulation']
            sim_report += f"**시뮬레이션 대상**: {name} (현재 예상 순위: {rank:.1f}위)\n\n"
            for feature, results in sims.items():
                sim_report += f"**'{feature}' 변화 시나리오:**\n\n"
                sim_report += pd.DataFrame(results).to_markdown(index=False) + "\n\n"

        # --- MD CONTENT ---
        md_content = f"""# 동탄 치과 순위 분석 보고서

## 분석 목표
1. 현재 순위에 영향을 미치는 핵심 요인과 그 영향력을 파악합니다.
2. 특정 병원의 순위를 높이기 위해 어떤 요소를 얼마나 개선해야 하는지 시뮬레이션합니다.
3. 지난 60일간의 순위 '변화'에 영향을 미친 요인을 분석합니다.

---

## 1. 현재 순위에 영향을 미치는 요인은 무엇인가?

'현재 순위'는 특정 시점의 병원 상태 값(리뷰 수, N지수 등)에 의해 결정됩니다.

### 요인 및 영향도 (상위 5개)
{format_importance(self.results['rank_influence'][0])}

### 분석 정확도 및 신뢰도
{format_reliability(self.results['rank_influence'][1])}

---

## 2. 순위를 높이려면 무엇을 해야 하는가? (시뮬레이션)

분석 1의 모델을 사용하여, 특정 병원의 지표를 변화시켰을 때 순위가 어떻게 변하는지 예측했습니다.

{sim_report}

---

## 3. 순위 '변화'에 영향을 미치는 요인은 무엇인가?

'순위 변화'는 지난 60일간의 '변화량' 및 '변화율' 지표들이 더 큰 영향을 미칩니다.

### 요인 및 영향도 (상위 5개)
{format_importance(self.results['rank_change_influence'][0])}

### 분석 정확도 및 신뢰도
{format_reliability(self.results['rank_change_influence'][1])}
"""
        report_path = os.path.join(self.result_path, 'final_analysis_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"보고서 저장 완료: {report_path}")

    def run_all(self):
        if self.load_and_preprocess():
            self.analyze_rank_influence()
            self.analyze_rank_change_influence()
            self.simulate_rank_improvement()
            self.generate_report()

if __name__ == "__main__":
    analyzer = FinalRankAnalyzer(
        data_path='blog_automation/place_analysis/data/raw_data/dongtan_chigwa/rank_vs60.csv',
        result_path='blog_automation/place_analysis/analysis_result'
    )
    analyzer.run_all()
