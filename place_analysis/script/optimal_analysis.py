import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, classification_report
from functools import reduce
import warnings
import os

warnings.filterwarnings('ignore')

class OptimalTimeSeriesAnalyzer:
    def __init__(self, data_dir, result_path):
        self.data_dir = data_dir
        self.result_path = result_path
        self.wide_df = None
        self.results = {}
        os.makedirs(self.result_path, exist_ok=True)

    def _clean_df(self, df):
        if 'company_name_category' in df.columns:
            df['company_name_category'] = df['company_name_category'].astype(str).str.strip()
        for col in df.columns:
            if col != 'company_name_category':
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace('-', '0', regex=False), 
                    errors='coerce'
                ).fillna(0)
        return df

    def load_and_integrate_data(self):
        print("1. 모든 시계열 데이터 통합 및 특성 재구성...")
        time_points = [1, 5, 10, 20, 30, 60]
        dataframes = []
        for tp in time_points:
            try:
                file_path = os.path.join(self.data_dir, f'rank_vs{tp}.csv')
                df = self._clean_df(pd.read_csv(file_path))
                df = df.add_suffix(f'_{tp}')
                df.rename(columns={f'company_name_category_{tp}': 'company_name_category'}, inplace=True)
                dataframes.append(df)
            except FileNotFoundError:
                print(f"경고: rank_vs{tp}.csv 파일을 찾을 수 없습니다. 분석에서 제외합니다.")
        
        # 모든 데이터프레임을 company_name_category 기준으로 병합
        self.wide_df = reduce(lambda left, right: pd.merge(left, right, on='company_name_category', how='outer'), dataframes)
        
        # 타겟 변수 및 파생 변수 생성 (60일 기준)
        self.wide_df = self.wide_df.fillna(0)
        self.wide_df['rank'] = self.wide_df['rank_60']
        self.wide_df['grade'] = ((self.wide_df['rank_60'] - 1) // 10).astype(int)
        self.wide_df['rank_change_direction'] = np.sign(self.wide_df['rank_change_60']).astype(int)
        
        grade_60 = ((self.wide_df['rank_60'] - 1) // 10)
        grade_1 = ((self.wide_df['rank_1'] - 1) // 10)
        self.wide_df['grade_change_direction'] = np.sign(grade_60 - grade_1).astype(int)
        
        print(f"데이터 통합 완료. 최종 특성 개수: {self.wide_df.shape[1]}")
        return self.wide_df

    def _find_optimal_model(self, target_col, features, model_type='regressor'):
        X = self.wide_df[features].copy()
        y = self.wide_df[target_col].copy()

        stratify_col = y if model_type == 'classifier' and y.nunique() > 1 else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_col)
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if model_type == 'regressor':
            models_config = {
                'RandomForestRegressor': (RandomForestRegressor(random_state=42), {'n_estimators': [100, 200], 'max_depth': [10, 20]}),
                'GradientBoostingRegressor': (GradientBoostingRegressor(random_state=42), {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]})
            }
            scoring = 'r2'
        else: # classifier
            models_config = {
                'RandomForestClassifier': (RandomForestClassifier(random_state=42, class_weight='balanced'), {'n_estimators': [100, 200], 'max_depth': [5, 10]}),
                'GradientBoostingClassifier': (GradientBoostingClassifier(random_state=42), {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]})
            }
            scoring = 'accuracy'

        best_model = None
        best_score = -np.inf
        best_model_name = ""

        for name, (model, params) in models_config.items():
            print(f"  - {name} 모델 최적화 중...")
            grid_search = GridSearchCV(model, params, cv=3, scoring=scoring, n_jobs=-1)
            grid_search.fit(X_train_scaled, y_train)
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model = grid_search.best_estimator_
                best_model_name = name
        
        print(f"  => 최적 모델: {best_model_name} (CV 점수: {best_score:.3f})")

        y_pred = best_model.predict(X_test_scaled)
        if model_type == 'regressor':
            reliability = {'R-squared': r2_score(y_test, y_pred), 'MAE': mean_absolute_error(y_test, y_pred)}
        else:
            reliability = {'Accuracy': accuracy_score(y_test, y_pred), 'Classification Report': classification_report(y_test, y_pred, output_dict=True, zero_division=0)}
        
        feature_importance = pd.DataFrame({'feature': features, 'importance': best_model.feature_importances_}).sort_values(by='importance', ascending=False)
        
        return best_model_name, feature_importance, reliability
    
    def analyze_all(self):
        # 1. 순위 영향 요인 분석
        print("\n2. 순위 영향 요인 분석...")
        features_rank = [c for c in self.wide_df.columns if any(p in c for p in ['n1', 'n2', 'visitor_reviews', 'blog_reviews']) and 'change' not in c and 'rank' not in c and 'grade' not in c]
        self.results['rank_influence'] = self._find_optimal_model('rank', features_rank, 'regressor')

        # 2. 등급 영향 요인 분석
        print("\n3. 등급 영향 요인 분석...")
        self.results['grade_influence'] = self._find_optimal_model('grade', features_rank, 'classifier')
        
        # 3. 순위 변화 영향 요인 분석
        print("\n4. 순위 '변화' 영향 요인 분석...")
        features_change = [c for c in self.wide_df.columns if any(p in c for p in ['_change', 'rate', '_ma']) and 'rank' not in c and 'grade' not in c]
        if self.wide_df['rank_change_direction'].nunique() > 1:
            self.results['rank_change_influence'] = self._find_optimal_model('rank_change_direction', features_change, 'classifier')
        else:
            self.results['rank_change_influence'] = ("N/A", pd.DataFrame(), {'Error': '단일 클래스 데이터'})

        # 4. 등급 변화 영향 요인 분석
        print("\n5. 등급 '변화' 영향 요인 분석...")
        if self.wide_df['grade_change_direction'].nunique() > 1:
            self.results['grade_change_influence'] = self._find_optimal_model('grade_change_direction', features_change, 'classifier')
        else:
            self.results['grade_change_influence'] = ("N/A", pd.DataFrame(), {'Error': '단일 클래스 데이터'})

    def generate_report(self):
        print("\n6. 최종 보고서 생성...")
        
        def format_reliability(name, rel_dict):
            report = f"- **최적 모델**: {name}\n"
            if 'R-squared' in rel_dict:
                report += f"- **분석 정확도 (R-squared)**: {rel_dict['R-squared']:.3f}\n"
                report += f"- **분석 신뢰도 (평균절대오차)**: {rel_dict['MAE']:.2f} 위\n"
            if 'Accuracy' in rel_dict:
                report += f"- **분석 정확도 (Accuracy)**: {rel_dict['Accuracy']:.3f}\n"
                report += "- **분석 신뢰도 (상세 리포트)**:\n\n"
                report_df = pd.DataFrame(rel_dict['Classification Report']).transpose().drop('accuracy')
                report += report_df[['precision', 'recall', 'f1-score', 'support']].to_markdown()
            if 'Error' in rel_dict:
                 report += f"- **분석 불가**: {rel_dict['Error']}\n"
            return report
        
        def format_importance(importance_df):
            if importance_df.empty: return "데이터 부족으로 분석 불가\n"
            return importance_df.head(10).to_markdown(index=False)
        
        md_content = f"""# 동탄 치과 순위 최종 분석 보고서
## 분석 개요
- **분석 대상**: 동탄 지역 치과 목록
- **분석 기간**: 1일, 5일, 10일, 20일, 30일, 60일 간의 데이터 변화를 통합 분석
- **분석 목표**: 순위 및 등급에 영향을 미치는 핵심 요인을 파악하고, 최적화된 모델로 신뢰도를 평가합니다.

## 시계열 데이터 활용 현황
- **분석에 사용된 총 업체 수**: {len(self.wide_df)}개
- **통합된 시계열 데이터 포인트**: 1, 5, 10, 20, 30, 60일 전 데이터

---
"""
        
        analyses = {
            "1. 순위에 영향을 미치는 요인 분석": 'rank_influence',
            "2. 등급(10위 단위)에 영향을 미치는 요인 분석": 'grade_influence',
            "3. 순위 '변화'에 영향을 미치는 요인 분석": 'rank_change_influence',
            "4. 등급 '변화'에 영향을 미치는 요인 분석": 'grade_change_influence'
        }
        
        for title, key in analyses.items():
            md_content += f"## {title}\n"
            if key in self.results:
                name, fi, rel = self.results[key]
                md_content += "### 요인 및 영향도 (Top 10)\n"
                md_content += format_importance(fi) + "\n\n"
                md_content += "### 분석 정확도 및 신뢰도\n"
                md_content += format_reliability(name, rel) + "\n"
                if key == 'grade_influence' and 'rank_influence' in self.results:
                    mae = self.results['rank_influence'][2]['MAE']
                    md_content += f"**해설**: 등급 예측 정확도가 낮은 이유는 순위 예측 모델의 평균 오차(MAE)가 약 {mae:.2f}위로, 이는 등급 하나(10위)의 범위를 넘는 오차이기 때문입니다. 모델이 대략적인 순위 범위는 잘 예측하지만, 10위 단위의 정확한 등급 구간을 맞추는 데에는 한계가 있습니다.\n"
            md_content += "---\n"

        report_path = os.path.join(self.result_path, 'final_analysis_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"보고서 저장 완료: {report_path}")

if __name__ == "__main__":
    analyzer = OptimalTimeSeriesAnalyzer(
        data_dir='blog_automation/place_analysis/data/raw_data/dongtan_chigwa',
        result_path='blog_automation/place_analysis/analysis_result'
    )
    if analyzer.load_and_integrate_data() is not None:
        analyzer.analyze_all()
        analyzer.generate_report()
