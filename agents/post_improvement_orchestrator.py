import pandas as pd
import joblib
import os
from datetime import datetime
import time
import json
import numpy as np
import re
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.google import Gemini
from sentence_transformers import SentenceTransformer, util
from sklearn.ensemble import StackingRegressor

# Import configurations
import blog_automation.agents.post_improvement_config as config

# Load environment variables (for API keys)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))


# --- LLM 및 에이전트 정의 (agno 프레임워크 사용) ---
LLM = Gemini(
    id=os.getenv("GEMINI_MODEL", "gemini-1.5-pro-latest"),
    api_key=os.getenv("GEMINI_API_KEY")
)

editor_agent = Agent(
    name="PostEditorAgent",
    role="SEO 콘텐츠 편집 전문가",
    model=LLM,
    instructions=[
        "당신은 특정 데이터 지표를 개선하기 위해 블로그 포스트를 수정하는 SEO 콘텐츠 편집 전문가입니다.",
        "현재 포스트의 제목, 본문, 그리고 개선 목표 피처와 구체적인 개선 제안이 주어집니다.",
        "당신의 임무는 제안에 따라 `post_title`과 `post_body`를 수정하는 것입니다.",
        "포스트의 핵심 주제와 메시지는 유지해야 합니다.",
        "결과는 반드시 순수한 JSON 형식으로만 반환해야 하며, 'new_title'과 'new_body' 키를 포함해야 합니다. 절대로 마크다운이나 다른 텍스트로 감싸지 마십시오.",
        "응답은 반드시 `{`로 시작하고 `}`로 끝나야 합니다."
    ],
    show_tool_calls=False,
    markdown=False
)

# --- 피처 계산 함수 (재평가용) ---
# (이전과 동일, 생략)
_model = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return _model
def calculate_relative_query_fulfillment_score(df: pd.DataFrame) -> pd.DataFrame:
    feature_name = 'relative_query_fulfillment_score'
    if feature_name not in df.columns: df[feature_name] = 1.0 
    return df
def calculate_relative_semantic_actionability(df: pd.DataFrame) -> pd.DataFrame:
    feature_name = 'relative_semantic_actionability'
    if feature_name not in df.columns: df[feature_name] = 1.0
    return df
def calculate_title_body_semantic_cohesion(df: pd.DataFrame) -> pd.DataFrame:
    feature_name = 'title_body_semantic_cohesion'
    try:
        model = get_model()
        titles = df['post_title'].fillna('').astype(str).tolist()
        bodies = df['post_body'].fillna('').astype(str).tolist()
        title_embeddings = model.encode(titles, convert_to_tensor=True, show_progress_bar=False)
        body_embeddings = model.encode(bodies, convert_to_tensor=True, show_progress_bar=False)
        cosine_scores = util.cos_sim(title_embeddings, body_embeddings).diag()
        df[feature_name] = cosine_scores.cpu().tolist()
    except Exception: df[feature_name] = np.nan
    return df
def calculate_title_hook_pattern_presence(df: pd.DataFrame) -> pd.DataFrame:
    feature_name = 'title_hook_pattern_presence'
    hook_keywords = ['방법', '후기', '비용', '해결', '정리', '추천', '꿀팁', '비교', '총정리', '솔직']
    combined_pattern = re.compile(r'(\d+)|(\?|까\s*?$)|' + f'({"|".join(hook_keywords)})')
    titles = df['post_title'].fillna('')
    df[feature_name] = titles.str.contains(combined_pattern, regex=True, na=False).astype(int)
    return df
CTR_FEATURE_FUNCS = [calculate_relative_query_fulfillment_score, calculate_relative_semantic_actionability, calculate_title_body_semantic_cohesion, calculate_title_hook_pattern_presence]


# --- 에이전트 함수 ---
def run_analyzer_agent(post_data: pd.Series, all_features: list) -> dict:
    """포스트의 초기 상태를 분석합니다."""
    print("    -> 초기 상태 분석 중...")
    
    analysis_result = {
        'post_id': post_data['post_id'],
        'initial_predicted_ctr': post_data['predicted_ctr'],
        'diagnosis': "전체 피처에 대한 순차적 개선 작업 준비",
        'suggestion': "각 피처를 개별 목표로 설정하여 개선 사이클 시작"
    }
    # 모든 피처 값을 결과에 포함
    for feature in all_features:
        analysis_result[feature] = post_data.get(feature)
        
    return analysis_result

def run_editor_agent(post_data: dict, target_feature: str) -> dict:
    """LLM을 사용하여 특정 피처 개선을 목표로 포스트를 수정합니다."""
    print(f"    -> 편집 에이전트 실행 (목표 피처: {target_feature})")
    
    suggestion_map = {
        'relative_query_fulfillment_score': '본문이 대표 검색어의 의도를 충분히 만족시키지 못하고 있습니다. 검색어에 대한 직접적인 답변, 관련 정보, 예시 등을 보강하여 내용을 더 구체적이고 충실하게 만드세요.',
        'title_body_semantic_cohesion': '제목과 본문의 주제가 의미적으로 잘 연결되지 않습니다. 제목을 본문 내용과 더 일치시키거나, 본문을 제목의 약속에 맞게 수정하여 일관성을 높이세요.',
        'relative_semantic_actionability': '콘텐츠에 독자가 실제로 취할 수 있는 행동(방법, 해결책, 가이드)이 부족합니다. "하는 법", "단계별 가이드", "해결책" 같은 구체적인 실행 방안을 제시하여 실용성을 높이세요.',
        'title_hook_pattern_presence': '제목이 독자의 클릭을 유도하는 매력적인 패턴(질문, 숫자, 특정 키워드)을 충분히 사용하지 않고 있습니다. 제목에 질문을 던지거나, 리스트 형식(예: 5가지 방법)을 사용하거나, "후기", "비교" 등의 키워드를 넣어 더 흥미롭게 만드세요.'
    }
    suggestion = suggestion_map.get(target_feature, f"{target_feature} 지표를 개선하는 방향으로 콘텐츠를 전반적으로 수정하세요.")

    prompt = f"""
**개선 목표 피처:** {target_feature}

**구체적인 개선 제안:**
{suggestion}

**현재 포스트 제목:**
{post_data['post_title']}

**현재 포스트 본문 (일부):**
{post_data['post_body'][:300]}...

**지시사항:**
위의 '구체적인 개선 제안'에 따라 **현재 포스트 제목과 본문 전체를 수정**해주세요.
"""

    try:
        response_raw = editor_agent.run(prompt).content
        json_match = re.search(r'\{.*\}', response_raw, re.DOTALL)
        if not json_match:
            print(f"      [오류] 에이전트가 유효한 JSON을 반환하지 않았습니다. 원본 데이터를 유지합니다.")
            return post_data
        
        edited_content = json.loads(json_match.group(0))
        new_post_data = post_data.copy()
        new_post_data['post_title'] = edited_content.get('new_title', post_data['post_title'])
        new_post_data['post_body'] = edited_content.get('new_body', post_data['post_body'])
        print(f"      - 편집 완료: 제목이 '{new_post_data['post_title']}' (으)로 변경되었습니다.")
        return new_post_data
    except Exception as e:
        print(f"      [오류] 에이전트 호출 실패: {e}. 원본 데이터를 유지합니다.")
        return post_data

def run_reevaluator_agent(edited_post_data: dict, model, feature_names: list):
    """수정된 포스트의 피처를 재계산하고 점수를 예측합니다."""
    print(f"    -> 재평가 에이전트 실행...")
    post_df = pd.DataFrame([edited_post_data])
    for func in CTR_FEATURE_FUNCS:
        post_df = func(post_df)
    
    final_features_df = post_df[feature_names].fillna(0)
    new_score = model.predict(final_features_df)[0]
    print(f"      - 재평가 완료: 새로운 예측 점수는 {new_score:.4f} 입니다.")
    return new_score, final_features_df.to_dict('records')[0]

def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", str(name))

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)): return obj.item()
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

# --- 메인 오케스트레이터 ---
class PostImprovementOrchestrator:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_output_path = os.path.join(config.OUTPUT_DATA_PATH, self.timestamp)
        os.makedirs(self.run_output_path, exist_ok=True)
        print(f"이번 실행의 모든 결과는 다음 폴더에 저장됩니다: {self.run_output_path}")

        self.model = joblib.load(config.CTR_MODEL_PATH)
        self.master_df = pd.read_csv(config.MASTER_DATA_PATH)
        self._extract_feature_importances()
        self.analysis_history = []
        self.edit_history = []

    def _extract_feature_importances(self):
        # (이전과 동일, 생략)
        self.feature_importances, self.feature_names = {}, []
        if hasattr(self.model, 'feature_names_in_'): self.feature_names = self.model.feature_names_in_
        else:
            try:
                features_path = os.path.join(config.DATA_PATH, config.FEATURE_FILES['ctr'])
                ctr_features_df = pd.read_csv(features_path)
                self.feature_names = [col for col in ctr_features_df.columns if col not in ['post_id', 'post_identifier']]
            except Exception: self.feature_names = list(CTR_FEATURE_FUNCS.keys())
        if hasattr(self.model, 'feature_importances_'): self.feature_importances = pd.Series(self.model.feature_importances_, index=self.feature_names).to_dict()
        else: self.feature_importances = {name: 1.0 for name in self.feature_names}

    def _predict_initial_ctr(self):
        print("--- 모든 포스트의 초기 CTR 예측 중 ---")
        self.master_df['post_id'] = self.master_df['post_id'].astype(str)
        features_path = os.path.join(config.DATA_PATH, config.FEATURE_FILES['ctr'])
        try:
            feature_df = pd.read_csv(features_path)
            if 'post_identifier' in feature_df.columns: feature_df = feature_df.rename(columns={'post_identifier': 'post_id'})
            feature_df['post_id'] = feature_df['post_id'].astype(str)
            feature_df = feature_df.drop_duplicates(subset=['post_id'], keep='first')
        except FileNotFoundError:
            self.master_df['predicted_ctr'] = 0
            print(f"오류: 피처 파일을 찾을 수 없습니다 ({features_path}). 예측을 중단합니다.")
            return
        
        self.master_df = pd.merge(self.master_df, feature_df, on='post_id', how='left')
        self.master_df[self.feature_names] = self.master_df[self.feature_names].fillna(0)
        self.master_df['predicted_ctr'] = self.model.predict(self.master_df[self.feature_names])
        print(f"초기 예측 완료. 평균 예측 CTR: {self.master_df['predicted_ctr'].mean():.4f}")

    def run_improvement_process(self):
        self._predict_initial_ctr()
        posts_to_improve = self.master_df.sort_values('predicted_ctr').head(config.NUM_POSTS_TO_IMPROVE)
        
        print(f"\n--- 총 {len(posts_to_improve)}개 포스트에 대한 개선 작업을 시작합니다 ---")

        for i, (idx, initial_post) in enumerate(posts_to_improve.iterrows()):
            post_id = initial_post['post_id']
            print(f"\n[{i+1}/{len(posts_to_improve)}] 포스트 개선 작업 시작 (ID: {post_id})")
            
            # 1. 초기 분석 (단 1회)
            analysis_result = run_analyzer_agent(initial_post, self.feature_names)
            self.analysis_history.append(analysis_result)

            # 2. Edit History 초기화
            current_best_post = initial_post.to_dict()
            edit_log_for_post = {
                'post_id': post_id,
                'initial': {
                    'post_title': initial_post['post_title'],
                    'post_body': initial_post['post_body'],
                    'feature_values': {f: initial_post.get(f) for f in self.feature_names},
                    'model_score': initial_post['predicted_ctr']
                }
            }

            # 3. 피처별 순차 개선 루프
            for feature_to_improve in self.feature_names:
                print(f"  - 피처 '{feature_to_improve}' 개선 시작...")
                best_attempt_for_feature = {'score': -1, 'data': None, 'features': None}
                
                for attempt in range(config.MAX_ATTEMPTS_PER_FEATURE):
                    print(f"    [시도 {attempt + 1}/{config.MAX_ATTEMPTS_PER_FEATURE}]")
                    
                    edited_post = run_editor_agent(current_best_post, feature_to_improve)
                    new_score, new_features = run_reevaluator_agent(edited_post, self.model, self.feature_names)
                    
                    version_key = f"v{attempt+1}_{feature_to_improve}"
                    edit_log_for_post[version_key] = {
                        'post_title': edited_post['post_title'],
                        'post_body': edited_post['post_body'],
                        'feature_values': new_features,
                        'model_score': new_score
                    }
                    
                    if new_score > best_attempt_for_feature['score']:
                        best_attempt_for_feature = {'score': new_score, 'data': edited_post, 'features': new_features}

                # 현재 피처에 대한 최고 버전으로 업데이트
                current_best_post = best_attempt_for_feature['data']
                current_best_post.update(best_attempt_for_feature['features'])
                print(f"  - 피처 '{feature_to_improve}' 개선 완료. 최고 점수: {best_attempt_for_feature['score']:.4f}")

            self.edit_history.append(edit_log_for_post)

        self.save_history_files()
        print("\n--- 모든 작업이 성공적으로 완료되었습니다. ---")

    def save_history_files(self):
        print("\n--- 결과 파일 저장 중 ---")
        analysis_df = pd.DataFrame(self.analysis_history)
        if not analysis_df.empty:
            cols_order = ['post_id', 'initial_predicted_ctr', 'diagnosis', 'suggestion'] + self.feature_names
            analysis_df = analysis_df.reindex(columns=cols_order, fill_value=np.nan)
        analysis_path = os.path.join(self.run_output_path, "analysis_history.csv")
        analysis_df.to_csv(analysis_path, index=False, encoding='utf-8-sig')
        print(f"분석 히스토리 저장 완료: {analysis_path}")
        
        edit_history_path = os.path.join(self.run_output_path, "edit_history.json")
        with open(edit_history_path, 'w', encoding='utf-8') as f:
            json.dump(self.edit_history, f, ensure_ascii=False, indent=4, cls=NpEncoder)
        print(f"편집 히스토리 저장 완료: {edit_history_path}")

def main():
    orchestrator = PostImprovementOrchestrator()
    orchestrator.run_improvement_process()

if __name__ == "__main__":
    import warnings
    warnings.catch_warnings()
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FutureWarning)
    main() 