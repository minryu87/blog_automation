import pandas as pd
import numpy as np
import os
import json
import re
import joblib
from datetime import datetime
import warnings
from sentence_transformers import SentenceTransformer, util
import torch
from dotenv import load_dotenv

import post_improvement_config as config
from agno.agent import Agent
from agno.models.google import Gemini

# .env 파일에서 환경 변수를 로드 (상위 디렉토리에 위치)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.cluster._kmeans')

# --- Utilities ---

class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def sanitize_filename(filename):
    """Sanitizes a string to be a valid filename."""
    return re.sub(r'[\\/*?:"<>|]', "", filename)

# --- Feature Recalculation Functions (from generate_feature_datasets.py) ---

_model = None

def get_model():
    """Initializes and returns the sentence transformer model."""
    global _model
    if _model is None:
        print("Sentence Transformer 모델 로딩 중... (최초 1회)")
        _model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return _model

# (The four CTR feature calculation functions will be pasted here)
def calculate_relative_query_fulfillment_score(df: pd.DataFrame) -> pd.DataFrame:
    # ... Implementation from generate_feature_datasets.py ...
    feature_name = 'relative_query_fulfillment_score'
    if df.empty:
        df[feature_name] = np.nan
        return df

    model = get_model()
    
    df['post_body'] = df['post_body'].fillna('').astype(str)
    df['representative_query'] = df['representative_query'].fillna('').astype(str)

    unique_queries = df['representative_query'].unique().tolist()
    post_bodies = df['post_body'].tolist()

    query_embeddings = model.encode(unique_queries, convert_to_tensor=True, show_progress_bar=False)
    body_embeddings = model.encode(post_bodies, convert_to_tensor=True, show_progress_bar=False)

    query_embedding_map = {query: emb for query, emb in zip(unique_queries, query_embeddings)}
    
    ordered_query_embeddings = torch.stack([query_embedding_map.get(q, torch.zeros(model.get_sentence_embedding_dimension()).to(body_embeddings.device)) for q in df['representative_query']])
    
    fulfillment_scores = util.cos_sim(body_embeddings, ordered_query_embeddings).diag()
    df['temp_fulfillment_score'] = fulfillment_scores.cpu().numpy()

    avg_scores = df.groupby(['representative_query', 'source'])['temp_fulfillment_score'].mean().unstack()

    if 'ours' not in avg_scores.columns:
        avg_scores['ours'] = np.nan
    if 'competitor' not in avg_scores.columns:
        avg_scores['competitor'] = np.nan

    relative_scores = (avg_scores['ours'] / avg_scores['competitor']).fillna(1.0)
    relative_scores.replace([np.inf, -np.inf], 1.0, inplace=True)

    df[feature_name] = df['representative_query'].map(relative_scores)
    df[feature_name].fillna(1.0, inplace=True)

    df.drop(columns=['temp_fulfillment_score'], inplace=True)
    return df

def calculate_relative_semantic_actionability(df: pd.DataFrame) -> pd.DataFrame:
    # ... Implementation from generate_feature_datasets.py ...
    feature_name = 'relative_semantic_actionability'
    if df.empty:
        df[feature_name] = 1.0
        return df

    action_keywords = [
        '방법', '해결', '가이드', '팁', '전략', '단계', '하는 법', '솔루션',
        'how to', 'solution', 'guide', 'tip', 'strategy', 'steps', 'tutorial'
    ]
    model = get_model()
    action_embeddings = model.encode(action_keywords, convert_to_tensor=True, show_progress_bar=False)
    actionability_vector = action_embeddings.mean(dim=0)

    bodies = df['post_body'].fillna('').astype(str).tolist()
    body_embeddings = model.encode(bodies, convert_to_tensor=True, show_progress_bar=False)
    actionability_vector_device = actionability_vector.to(body_embeddings.device)
    actionability_scores = util.cos_sim(body_embeddings, actionability_vector_device)
    df['temp_action_score'] = actionability_scores.cpu().numpy().flatten()

    avg_competitor_score = df[df['source'] == 'competitor']['temp_action_score'].mean()
    
    if pd.isna(avg_competitor_score) or avg_competitor_score < 1e-6:
        avg_competitor_score = df['temp_action_score'].mean() # Fallback
    if pd.isna(avg_competitor_score) or avg_competitor_score < 1e-6:
        df[feature_name] = 1.0
    else:
        df[feature_name] = df['temp_action_score'] / avg_competitor_score
    
    df.drop(columns=['temp_action_score'], inplace=True)
    df[feature_name].fillna(1.0, inplace=True)
    return df

def calculate_title_body_semantic_cohesion(df: pd.DataFrame) -> pd.DataFrame:
    # ... Implementation from generate_feature_datasets.py ...
    feature_name = 'title_body_semantic_cohesion'
    if df.empty:
        df[feature_name] = 0.0
        return df
    model = get_model()
    titles = df['post_title'].fillna('').astype(str).tolist()
    bodies = df['post_body'].fillna('').astype(str).tolist()
    title_embeddings = model.encode(titles, convert_to_tensor=True, show_progress_bar=False)
    body_embeddings = model.encode(bodies, convert_to_tensor=True, show_progress_bar=False)
    cosine_scores = util.cos_sim(title_embeddings, body_embeddings).diag()
    df[feature_name] = cosine_scores.cpu().tolist()
    return df

def calculate_title_hook_pattern_presence(df: pd.DataFrame) -> pd.DataFrame:
    # ... Implementation from generate_feature_datasets.py ...
    feature_name = 'title_hook_pattern_presence'
    hook_keywords = [
        '방법', '후기', '비용', '해결', '정리', '추천', '꿀팁', '비교', '총정리', '솔직'
    ]
    combined_pattern = re.compile(
        r'(\d+)|' + 
        r'(\?|까\s*?$)|' + 
        f'({"|".join(hook_keywords)})'
    )
    titles = df['post_title'].fillna('')
    df[feature_name] = titles.str.contains(combined_pattern, regex=True, na=False).astype(int)
    return df

CTR_FEATURE_FUNCS = [
    calculate_relative_query_fulfillment_score,
    calculate_relative_semantic_actionability,
    calculate_title_body_semantic_cohesion,
    calculate_title_hook_pattern_presence,
]
ALL_CTR_FEATURE_NAMES = [
    'relative_query_fulfillment_score',
    'relative_semantic_actionability',
    'title_body_semantic_cohesion',
    'title_hook_pattern_presence',
]

# --- Agent Definitions ---

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

def run_editor_agent(post_data: dict, target_feature: str, agent) -> dict:
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
        # 1단계: 생성 완료 여부 직접 확인을 위해 전체 응답 객체 수신
        response_obj = agent.run(prompt)
        response_raw = response_obj.content
        
        # 1-1: 토큰 길이 제한으로 응답이 중간에 잘렸는지 확인
        if response_obj.finish_reason == 'length':
            raise ValueError("LLM 응답이 토큰 길이 제한으로 잘렸습니다.")

        json_match = re.search(r'\{.*\}', response_raw, re.DOTALL)
        if not json_match:
            raise ValueError("에이전트가 유효한 JSON을 반환하지 않았습니다.")
        
        edited_content = json.loads(json_match.group(0))
        new_title = edited_content.get('new_title', post_data['post_title'])
        new_body = edited_content.get('new_body', post_data['post_body'])

        # 2단계: 내용 기반 검증 (EOD 마커 확인)
        if not new_body.strip().endswith("<<EOD>>"):
            raise ValueError("생성된 본문에 EOD 마커가 없어 내용이 잘린 것으로 간주됩니다.")
            
        # 검증 후 실제 저장될 본문에서는 마커 제거
        new_body_cleaned = new_body.rsplit("<<EOD>>", 1)[0].strip()

        new_post_data = post_data.copy()
        new_post_data['post_title'] = new_title
        new_post_data['post_body'] = new_body_cleaned
        print(f"      - 편집 완료: 제목이 '{new_post_data['post_title']}' (으)로 변경되었습니다.")
        return new_post_data
    except Exception as e:
        # 여기서 발생한 예외는 상위 루프로 전파하여 재시도 등을 처리하도록 함
        print(f"      [오류] 에이전트 호출 또는 검증 실패: {e}. 이번 시도를 건너뜁니다.")
        raise  # 예외를 다시 발생시켜 상위에서 처리하도록 함

def run_reevaluator_agent(edited_post_data: dict, competitor_data_for_query: pd.DataFrame, model, feature_names_for_model: list) -> tuple[float, dict]:
    """
    Recalculates features for the edited post and predicts the new CTR.
    """
    print(f"    - 재평가 시작...")
    
    # 1. Prepare data for recalculation
    our_post_df = pd.DataFrame([edited_post_data])
    our_post_df['source'] = 'ours'

    # Combine our edited post with the static competitor data
    recalc_df = pd.concat([our_post_df, competitor_data_for_query], ignore_index=True)
    
    # 2. Recalculate all CTR features
    for func in CTR_FEATURE_FUNCS:
        recalc_df = func(recalc_df)
    
    # 3. Extract the new feature values for our post
    new_features_series = recalc_df[recalc_df['source'] == 'ours'].iloc[0]
    new_features_dict = new_features_series[ALL_CTR_FEATURE_NAMES].to_dict()

    # 4. Predict new score
    X_pred = new_features_series[feature_names_for_model].to_frame().T
    new_score = model.predict(X_pred)[0]
    
    print(f"    - 재평가 완료: 새로운 예측 CTR = {new_score:.4f}")
    return new_score, new_features_dict

# --- Orchestrator Class ---

class PostImprovementOrchestrator:
    def __init__(self):
        self.model = self._load_model()
        self.feature_names_for_model = self._extract_feature_importances()
        
        # 마스터 데이터를 먼저 로드하여 candidate와 competitor 데이터 생성에 모두 활용
        master_df = self._load_master_data()
        self.candidate_posts = self._prepare_candidates(master_df)
        self.competitor_data = self._prepare_competitor_data(master_df)

        self.run_output_path = self._setup_output_directory()
        self.editor_agent = self._setup_editor_agent()

    def _load_model(self):
        print("모델 로딩 중...")
        try:
            return joblib.load(config.CTR_MODEL_PATH)
        except FileNotFoundError:
            print(f"오류: CTR 모델 파일을 찾을 수 없습니다 ({config.CTR_MODEL_PATH}). 프로그램을 종료합니다.")
            exit()

    def _extract_feature_importances(self):
        # Model is trained on these two features
        return ['relative_query_fulfillment_score', 'title_body_semantic_cohesion']

    def _load_master_data(self):
        """
        Loads the complete master data file, handling the column name difference.
        """
        print("마스터 데이터 로딩 중 (`master_post_data.csv`)...")
        try:
            df = pd.read_csv(config.COMPETITOR_DATA_PATH)
            
            # 'post_identifier'와 'post_id' 컬럼이 모두 존재할 경우의 중복 문제 해결
            if 'post_identifier' in df.columns:
                # 만약 'post_id' 컬럼도 이미 존재한다면, rename으로 중복이 발생하므로 먼저 삭제
                if 'post_id' in df.columns:
                    df.drop(columns=['post_id'], inplace=True)
                # 이제 'post_identifier'를 'post_id'로 안전하게 rename
                df.rename(columns={'post_identifier': 'post_id'}, inplace=True)
                
            df['post_id'] = df['post_id'].astype(str)
            return df
        except FileNotFoundError:
            print(f"오류: 마스터 데이터 파일({config.COMPETITOR_DATA_PATH})을 찾을 수 없습니다.")
            return pd.DataFrame()

    def _prepare_candidates(self, master_df: pd.DataFrame) -> pd.DataFrame:
        """
        Loads candidate IDs and merges with master data to get title, body, etc.
        """
        print("개선 대상 포스트 데이터 로딩 및 준비 중...")
        try:
            candidates_df = pd.read_csv(config.CANDIDATE_POSTS_PATH)
            candidates_df['post_id'] = candidates_df['post_id'].astype(str)

            our_posts_info = master_df[master_df['source'] == 'ours'][[
                'post_id', 'post_title', 'post_body'
            ]].copy()
            
            our_posts_info.drop_duplicates(subset='post_id', keep='first', inplace=True)

            merged_df = pd.merge(
                candidates_df,
                our_posts_info,
                on='post_id',
                how='left'
            )
            
            merged_df['post_title'].fillna('제목 없음', inplace=True)
            merged_df['post_body'].fillna('', inplace=True)

            print(f"총 {len(merged_df)}개의 개선 대상 포스트 정보를 준비했습니다.")
            return merged_df
        except FileNotFoundError:
            print(f"오류: 개선 대상 포스트 파일({config.CANDIDATE_POSTS_PATH})을 찾을 수 없습니다.")
            return pd.DataFrame()

    def _prepare_competitor_data(self, master_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts competitor data from the loaded master dataframe.
        """
        return master_df[master_df['source'] == 'competitor'].copy()

    def _setup_output_directory(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_path = os.path.join(config.OUTPUT_DATA_PATH, self.timestamp)
        os.makedirs(run_output_path, exist_ok=True)
        print(f"이번 실행의 모든 결과는 다음 폴더에 저장됩니다: {run_output_path}")
        return run_output_path

    def _setup_editor_agent(self):
        LLM = Gemini(
            id=os.getenv("GEMINI_MODEL", "gemini-1.5-pro-latest"),
            api_key=os.getenv("GEMINI_API_KEY")
        )
        return Agent(
            name="PostEditorAgent",
            role="SEO 콘텐츠 편집 전문가",
            model=LLM,
            instructions=[
                "당신은 특정 데이터 지표를 개선하기 위해 블로그 포스트를 수정하는 SEO 콘텐츠 편집 전문가입니다.",
                "현재 포스트의 제목, 본문, 그리고 개선 목표 피처와 구체적인 개선 제안이 주어집니다.",
                "당신의 임무는 제안에 따라 `post_title`과 `post_body`를 수정하는 것입니다.",
                "포스트의 핵심 주제와 메시지는 유지해야 합니다.",
                "결과는 반드시 순수한 JSON 형식으로만 반환해야 하며, 'new_title'과 'new_body' 키를 포함해야 합니다. 절대로 마크다운이나 다른 텍스트로 감싸지 마십시오.",
                "응답은 반드시 `{`로 시작하고 `}`로 끝나야 합니다.",
                "매우 중요: `new_body`의 내용이 모두 생성된 후, 반드시 마지막 줄에 `<<EOD>>` 라는 마커를 포함해야 합니다. 이것은 내용이 중간에 잘리지 않았음을 보증하는 역할을 합니다."
            ],
            show_tool_calls=False,
            markdown=False
        )

    def run_improvement_process(self):
        posts_to_improve = self.candidate_posts.head(config.NUM_POSTS_TO_IMPROVE)
        
        print(f"\n--- 총 {len(posts_to_improve)}개 포스트에 대한 개선 작업을 시작합니다 ---")

        for i, (idx, initial_post) in enumerate(posts_to_improve.iterrows()):
            post_id = initial_post['post_id']
            sanitized_post_id = sanitize_filename(str(post_id))
            print(f"\n[{i+1}/{len(posts_to_improve)}] 포스트 개선 작업 시작 (ID: {post_id})")

            # --- 파일 경로 정의 ---
            analysis_path = os.path.join(self.run_output_path, f"{sanitized_post_id}_analysis_history.csv")
            edit_history_path = os.path.join(self.run_output_path, f"{sanitized_post_id}_edit_history.json")
            
            # 1. 포스트별 분석 파일 생성 (단 1회)
            analysis_result = run_analyzer_agent(initial_post, self.feature_names_for_model)
            analysis_df = pd.DataFrame([analysis_result])
            cols_order = ['post_id', 'initial_predicted_ctr', 'diagnosis', 'suggestion'] + [f for f in self.feature_names_for_model if f in analysis_df.columns]
            analysis_df.reindex(columns=cols_order).to_csv(analysis_path, index=False, encoding='utf-8-sig')
            print(f"    -> 초기 분석 완료 및 {sanitized_post_id}_analysis_history.csv 에 기록.")
            
            # 2. 포스트별 편집 히스토리 초기화 및 '초기 파일' 생성
            current_best_post = initial_post.to_dict()
            edit_log_for_post = {
                'post_id': post_id,
                'initial': {
                    'post_title': initial_post['post_title'],
                    'post_body': initial_post['post_body'],
                    'feature_values': {f: initial_post.get(f) for f in self.feature_names_for_model},
                    'model_score': initial_post['predicted_ctr']
                }
            }
            with open(edit_history_path, 'w', encoding='utf-8') as f:
                json.dump(edit_log_for_post, f, ensure_ascii=False, indent=4, cls=NpEncoder)
            print(f"    -> 초기 상태 기록 및 {sanitized_post_id}_edit_history.json 생성.")

            # Get the relevant competitor data for this post's query
            query = initial_post.get('representative_query')
            competitors_for_query = self.competitor_data[self.competitor_data['representative_query'] == query]
            
            # 3. 피처별 순차 개선 루프
            for feature_to_improve in ALL_CTR_FEATURE_NAMES:
                print(f"  - 피처 '{feature_to_improve}' 개선 시작...")
                
                # FIX: 현재 최고 버전의 점수로 '이번 피처의 최고 시도'를 초기화합니다.
                # 이렇게 하면 점수가 음수여도 올바르게 비교 및 업데이트가 가능합니다.
                best_attempt_for_feature = {
                    'score': current_best_post.get('predicted_ctr', float('-inf')), 
                    'data': current_best_post, 
                    'features': {f: current_best_post.get(f) for f in self.feature_names_for_model}
                }
                
                for attempt in range(config.MAX_ATTEMPTS_PER_FEATURE):
                    print(f"    [시도 {attempt + 1}/{config.MAX_ATTEMPTS_PER_FEATURE}]")
                    
                    try:
                        # 편집은 항상 이전 피처 개선이 완료된 '현재 최고 버전'을 기반으로 진행합니다.
                        edited_post = run_editor_agent(current_best_post, feature_to_improve, self.editor_agent)
                        
                        new_score, new_features = run_reevaluator_agent(
                            edited_post, 
                            competitors_for_query, 
                            self.model, 
                            self.feature_names_for_model
                        )
                        
                        version_key = f"v{attempt+1}_{feature_to_improve}"
                        edit_log_for_post[version_key] = {
                            'post_title': edited_post['post_title'],
                            'post_body': edited_post['post_body'],
                            'feature_values': new_features,
                            'model_score': new_score
                        }

                        # 각 시도 후 즉시 edit_history.json 덮어쓰기
                        with open(edit_history_path, 'w', encoding='utf-8') as f:
                            json.dump(edit_log_for_post, f, ensure_ascii=False, indent=4, cls=NpEncoder)
                        
                        if new_score > best_attempt_for_feature['score']:
                            print(f"      - ⭐ 최고 점수 경신! ({best_attempt_for_feature['score']:.4f} -> {new_score:.4f})")
                            best_attempt_for_feature = {'score': new_score, 'data': edited_post, 'features': new_features}
                    
                    except Exception as e:
                        # run_editor_agent에서 발생한 오류를 여기서 잡아서, 현재 시도를 중단하고 다음 시도로 넘어감
                        print(f"      [경고] 시도 {attempt + 1} 실패. 다음 시도를 진행합니다. (오류: {e})")
                        # 실패한 시도에 대한 로그를 남길 수도 있음 (선택적)
                        version_key = f"v{attempt+1}_{feature_to_improve}_FAILED"
                        edit_log_for_post[version_key] = {
                            'post_title': "오류로 인해 생성 실패",
                            'post_body': f"오류: {e}",
                            'feature_values': {},
                            'model_score': -1
                        }
                        with open(edit_history_path, 'w', encoding='utf-8') as f:
                            json.dump(edit_log_for_post, f, ensure_ascii=False, indent=4, cls=NpEncoder)
                        continue # 다음 시도로 넘어감

                # 현재 피처에 대한 최고 버전으로 업데이트합니다.
                current_best_post = best_attempt_for_feature['data']
                current_best_post.update(best_attempt_for_feature['features'])
                # 다음 루프에서 점수 비교를 위해 'predicted_ctr' 키도 업데이트 해줍니다.
                current_best_post['predicted_ctr'] = best_attempt_for_feature['score']
                print(f"  - 피처 '{feature_to_improve}' 개선 완료. 최종 선택된 버전의 점수: {best_attempt_for_feature['score']:.4f}")
            
            # 4. 포스트 작업 완료 로그
            print(f"-> [{i+1}/{len(posts_to_improve)}] 포스트 작업 완료. 최종 결과가 {sanitized_post_id}_edit_history.json 에 저장되었습니다.")
            
        print("\n--- 모든 작업이 성공적으로 완료되었습니다. ---")

def main():
    orchestrator = PostImprovementOrchestrator()
    orchestrator.run_improvement_process()

if __name__ == "__main__":
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", FutureWarning)
        main() 