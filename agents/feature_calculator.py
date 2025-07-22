import os
import pandas as pd
import json
import re
import traceback
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.google import Gemini
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch

# --- 1. 설정 및 상수 정의 ---
# .env 파일에서 환경 변수 로드 (blog_automation/.env)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# --- agno.py를 참고하여 LLM 및 에이전트 정의 ---
LLM = Gemini(
    id=os.getenv("GEMINI_MODEL", "gemini-2.5-pro"),
    api_key=os.getenv("GEMINI_API_KEY")
)

code_debugger_agent = Agent(
    name="CodeDebuggerAgent",
    role="Python Code Debugging Expert",
    model=LLM,
    instructions=[
        "You are an expert Python code debugger.",
        "You will be given a Python script that failed to run, along with the error traceback, and the intended feature name.",
        "The original code was supposed to create a new column in a pandas DataFrame called 'df'.",
        "Your task is to analyze the error and provide a corrected, complete Python script.",
        "Your output MUST be a single, raw JSON object and nothing else. Do not wrap it in markdown.",
        "The JSON must contain one key: `corrected_code` (string).",
        "The value of `corrected_code` must be a string containing the full, corrected Python script, including all necessary imports.",
    ],
    show_tool_calls=False,
    markdown=False,
)

# --- 새로운 피처 명세서 ---
# 각 피처에 대해 가장 성공적인 버전이 있었던 소스 파일을 명시적으로 지정
# target_metric: 'ctr' 또는 'inflow'
# history_file: p-value를 확인할 history 파일
# code_log_file: 실제 코드를 가져올 code_log 파일
FEATURE_SPECIFICATIONS = [
    # --- CTR Features ---
    {
        "feature_name": "relative_query_fulfillment_score", 
        "target_metric": "ctr",
        "history_file": "P3_Benchmark_CTR_Query_Relation_history.json",
        "code_log_file": "P3_Benchmark_CTR_Query_Relation_code_log.json"
    },
    {
        "feature_name": "relative_semantic_actionability", 
        "target_metric": "ctr",
        "history_file": "P2_Benchmark_CTR_Semantic_history.json",
        "code_log_file": "P2_Benchmark_CTR_Semantic_code_log.json"
    },
    {
        "feature_name": "title_body_semantic_cohesion", 
        "target_metric": "ctr",
        "history_file": "P3_Benchmark_CTR_TitleBody_Relation_history.json",
        "code_log_file": "P3_Benchmark_CTR_TitleBody_Relation_code_log.json"
    },
    {
        "feature_name": "title_hook_pattern_presence", 
        "target_metric": "ctr",
        "history_file": "P1_Internal_CTR_All_Intrinsic_history.json",
        "code_log_file": "P1_Internal_CTR_All_Intrinsic_code_log.json"
    },
    # --- Inflow Features ---
    {
        "feature_name": "semantic_trajectory_score", 
        "target_metric": "inflow",
        "history_file": "B그룹_본문Semantic분석_history.json",
        "code_log_file": "B그룹_본문Semantic분석_code_log.json" # Assume this exists
    },
    {
        "feature_name": "archetypal_purity_score", 
        "target_metric": "inflow",
        "history_file": "B그룹_본문Semantic분석_history.json",
        "code_log_file": "B그룹_본문Semantic분석_code_log.json"
    },
    {
        "feature_name": "sub_topic_contrast_score", 
        "target_metric": "inflow",
        "history_file": "B그룹_본문Semantic분석_history.json",
        "code_log_file": "B그룹_본문Semantic분석_code_log.json"
    },
    {
        "feature_name": "semantic_escape_velocity_score", 
        "target_metric": "inflow",
        "history_file": "B그룹_본문Semantic분석_history.json",
        "code_log_file": "B그룹_본문Semantic분석_code_log.json"
    },
    {
        "feature_name": "semantic_contrast_score", 
        "target_metric": "inflow",
        "history_file": "B그룹_본문Semantic분석_history.json",
        "code_log_file": "B그룹_본문Semantic분석_code_log.json"
    },
    {
        "feature_name": "numeric_data_density", 
        "target_metric": "inflow",
        "history_file": "P1_Internal_Inflow_All_Intrinsic_history.json",
        "code_log_file": "P1_Internal_Inflow_All_Intrinsic_code_log.json"
    },
]


FEATURE_ANALYSIS_PATH = "/Users/min/codes/medilawyer_sales/blog_automation/data/modeling/feature_analysis"
OUTPUT_PATH = "/Users/min/codes/medilawyer_sales/blog_automation/data/modeling/feature_calculate"
MASTER_DF_PATH = "/Users/min/codes/medilawyer_sales/blog_automation/data/data_processed/master_post_data.csv"
MAX_DEBUG_ATTEMPTS = 3

# 이전에 선정한 최종 피처 후보군 목록 - 폐기하고 FEATURE_SPECIFICATIONS 사용
# CTR_FEATURE_LIST = [ ... ]
# INFLOW_FEATURE_LIST = [ ... ]

# --- 모델 초기화 ---
_semantic_model = None

def get_semantic_model():
    """Lazy-loads and returns a SentenceTransformer model."""
    global _semantic_model
    if _semantic_model is None:
        _semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return _semantic_model

# --- 2. 핵심 함수 (수정) ---

def find_best_feature_code(feature_spec: dict) -> str:
    """
    주어진 명세에 따라 가장 성공적인 버전의 피처 코드를 찾습니다.
    1. history 파일에서 p-value가 가장 낮은 버전을 찾습니다.
    2. 해당 버전의 hypothesis와 의미론적으로 가장 유사한 코드를 code_log에서 추출합니다.
    """
    feature_name = feature_spec['feature_name']
    target_metric_key = 'non_brand_average_ctr' if feature_spec['target_metric'] == 'ctr' else 'non_brand_inflow'
    history_filepath = os.path.join(FEATURE_ANALYSIS_PATH, feature_spec['history_file'])
    code_log_filepath = os.path.join(FEATURE_ANALYSIS_PATH, feature_spec['code_log_file'])

    print(f"[{feature_name}] 최적 코드 탐색 중 (소스: {feature_spec['history_file']})...")

    best_hypothesis = None
    min_p_value = float('inf')

    try:
        # Step 1: history 파일에서 최고의 가설(hypothesis) 찾기
        with open(history_filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # 이미 유효한 JSON 배열(리스트) 형식인지 확인
            if content.startswith('[') and content.endswith(']'):
                history_data_str = content
            else: # 여러 JSON 객체가 연속된 형태일 경우
                history_data_str = f"[{content.replace('}{', '},{')}]"
            
            history_data = json.loads(history_data_str)

        for entry in history_data:
            # 다양한 history 파일 구조에 대응
            current_feature_name = entry.get('feature_name') or entry.get('feature_created')
            
            if current_feature_name == feature_name:
                results = entry.get('correlation_results', {})
                
                # B그룹 파일처럼 target_metric이 키로 바로 오는 경우
                if target_metric_key in results:
                    metric_result = results.get(target_metric_key, {})
                    p_value = metric_result.get('p_value')
                # P그룹 파일처럼 target_metric 없이 바로 correlation, p_value가 오는 경우
                else:
                    p_value = results.get('p_value')

                if p_value is not None and p_value < min_p_value:
                    min_p_value = p_value
                    best_hypothesis = entry.get('hypothesis')

    except FileNotFoundError:
        print(f"  - 경고: History 파일 없음: {history_filepath}")
        return None
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  - 경고: History 파일 처리 중 오류: {e}")
        return None

    if not best_hypothesis:
        print(f"  - 경고: [{feature_name}]에 대한 유의미한 history를 찾지 못했습니다.")
        return None

    print(f"  - 최적 버전 발견 (p-value: {min_p_value:.4f}). 의미론적으로 가장 유사한 코드 탐색...")

    # Step 2: code_log 파일에서 의미론적으로 가장 유사한 코드 찾기
    try:
        with open(code_log_filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('[') and content.endswith(']'):
                code_log_str = content
            else:
                code_log_str = f"[{content.replace('}{', '},{')}]"
            
            code_log_data = json.loads(code_log_str)
        
        # code_log에서 해당 피처 이름을 가진 모든 가설과 코드 추출
        candidate_hypotheses = []
        candidate_codes = []
        for entry in code_log_data:
            current_feature_name = entry.get('feature_name') or entry.get('feature_created')
            if current_feature_name == feature_name and entry.get('hypothesis') and entry.get('code'):
                candidate_hypotheses.append(entry['hypothesis'])
                candidate_codes.append(entry['code'])

        if not candidate_hypotheses:
            print(f"  - 경고: Code log에 [{feature_name}]에 대한 유효한 항목이 없습니다.")
            return None
            
        # 의미론적 유사도 계산
        model = get_semantic_model()
        best_hypothesis_embedding = model.encode(best_hypothesis, convert_to_tensor=True, show_progress_bar=False)
        candidate_embeddings = model.encode(candidate_hypotheses, convert_to_tensor=True, show_progress_bar=False)
        
        similarities = util.cos_sim(best_hypothesis_embedding, candidate_embeddings)
        best_match_index = similarities.argmax()
        
        print(f"  - 가장 유사한 가설 매칭 (유사도: {similarities[0][best_match_index]:.4f}). 코드 발견!")
        return candidate_codes[best_match_index]

    except FileNotFoundError:
        print(f"  - 경고: Code log 파일 없음: {code_log_filepath}")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  - 경고: Code log 파일 처리 중 오류: {e}")

    print(f"  - 경고: 최적 버전에 해당하는 코드를 code_log에서 찾지 못했습니다.")
    return None


def execute_with_ai_debugger(code_to_run: str, feature_name: str, current_df: pd.DataFrame) -> pd.DataFrame:
    """
    주어진 코드를 동적으로 실행하고, 오류 발생 시 AI 디버거를 통해 수정합니다.
    수정된 핵심: exec 실행 시 필요한 모듈과 함수를 담은 네임스페이스를 명시적으로 전달합니다.
    """
    max_attempts = MAX_DEBUG_ATTEMPTS
    for attempt in range(1, max_attempts + 1):
        try:
            # --- 네임스페이스 설정 (핵심 수정사항) ---
            # exec로 실행될 코드에 필요한 모든 라이브러리와 헬퍼 함수를 딕셔너리로 준비
            execution_globals = {
                'pd': pd,
                'np': np,
                'torch': torch,
                'util': util,
                'SentenceTransformer': SentenceTransformer,
                'get_model': get_semantic_model, # Lazy-loading 모델 함수 포함
                '__builtins__': __builtins__ # 기본 내장 함수 포함
            }
            local_namespace = {'df': current_df.copy()} # 입력 데이터프레임

            # exec 함수에 global과 local 네임스페이스를 명시적으로 전달
            exec(code_to_run, execution_globals, local_namespace)
            
            # 함수 실행 후 변경된 데이터프레임을 반환
            result_df = local_namespace.get('df')

            if result_df is None or feature_name not in result_df.columns:
                error_message = f"코드는 오류 없이 실행되었으나, '{feature_name}' 컬럼이 생성되지 않았습니다."
                raise ValueError(error_message)

            print(f"  - [{feature_name}] 코드 실행 성공.")
            return result_df

        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"  - 시도 {attempt}/{max_attempts}: [{feature_name}] 코드 실행 중 오류 발생.")
            # print(error_traceback) # Keep log clean, only print if needed for deep debug

            if attempt >= max_attempts -1:
                print(f"  - 실패: [{feature_name}] 최대 재시도 횟수 초과. 이 피처를 건너뜁니다.")
                # 원본 데이터프레임을 그대로 반환
                return current_df

            # --- AI 디버거 호출 ---
            print("  - AI 디버거를 호출하여 코드 수정을 시도합니다...")
            
            # 오류 유형에 따라 프롬프트 분기
            if isinstance(e, ValueError) and "컬럼이 생성되지 않았습니다" in str(e):
                # '조용한 실패'에 대한 프롬프트
                prompt = f"""
                Your task is to debug a Python script that failed to create a new column named '{feature_name}' in a pandas DataFrame 'df'.

                --- Analysis of the Failure ---
                This is a critical logical error. The code completed execution without a syntax or runtime error, but it FAILED to create the expected output column ('{feature_name}').
                This 'silent failure' often happens if a condition (e.g., an `if` statement) causes the column creation logic to be skipped for all rows, or if the column is created on a copy of the dataframe that is then discarded.

                --- Your Goal ---
                Modify the code to ensure that the '{feature_name}' column is ALWAYS created and added to the returned DataFrame.
                If a value cannot be calculated for a specific row, it must be filled with a sensible default value (e.g., 0, 0.0, or np.nan). The column's existence is mandatory.

                --- Failed Code ---
                {code_to_run}

                Please provide the full, corrected Python script in the required JSON format.
                """
            else:
                # 일반적인 실행 오류에 대한 프롬프트
                prompt = f"""
                Your task is to debug a Python script that failed to create a new column named '{feature_name}' in a pandas DataFrame 'df'.

                --- Analysis of the Failure ---
                The script failed with the following error traceback:
                {error_traceback}

                --- Your Goal ---
                Analyze the root cause of this error and provide a corrected version of the full Python script.

                --- Failed Code ---
                {code_to_run}

                Please provide the full, corrected Python script in the required JSON format.
                """

            try:
                # agno 에이전트를 사용하여 코드 수정
                agent_response_raw = code_debugger_agent.run(prompt).content
                
                # JSON 응답 파싱 (안정성 강화)
                if not agent_response_raw or not agent_response_raw.strip().startswith('{'):
                    print("  - 경고: AI가 유효한 JSON 응답을 반환하지 않았습니다. (응답이 비었거나 JSON 형식이 아님)")
                    corrected_code = None
                else:
                    response_json = json.loads(agent_response_raw)
                    corrected_code = response_json.get('corrected_code')

                if corrected_code:
                    code_to_run = corrected_code
                    print("  - AI가 코드를 수정했습니다. 재시도합니다.")
                else:
                    print("  - 경고: AI의 응답에서 'corrected_code'를 찾지 못했습니다.")

            except (json.JSONDecodeError, AttributeError) as ai_error:
                print(f"  - 실패: AI 디버거 응답 처리 중 오류 발생: {ai_error}")
                print(f"  - 받은 응답: {agent_response_raw}")
                # AI 호출 실패 시 루프를 중단하고 다음 피처로 넘어감
                return current_df
                
    # 모든 시도가 실패한 경우
    return current_df


# --- 3. 메인 컨트롤 타워 ---

def main():
    """
    전체 피처 계산 및 검증 과정을 총괄하는 메인 함수
    """
    print("피처 재계산 및 검증 에이전트를 시작합니다.")

    # 데이터 로딩
    print(f"마스터 데이터프레임 로딩: {MASTER_DF_PATH}")
    if not os.path.exists(MASTER_DF_PATH):
        print(f"오류: 마스터 데이터프레임 파일을 찾을 수 없습니다. 경로: {MASTER_DF_PATH}")
        return
    master_df = pd.read_csv(MASTER_DF_PATH)
    print("  - 마스터 데이터프레임 컬럼:", master_df.columns.tolist())

    # 출력 폴더 생성
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # 사용할 식별자 컬럼 목록 (KeyError 방지를 위해 실제 존재하는 컬럼으로 구성)
    identifier_cols = ['post_identifier', 'source', 'top_category_name', 'representative_query']
    
    # master_df에 해당 컬럼들이 모두 존재하는지 확인하고, 없으면 예외 처리
    missing_cols = [col for col in identifier_cols if col not in master_df.columns]
    if missing_cols:
        print(f"오류: 마스터 데이터프레임에 필수 식별자 컬럼이 없습니다: {missing_cols}")
        return

    # --- CTR 피처 계산 ---
    print("\n--- CTR 피처 계산 시작 ---")
    # 계산된 피처를 저장할 초기 데이터프레임 (식별자 포함)
    ctr_results_df = master_df[identifier_cols].copy()
    
    temp_ctr_df = master_df.copy()
    for spec in FEATURE_SPECIFICATIONS:
        if spec['target_metric'] != 'ctr': continue
        
        feature_name = spec['feature_name']
        code = find_best_feature_code(spec)
        if not code:
            continue
        # AI 디버거는 전체 df를 받아 피처를 계산하고, 해당 피처가 추가된 df를 반환
        temp_ctr_df_updated = execute_with_ai_debugger(code, feature_name, temp_ctr_df)
        
        # 피처가 성공적으로 추가되었는지 확인하고 결과 DF에 추가
        if feature_name in temp_ctr_df_updated.columns:
            ctr_results_df[feature_name] = temp_ctr_df_updated[feature_name]
            temp_ctr_df = temp_ctr_df_updated # 다음 계산을 위해 업데이트된 DF 유지
    
    ctr_output_file = os.path.join(OUTPUT_PATH, "ctr_feature_value.csv")
    print(f"CTR 피처 계산 완료. 파일 저장: {ctr_output_file}")
    ctr_results_df.to_csv(ctr_output_file, index=False, encoding='utf-8-sig')

    # --- Inflow 피처 계산 ---
    print("\n--- Inflow 피처 계산 시작 ---")
    inflow_results_df = master_df[identifier_cols].copy()
    
    temp_inflow_df = master_df.copy()
    for spec in FEATURE_SPECIFICATIONS:
        if spec['target_metric'] != 'inflow': continue

        feature_name = spec['feature_name']
        code = find_best_feature_code(spec)
        if not code:
            continue
        temp_inflow_df_updated = execute_with_ai_debugger(code, feature_name, temp_inflow_df)

        if feature_name in temp_inflow_df_updated.columns:
            inflow_results_df[feature_name] = temp_inflow_df_updated[feature_name]
            temp_inflow_df = temp_inflow_df_updated # 다음 계산을 위해 업데이트된 DF 유지

    inflow_output_file = os.path.join(OUTPUT_PATH, "inflow_feature_value.csv")
    print(f"Inflow 피처 계산 완료. 파일 저장: {inflow_output_file}")
    inflow_results_df.to_csv(inflow_output_file, index=False, encoding='utf-8-sig')

    print("\n모든 작업이 완료되었습니다.")


if __name__ == "__main__":
    main() 