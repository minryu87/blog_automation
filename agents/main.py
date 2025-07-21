import os
import sys
import json
import logging
import re
from dotenv import load_dotenv
import pandas as pd

from scipy.stats import pearsonr
import numpy as np
from datetime import datetime
import traceback
import subprocess

# --- Configure logging ---
logging.basicConfig(level=logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- Local Imports ---
from . import config
from .team import feature_engineer_agent, failure_analysis_agent # MODIFIED: Import instances directly
from .tools import HistoryTool, HumanFeedbackTool, LogFileTool
from scipy.stats import pearsonr
from aistudio.tools.code_writer_tool import CodeWriterTool
from aistudio.tools.history_tool import HistoryTool

# --- 0. Load Environment Variables ---
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# --- 1. Meta-Orchestrator Configuration ---
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# --- Constants and Configuration ---
# Load settings from the config file
GOAL = config.GOAL
MAX_ITERATIONS_PER_TASK = config.MAX_ITERATIONS_PER_TASK
MAX_CORRECTION_ATTEMPTS = config.MAX_CORRECTION_ATTEMPTS
EXPERIMENT_PIPELINE = config.EXPERIMENT_PIPELINE
TASK_FILE_NAMES = config.TASK_FILE_NAMES # NEW: Load the file names

# Get the absolute path to the agents directory
AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATASET_PATH = os.path.join(os.path.dirname(AGENTS_DIR), 'data', 'data_processed', 'master_post_data.csv')
FEEDBACK_FILE_PATH = os.path.join(AGENTS_DIR, 'feedback.md') # NEW: Define absolute path

# --- Agent Initialization ---
# No longer needed as we import instances directly from team.py
# feature_engineer_agent = FeatureEngineerAgent()
# failure_analysis_agent = FailureAnalysisAgent()

# --- Main Functions ---
def extract_json_from_string(text: str) -> dict | None:
    """문자열에서 첫 번째 JSON 객체를 추출하여 딕셔너리로 반환합니다."""
    if not isinstance(text, str):
        return None
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        start_index = text.find('{')
        end_index = text.rfind('}')
        if start_index != -1 and end_index != -1 and start_index < end_index:
            json_str = text[start_index:end_index+1].strip()
        else:
            return None
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

def _install_and_retry(error_output: str, script_path: str, input_csv: str, output_csv: str) -> tuple[pd.DataFrame | None, str | None]:
    """
    ModuleNotFoundError가 발생할 경우 패키지를 설치하고 다시 실행하려고 시도합니다.
    """
    # 누락된 모듈 이름을 찾기 위한 정규식
    match = re.search(r"ModuleNotFoundError: No module named '(\w+)'", error_output)
    if not match:
        return None, error_output # ModuleNotFoundError가 아니면 원래 오류 반환

    library_name = match.group(1)
    print(f"[Self-Healing] 누락된 라이브러리 발견: '{library_name}'. 설치를 시도합니다...")
    
    try:
        # 라이브러리 설치 시도
        install_result = subprocess.run(
            [sys.executable, "-m", "pip", "install", library_name],
            capture_output=True, text=True, check=True, encoding='utf-8'
        )
        print(f"[Self-Healing] '{library_name}' 성공적으로 설치되었습니다. 스크립트 실행을 다시 시도합니다...")

        # 스크립트 다시 실행
        retry_result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, encoding='utf-8', check=False
        )
        
        if retry_result.returncode != 0:
            return None, retry_result.stderr # 설치 후에도 실패

        if os.path.exists(output_csv):
            modified_df = pd.read_csv(output_csv)
            return modified_df, None # 재시도 성공
        else:
            return None, "재시도 후에도 출력 파일이 생성되지 않았습니다."

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        install_error = e.stderr if hasattr(e, 'stderr') else str(e)
        return None, f"'{library_name}' 설치에 실패했습니다. Pip 오류: {install_error}"
    except Exception as e:
        return None, f"예상치 못한 오류가 발생했습니다: {str(e)}"

def execute_feature_code(feature_code: str, base_df: pd.DataFrame, execution_mode: str = 'dataframe') -> tuple[pd.DataFrame | float | None, str | None]:
    """
    생성된 파이썬 코드를 격리된 서브프로세스에서 실행합니다.
    'dataframe' 모드: 전체 DataFrame을 처리하고 수정된 DataFrame을 반환합니다.
    'comparison' 모드: 우리 포스트(series)와 경쟁자 DataFrame을 받아 단일 숫자 값을 반환합니다.
    """
    temp_script_path = os.path.join(AGENTS_DIR, '_temp_feature_generator.py')
    temp_input_csv_path = os.path.join(AGENTS_DIR, '_temp_input.csv')
    temp_output_path = os.path.join(AGENTS_DIR, '_temp_output.json') # JSON으로 결과 반환

    script_content = f"""
import pandas as pd
import numpy as np
import sys
import os
from sentence_transformers import SentenceTransformer, util
import torch
import re
import json

# --- Generated Feature Code ---
{feature_code}
# --- Execution Logic ---
def run():
    input_path = '{temp_input_csv_path}'
    output_path = '{temp_output_path}'
    try:
        df = pd.read_csv(input_path)
        result_data = {{}}
        if '{execution_mode}' == 'dataframe':
        func_name = next((name for name in globals() if name.startswith('generate_feature')), None)
        if not func_name:
                raise ValueError("generate_feature로 시작하는 함수를 찾을 수 없습니다.")
        generate_func = globals()[func_name]
        modified_df = generate_func(df)
            # DataFrame 모드에서는 결과 CSV를 별도로 저장
            modified_df.to_csv('{temp_input_csv_path.replace('.csv', '_modified.csv')}', index=False)
        elif '{execution_mode}' == 'comparison':
            func_name = next((name for name in globals() if name.startswith('generate_comparison_feature')), None)
            if not func_name:
                raise ValueError("generate_comparison_feature로 시작하는 함수를 찾을 수 없습니다.")
            generate_func = globals()[func_name]
            our_post = df[df['source'] == 'ours'].iloc[0]
            competitors = df[df['source'] == 'competitors']
            feature_value = generate_func(our_post, competitors)
            result_data['feature_value'] = feature_value

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f)

    except Exception as e:
        print(f"스크립트 실행 중 오류 발생: {{e}}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    run()
"""

    modified_df_path = temp_input_csv_path.replace('.csv', '_modified.csv')

    try:
        with open(temp_script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)

        base_df.to_csv(temp_input_csv_path, index=False)

        result = subprocess.run(
            [sys.executable, temp_script_path],
            capture_output=True, text=True, encoding='utf-8', check=False
        )

        if result.returncode != 0:
            if "ModuleNotFoundError" in result.stderr:
                # Self-healing은 DataFrame 모드에서만 의미가 있으므로, 여기서는 간단히 처리
                return None, result.stderr
            return None, result.stderr

        if execution_mode == 'dataframe':
            if os.path.exists(modified_df_path):
                return pd.read_csv(modified_df_path), None
            else:
                return None, "DataFrame 모드에서 출력 파일이 생성되지 않았습니다."
        elif execution_mode == 'comparison':
            if os.path.exists(temp_output_path):
                with open(temp_output_path, 'r', encoding='utf-8') as f:
                    output_json = json.load(f)
                return output_json.get('feature_value'), None
        else:
                return None, "Comparison 모드에서 출력 JSON 파일이 생성되지 않았습니다."

    except Exception as e:
        return None, traceback.format_exc()
    finally:
        for path in [temp_script_path, temp_input_csv_path, temp_output_path, modified_df_path]:
            if os.path.exists(path):
                os.remove(path)

def interpret_correlation(correlation, p_value):
    """상관관계 분석 결과의 통계적 유의미성과 의미를 해석합니다."""
    if correlation is None or p_value is None:
        return "상관관계를 계산할 수 없습니다. 이는 보통 생성된 피처의 분산이 0일 때 (모든 값이 동일할 때) 발생합니다."

    # 유의 수준
    alpha = 0.05
    is_significant = p_value < alpha

    # 상관관계 강도
    abs_corr = abs(correlation)
    if abs_corr >= 0.7:
        strength = "매우 강한"
    elif abs_corr >= 0.5:
        strength = "강한"
    elif abs_corr >= 0.3:
        strength = "중간 정도의"
    else:
        strength = "약한"

    # 방향성
    direction = "양의" if correlation > 0 else "음의"

    # 결론 문장
    conclusion = f"{strength} {direction} 상관관계({correlation:.4f})를 발견했습니다. "
    if is_significant:
        conclusion += f"이 결과는 통계적으로 유의미하며(p-value: {p_value:.4f}), 이 관계가 우연에 의한 것일 가능성은 낮습니다."
    else:
        conclusion += f"하지만 이 결과는 통계적으로 유의미하지 않으므로(p-value: {p_value:.4f}), 우연에 의한 결과일 가능성을 배제할 수 없습니다."
        
    return conclusion

def analyze_correlation(df: pd.DataFrame, feature_name: str, target_metrics: list = None) -> dict:
    """지정된 타겟 지표에 대해 피처의 상관관계를 분석합니다."""
    if target_metrics is None:
        target_metrics = ['non_brand_inflow', 'non_brand_average_ctr']

    report = {
        "correlation_results": {
            "non_brand_inflow": {"correlation": None, "p_value": None, "interpretation": ""},
            "non_brand_average_ctr": {"correlation": None, "p_value": None, "interpretation": ""},
        },
        "overall_conclusion": "분석에 실패했습니다."
    }
    try:
        if feature_name not in df.columns:
            raise ValueError(f"실행 후 데이터프레임에서 '{feature_name}' 피처를 찾을 수 없습니다.")

        for target in target_metrics:
            cleaned_df = df[[feature_name, target]].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(cleaned_df) > 1 and pd.api.types.is_numeric_dtype(cleaned_df[feature_name]) and cleaned_df[feature_name].nunique() > 1:
                corr, p_value = pearsonr(cleaned_df[feature_name], cleaned_df[target])
                report["correlation_results"][target]["correlation"] = None if np.isnan(corr) else corr
                report["correlation_results"][target]["p_value"] = None if np.isnan(p_value) else p_value
            else:
                # 상관관계를 계산할 수 없는 경우
                corr, p_value = None, None
            
            interpretation = interpret_correlation(corr, p_value)
            report["correlation_results"][target]["interpretation"] = interpretation

        report["overall_conclusion"] = "분석이 완료되었습니다. 각 타겟에 대한 해석을 참고하세요."
    except Exception as e:
        report["overall_conclusion"] = f"상관관계 분석 중 오류 발생: {str(e)}"
    return report

def analyze_and_log_failure(error_traceback: str, code: str, hypothesis: str):
    """LLM 에이전트를 사용하여 논리적 실패의 근본 원인을 분석하고 기록합니다."""
    print("[실패 분석] FailureAnalysisAgent를 사용하여 근본 원인 분석 중...")
    
    analysis_prompt = (
        f"다음 실패한 코드 실행을 분석하여 근본적인 논리적 원인을 찾아내세요.\n\n"
        f"--- 가설 ---\n{hypothesis}\n\n"
        f"--- 실패한 코드 ---\n{code}\n\n"
        f"--- 에러 트레이스백 ---\n{error_traceback}\n\n"
        "이 정보를 바탕으로, 개발자의 접근 방식에 있었던 근본적인 논리적 오류는 무엇이었습니까? "
        "반드시 한국어로 답변해주세요."
    )
    
    try:
        response_raw = failure_analysis_agent.run(analysis_prompt).content
        response_json = extract_json_from_string(response_raw)
        failure_reason = response_json.get("failure_reason", "알 수 없는 논리적 오류")
    except Exception as e:
        print(f"[실패 분석] 에이전트가 오류를 분석하는 데 실패했습니다: {e}")
        failure_reason = "알 수 없는 논리적 오류: 분석 에이전트 자체에서 오류가 발생했습니다."

    # 요약된 실패 원인을 파일에 기록
    log_path = os.path.join(AGENTS_DIR, 'logical_failure_history.log')
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"[{datetime.now().isoformat()}] {failure_reason}\n")
    print(f"[실패 분석] 논리적 실패 기록 완료: {failure_reason}")


def get_history_summary(history_tool: HistoryTool) -> str:
    """
    Reads the history file using the provided tool instance and summarizes the previously failed features.
    """
    failed_features = []
    try:
        # Use the provided, correctly initialized history_tool instance
        history_data = history_tool.read_history()
        for item in history_data:
            # Check for failure status more robustly
            if item.get("status") == "failed":
                failed_features.append(item.get("feature_created"))
    except Exception as e:
        # Provide more context on error
        return f"이전 기록을 요약하는 중 오류 발생 ({type(e).__name__}): {e}"

    if not failed_features:
        return "이전 실패 기록이 없습니다. 새로운 가설을 자유롭게 시도하세요."
    
    return f"총 {len(failed_features)}개의 이전 실패 사례가 있습니다: {', '.join(filter(None, failed_features))}"


def get_task_feedback(task_number: int) -> str:
    """
    Reads feedback.md and extracts the specific instructions for a given task number,
    handling grouped tasks like [작업 3 & 4].
    """
    try:
        with open(FEEDBACK_FILE_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        return f"오류: 피드백 파일({FEEDBACK_FILE_PATH})을 찾을 수 없습니다."

    # Use regex to find the specific task block for a single task number
    task_pattern = re.compile(
        rf"### \[작업 {task_number}\]:(.*?)(?=\n### \[작업|\Z)",
        re.DOTALL | re.IGNORECASE
    )
    match = task_pattern.search(content)

    if not match:
        return f"오류: 피드백 파일에서 [작업 {task_number}]에 대한 지침을 찾을 수 없습니다."

    return match.group(0).strip()

def get_task_metadata(task_feedback: str) -> dict | None:
    """
    Parses the task feedback to extract structured metadata from an HTML comment.
    
    Returns a dictionary with 'task_type' and 'target_metric', or None if parsing fails.
    """
    metadata_match = re.search(r"<!--(.*?)-->", task_feedback, re.DOTALL)
    if not metadata_match:
        return None

    metadata_text = metadata_match.group(1)
    
    try:
        task_type = re.search(r"task_type:\s*(\w+)", metadata_text).group(1)
        target_metric = re.search(r"target_metric:\s*(\w+)", metadata_text).group(1)
        return {"task_type": task_type, "target_metric": target_metric}
    except (AttributeError, IndexError):
        return None

def execute_generated_code(code_string: str, df: pd.DataFrame, feature_name: str, task_type: str) -> tuple[pd.DataFrame | None, str | None]:
    """
    AI가 생성한 코드를 안전하게 실행하고, 그 결과로 피처가 DataFrame에 추가되었는지 검증합니다.
    성공 시 (수정된 DataFrame, None)을, 실패 시 (None, 오류 메시지)를 반환합니다.
    """
    try:
        # 필요한 라이브러리와 함께 코드 실행 컨텍스트 준비
        G = {
            'pd': pd,
            'np': np,
            'torch': __import__('torch'),
            'SentenceTransformer': __import__('sentence_transformers').SentenceTransformer,
            'util': __import__('sentence_transformers').util
        }
        
        # AI가 생성한 코드를 실행하여 'generate_feature' 또는 'generate_comparison_feature' 함수를 정의
        exec(code_string, G)
        
        # 태스크 종류에 따라 AI가 생성해야 할 함수 이름 결정
        func_name = "generate_comparison_feature" if task_type in ["PART_2", "PART_3"] else "generate_feature"
        
        if func_name in G:
            feature_function = G[func_name]
            # 데이터프레임의 복사본에 함수를 적용하여 원본 데이터 보호
            df_modified = feature_function(df.copy())

            # 핵심 검증 단계: 기대한 피처가 실제로 생성되었는지 확인
            if feature_name not in df_modified.columns:
                logical_error = f"Logical Error: The function '{func_name}' ran without syntax errors, but the expected feature column '{feature_name}' was not found in the resulting DataFrame. Please check your code to ensure the new column is correctly assigned."
                return None, logical_error
            
            return df_modified, None  # 성공
        else:
            return None, f"Execution Error: The generated code did not define the expected function '{func_name}'."
            
    except Exception:
        # 코드 실행 중 발생한 모든 구문 또는 런타임 오류 포착
        return None, traceback.format_exc()

def run_correlation_analysis_and_update_history(
    df: pd.DataFrame, 
    feature_name: str, 
    hypothesis: str, 
    target_metric: str, # Use the specific target_metric for this task
    history_tool: HistoryTool
):
    """
    지정된 단일 목표 지표에 대해 상관관계 분석을 수행하고 history.json을 업데이트합니다.
    """
    if feature_name not in df.columns:
        error_message = f"실행 후 데이터프레임에서 '{feature_name}' 피처를 찾을 수 없습니다."
        print(f"[상관관계 분석 오류] {error_message}")
        history_tool.add_event({
            "feature_created": feature_name,
            "hypothesis": hypothesis,
            "conclusion": f"상관관계 분석 중 오류 발생: {error_message}",
            "status": "error",
            "correlation_results": {
                target_metric: {"correlation": None, "p_value": None, "interpretation": ""}
            }
        })
        return

    correlation_results = {}
    
    # 이제 단일 target_metric에 대해서만 분석을 수행합니다.
    df_clean = df[[feature_name, target_metric]].dropna()
    
    if len(df_clean) < 2:
        interpretation = "데이터 부족으로 상관관계를 계산할 수 없습니다."
        correlation = None
        p_value = None
    else:
        correlation, p_value = pearsonr(df_clean[feature_name], df_clean[target_metric])
        if p_value < 0.05:
            if correlation > 0.2:
                interpretation = "통계적으로 유의미한 약한 양의 상관관계"
            elif correlation > 0.5:
                interpretation = "통계적으로 유의미한 중간 정도의 양의 상관관계"
            elif correlation > 0.8:
                interpretation = "통계적으로 유의미한 강한 양의 상관관계"
            elif correlation < -0.2:
                interpretation = "통계적으로 유의미한 약한 음의 상관관계"
            elif correlation < -0.5:
                interpretation = "통계적으로 유의미한 중간 정도의 음의 상관관계"
            elif correlation < -0.8:
                interpretation = "통계적으로 유의미한 강한 음의 상관관계"
            else:
                interpretation = "통계적으로 유의미하지만, 상관관계는 미미함"
        else:
            interpretation = "통계적으로 유의미하지 않은 상관관계"

    correlation_results[target_metric] = {
        "correlation": correlation,
        "p_value": p_value,
        "interpretation": interpretation
    }

    conclusion = f"피처 '{feature_name}'와(과) '{target_metric}' 간의 상관관계는 {correlation:.4f} (p-value: {p_value:.4f})이며, 이는 '{interpretation}'로 해석됩니다."

    history_tool.add_event({
        "feature_created": feature_name,
        "hypothesis": hypothesis,
        "conclusion": conclusion,
        "status": "success",
        "correlation_results": correlation_results
    })
    print(f"상관관계 분석 완료: {conclusion}")

def run_meta_orchestrator():
    """
    새로운 3단계 프레임워크에 따라 피처 엔지니어링 프로세스를 총괄하는 메인 오케스트레이터입니다.
    """
    print("--- meta_orchestrator 시작 (3단계 프레임워크 버전) ---")

    try:
        master_df = pd.read_csv(BASE_DATASET_PATH)
        print(f"기본 데이터셋 로드 성공: {BASE_DATASET_PATH} ({len(master_df)} 행)")
    except FileNotFoundError:
        print(f"치명적 오류: {BASE_DATASET_PATH} 에서 기본 데이터셋을 찾을 수 없습니다.")
        return

    for task_number in EXPERIMENT_PIPELINE:
        print(f"\n{'='*25} 파이프라인 작업 #{task_number} 시작 {'='*25}")
        
        task_feedback = get_task_feedback(task_number)
        if task_feedback.startswith("오류:"):
            print(task_feedback)
            continue

        metadata = get_task_metadata(task_feedback)
        if not metadata:
            print(f"[오류] 작업 #{task_number}의 피드백에서 메타데이터를 찾거나 파싱할 수 없습니다.")
            print("피드백 파일에 다음과 같은 형식의 주석이 있는지 확인하세요:")
            print("<!--\n task_type: PART_N\n target_metric: metric_name\n-->")
            continue
        
        task_type = metadata["task_type"]
        target_metric = metadata["target_metric"]
        
        print(f"작업 유형: {task_type}, 핵심 목표 지표: {target_metric}")
        # print(f"현재 작업 지침:\n{task_feedback}\n") # 이 출력은 너무 길어서 주석 처리

        task_name = TASK_FILE_NAMES.get(task_number, f"task_{task_number}")
        history_file = os.path.join(AGENTS_DIR, f"{task_name}_history.json")
        code_log_file = os.path.join(AGENTS_DIR, f"{task_name}_code_log.json")
        history_tool = HistoryTool(history_file=history_file)
        
        task_completed_successfully = False
        last_error = None

        for iteration in range(1, MAX_ITERATIONS_PER_TASK + 1):
            print(f"--- 작업 #{task_number} / 반복 {iteration}/{MAX_ITERATIONS_PER_TASK} ---")

            code_writer_tool, _ = setup_tools(
                task_number, 
                history_tool, 
                master_df, 
                task_type
            )
            
            agent_executor = create_agent_executor(
                code_writer_tool, 
                task_type, 
                correction_needed=last_error
            )

            try:
                results = agent_executor.run(
                    task_feedback=task_feedback, 
                    target_metric=target_metric,
                    df_head=master_df.head(3).to_markdown(index=False),
                    history=history_tool.load_history_str()
                )
            except Exception:
                last_error = f"AI 에이전트 실행 중 치명적 오류 발생:\n{traceback.format_exc()}"
                print(last_error)
                history_tool.add_event({"status": "fatal_error", "error": last_error})
                continue

            python_code = results.get("python_code")
            feature_name = results.get("feature_name")
            hypothesis = results.get("hypothesis")

            if not all([python_code, feature_name, hypothesis]):
                last_error = "AI가 불완전한 결과(코드, 피처명, 가설 누락)를 반환했습니다. 재시도합니다."
                print(f"[경고] {last_error}")
                history_tool.add_event({"status": "incomplete_response", "error": last_error})
                continue
            
            df_for_analysis = master_df if task_type != "PART_1" else master_df[master_df['source'] == 'ours']
            df_modified, execution_error = execute_generated_code(
                python_code, df_for_analysis, feature_name, task_type
            )

            if execution_error:
                last_error = f"코드 실행 또는 검증 실패:\n{execution_error}"
                print(f"[오류] {last_error}")
                history_tool.add_event({"status": "code_error", "error": last_error, "code": python_code})
                continue

            print(f"피처 '{feature_name}'가 성공적으로 생성 및 검증되었습니다.")
            master_df.update(df_modified)

            run_correlation_analysis_and_update_history(
                df=master_df,
                feature_name=feature_name,
                hypothesis=hypothesis,
                target_metric=target_metric,
                history_tool=history_tool
            )
            
            task_completed_successfully = True
            break
        
        if not task_completed_successfully:
            print(f"\n[최종 실패] 작업 #{task_number}이(가) 최대 반복 횟수({MAX_ITERATIONS_PER_TASK}) 내에 성공하지 못했습니다.")
            log_final_failure(code_log_file, task_number, last_error)

def log_final_failure(log_file: str, task_number: int, last_error: str):
    """Logs the final failure information to a JSON file."""
    failure_log = {
        "timestamp": datetime.now().isoformat(),
        "task_number": task_number,
        "status": "final_failure",
        "last_error_message": last_error
    }
    with open(log_file, 'a', encoding='utf-8') as f:
        json.dump(failure_log, f, ensure_ascii=False, indent=2)
        f.write(',\n')


def create_agent_executor(code_writer_tool: CodeWriterTool, task_type: str, correction_needed: str | None = None) -> AgentExecutor:
    # ... (기존 코드와 동일, correction_needed 프롬프트 추가 로직 구현 필요) ...
    # This part needs to be implemented to pass the error to the prompt.
    # For now, we assume the prompt is adapted to handle the `correction_needed` input.
    # ... (The rest of the function remains as is) ...
    pass # Placeholder for the actual implementation

def setup_tools(task_number: int, history_tool: HistoryTool, df: pd.DataFrame, task_type: str) -> tuple[CodeWriterTool, HistoryTool]:
    """
    Initializes and sets up the tools required for the agent.
    """
    # This function is assumed to exist and work correctly.
    # It sets up the CodeWriterTool and HistoryTool.
    pass # Placeholder for the actual implementation


def main():
    run_meta_orchestrator()

if __name__ == "__main__":
    main() 