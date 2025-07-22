import os
import sys
import json
import logging
import re
import signal
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
from .team import feature_engineer_agent, failure_analysis_agent
from .tools import HistoryTool, HumanFeedbackTool, LogFileTool, AGENTS_DIR

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
TASK_FILE_NAMES = config.TASK_FILE_NAMES
BASE_DATASET_PATH = config.BASE_DATASET_PATH
FEEDBACK_FILE_PATH = config.FEEDBACK_FILE_PATH

# --- Graceful Shutdown ---
shutdown_flag = False
force_shutdown = False

def signal_handler(sig, frame):
    global shutdown_flag, force_shutdown
    if not shutdown_flag:
        print("\n[Orchestrator] 종료 신호 감지. 현재 작업 완료 후 안전하게 종료합니다.")
        print("즉시 강제 종료하려면 Ctrl+C를 다시 누르세요.")
        shutdown_flag = True
    else:
        print("\n[Orchestrator] 강제 종료 신호 감지. 즉시 종료합니다.")
        force_shutdown = True
        sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)

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
    match = re.search(r"ModuleNotFoundError: No module named '(\w+)'", error_output)
    if not match:
        return None, error_output

    library_name = match.group(1)
    print(f"[Self-Healing] 누락된 라이브러리 발견: '{library_name}'. 설치를 시도합니다...")
    
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", library_name],
            capture_output=True, text=True, check=True, encoding='utf-8'
        )
        print(f"[Self-Healing] '{library_name}' 성공적으로 설치되었습니다. 스크립트 실행을 다시 시도합니다...")

        retry_result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, encoding='utf-8', check=False
        )
        
        if retry_result.returncode != 0:
            return None, retry_result.stderr

        if os.path.exists(output_csv):
            return pd.read_csv(output_csv), None
        else:
            return None, "재시도 후에도 출력 파일이 생성되지 않았습니다."
    except Exception as e:
        return None, f"'{library_name}' 설치 또는 재시도 중 오류 발생: {str(e)}"

def execute_feature_code(feature_code: str, base_df: pd.DataFrame) -> tuple[pd.DataFrame | None, str | None]:
    """
    Saves and executes the generated python code in an isolated subprocess.
    """
    temp_script_path = os.path.join(AGENTS_DIR, '_temp_feature_generator.py')
    temp_input_csv_path = os.path.join(AGENTS_DIR, '_temp_input.csv')
    temp_output_csv_path = os.path.join(AGENTS_DIR, '_temp_output.csv')

    script_content = f"""
import pandas as pd
import numpy as np
import sys
import os
from sentence_transformers import SentenceTransformer, util
import torch
import re

# --- Generated Feature Code ---
{feature_code}
# --- Execution Logic ---
def run():
    input_path = '{temp_input_csv_path}'
    output_path = '{temp_output_csv_path}'
    try:
        df = pd.read_csv(input_path)
        func_name = next((name for name in globals() if name.startswith('generate_feature')), None)
        if not func_name:
            raise ValueError("No function starting with 'generate_feature' found.")
        generate_func = globals()[func_name]
        modified_df = generate_func(df)
        modified_df.to_csv(output_path, index=False)
    except Exception as e:
        print(f"Error during script execution: {{e}}", file=sys.stderr)
        sys.exit(1)
if __name__ == '__main__':
    run()
"""
    try:
        with open(temp_script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        base_df.to_csv(temp_input_csv_path, index=False)
        result = subprocess.run(
            ["python", temp_script_path],
            capture_output=True, text=True, encoding='utf-8', check=False
        )
        if result.returncode != 0:
            if "ModuleNotFoundError" in result.stderr:
                return _install_and_retry(result.stderr, temp_script_path, temp_input_csv_path, temp_output_csv_path)
            else:
                return None, result.stderr
        if os.path.exists(temp_output_csv_path):
            modified_df = pd.read_csv(temp_output_csv_path)
            return modified_df, None
        else:
            return None, "Output file was not created by the script."
    except Exception as e:
        return None, traceback.format_exc()
    finally:
        for path in [temp_script_path, temp_input_csv_path, temp_output_csv_path]:
            if os.path.exists(path):
                os.remove(path)

def analyze_correlation(df: pd.DataFrame, feature_name: str, target_metric: str) -> dict:
    """지정된 단일 타겟 지표에 대해 피처의 상관관계를 분석하고 해석을 반환합니다."""
    if feature_name not in df.columns:
        raise ValueError(f"Feature '{feature_name}' not found in DataFrame.")
    
    df_for_analysis = df.copy()

    # 데이터 타입을 숫자로 강제 변환합니다. 변환할 수 없는 값은 NaN으로 처리됩니다.
    df_for_analysis[feature_name] = pd.to_numeric(df_for_analysis[feature_name], errors='coerce')
    df_for_analysis[target_metric] = pd.to_numeric(df_for_analysis[target_metric], errors='coerce')

    cleaned_df = df_for_analysis[[feature_name, target_metric]].replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(cleaned_df) < 2 or cleaned_df[feature_name].nunique() <= 1:
        corr, p_value = None, None
    else:
        corr, p_value = pearsonr(cleaned_df[feature_name], cleaned_df[target_metric])

    return {"correlation": corr, "p_value": p_value, "interpretation": interpret_correlation(corr, p_value)}

def interpret_correlation(correlation, p_value):
    """상관관계 분석 결과의 통계적 유의미성과 의미를 해석합니다."""
    if correlation is None or p_value is None:
        return "상관관계를 계산할 수 없습니다 (예: 피처의 분산이 0)."
    alpha = 0.05
    is_significant = p_value < alpha
    abs_corr = abs(correlation)
    strength = "매우 강한" if abs_corr >= 0.7 else "강한" if abs_corr >= 0.5 else "중간 정도의" if abs_corr >= 0.3 else "약한"
    direction = "양의" if correlation > 0 else "음의"
    conclusion = f"{strength} {direction} 상관관계({correlation:.4f})를 발견했습니다. "
    conclusion += f"이 결과는 통계적으로 유의미합니다(p-value: {p_value:.4f})." if is_significant else f"하지만 통계적으로 유의미하지 않습니다(p-value: {p_value:.4f})."
    return conclusion

def analyze_and_log_failure(error_traceback: str, code: str, hypothesis: str):
    """LLM 에이전트를 사용하여 논리적 실패의 근본 원인을 분석하고 기록합니다."""
    print("[실패 분석] FailureAnalysisAgent를 사용하여 근본 원인 분석 중...")
    analysis_prompt = (
        f"다음 실패한 코드 실행을 분석하여 근본적인 논리적 원인을 찾아내세요.\n\n"
        f"--- 가설 ---\n{hypothesis}\n\n"
        f"--- 실패한 코드 ---\n{code}\n\n"
        f"--- 에러 트레이스백 ---\n{error_traceback}\n\n"
    )
    try:
        response_raw = failure_analysis_agent.run(analysis_prompt).content
        response_json = extract_json_from_string(response_raw)
        failure_reason = response_json.get("failure_reason", "알 수 없는 논리적 오류")
    except Exception as e:
        failure_reason = f"분석 에이전트 자체에서 오류 발생: {e}"
    
    # LogFileTool을 사용하는 대신 직접 파일에 기록합니다.
    log_path = os.path.join(AGENTS_DIR, 'logical_failure_history.log')
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"[{datetime.now().isoformat()}] {failure_reason}\n")

    print(f"[실패 분석] 논리적 실패 기록 완료: {failure_reason}")

def get_task_feedback(task_number: int) -> str:
    """feedback.md에서 특정 작업 번호에 대한 지침과 그 상위 PART 설명을 함께 추출합니다."""
    try:
        with open(FEEDBACK_FILE_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        return f"오류: 피드백 파일({FEEDBACK_FILE_PATH})을 찾을 수 없습니다."

    # 전체 내용을 PART별로 분리합니다.
    parts = re.split(r'(?=## \[PART)', content)
    
    for part_content in parts:
        if not part_content.strip():
            continue
            
        # 현재 PART에 원하는 작업이 포함되어 있는지 확인합니다.
        task_pattern = re.compile(rf"### \[작업 {task_number}\]:", re.IGNORECASE)
        if task_pattern.search(part_content):
            # 작업이 포함된 경우, 해당 PART의 설명과 특정 TASK의 설명을 조합하여 반환합니다.
            
            # PART 블록 전체를 가져옵니다.
            part_match = re.search(r"## \[PART.*?\]:.*?(?=\n## \[PART|\Z)", part_content, re.DOTALL | re.IGNORECASE)
            part_block = part_match.group(0).strip() if part_match else ""

            # 특정 TASK 블록을 가져옵니다.
            task_match = re.search(rf"(### \[작업 {task_number}\]:.*?)(?=\n### \[작업|\Z)", part_content, re.DOTALL | re.IGNORECASE)
            task_block = task_match.group(1).strip() if task_match else ""
            
            # PART 설명과 TASK 설명을 합쳐서 반환합니다.
            if part_block and task_block:
                # PART 설명에서 TASK 목록은 제외하고 헤더 부분만 사용하기 위해 --- 기준으로 자릅니다.
                part_header = part_block.split('---')[0].strip()
                return f"{part_header}\n\n{task_block}"

    return f"오류: 피드백 파일에서 [작업 {task_number}]에 대한 지침을 찾을 수 없습니다."

def get_task_metadata(task_feedback: str) -> dict | None:
    """작업 피드백에서 구조화된 메타데이터를 파싱합니다."""
    # 메타데이터는 이제 ### [작업] 블록 안에만 존재합니다.
    task_metadata_pattern = re.compile(r"### \[작업 \d+\]:.*?\n<!--(.*?)-->", re.DOTALL)
    metadata_match = task_metadata_pattern.search(task_feedback)
    
    if not metadata_match: return None
    try:
        metadata_text = metadata_match.group(1)
        task_type = re.search(r"task_type:\s*(\w+)", metadata_text).group(1)
        target_metric = re.search(r"target_metric:\s*(\w+)", metadata_text).group(1)
        return {"task_type": task_type, "target_metric": target_metric}
    except (AttributeError, IndexError):
        return None

def run_meta_orchestrator():
    """피처 엔지니어링 프로세스를 총괄하는 메인 오케스트레이터입니다."""
    print("--- meta_orchestrator 시작 (파이프라인 버전) ---")
    try:
        master_df = pd.read_csv(BASE_DATASET_PATH)
        print(f"기본 데이터셋 로드 성공: {len(master_df)} 행")
    except FileNotFoundError:
        print(f"치명적 오류: {BASE_DATASET_PATH} 에서 기본 데이터셋을 찾을 수 없습니다.")
        return

    # TO-BE: 전체 파이프라인을 N번 반복하는 최상위 사이클 루프
    for cycle in range(1, MAX_ITERATIONS_PER_TASK + 1):
        if shutdown_flag: break
        print(f"\n{'='*30} 전체 사이클 #{cycle}/{MAX_ITERATIONS_PER_TASK} 시작 {'='*30}")

        # 기존의 작업 루프가 이제 내부 루프가 됩니다.
        for task_number in EXPERIMENT_PIPELINE:
            if shutdown_flag: break
            print(f"\n--- [사이클 {cycle}] 파이프라인 작업 #{task_number} ---")
            
            task_feedback = get_task_feedback(task_number)
            if task_feedback.startswith("오류:"):
                print(task_feedback); continue

            metadata = get_task_metadata(task_feedback)
            if not metadata:
                print(f"[오류] 작업 #{task_number}의 피드백에서 메타데이터를 파싱할 수 없습니다."); continue
            
            task_type, target_metric = metadata["task_type"], metadata["target_metric"]
            print(f"작업 유형: {task_type}, 핵심 목표 지표: {target_metric}")

            task_name = TASK_FILE_NAMES.get(task_number, f"task_{task_number}")
            history_tool = HistoryTool(history_file=os.path.join(AGENTS_DIR, f"{task_name}_history.json"))
            feedback_tool = HumanFeedbackTool()
            code_log_file = os.path.join(AGENTS_DIR, f"{task_name}_code_log.json")
            
            # 각 작업에 대해 새로운 피처 생성을 1회 시도합니다 (내부 자가-수정 포함).
            last_error = None
            
            for attempt in range(1, MAX_CORRECTION_ATTEMPTS + 1):
                if shutdown_flag: break
                print(f"\n--- 작업 #{task_number} / 시도 {attempt}/{MAX_CORRECTION_ATTEMPTS} ---")

                print("[Orchestrator] 프롬프트 생성 중...")
                past_experiments = history_tool.read_history()
                user_feedback = feedback_tool.read_feedback()
                
                df_for_prompt = master_df[master_df['source'] == 'ours'] if task_type == 'PART_1' else master_df
                
                prompt = (
                    f"{task_feedback}\n\n"
                    f"--- 이전 실험 요약 ---\n"
                    f"{json.dumps(past_experiments, indent=2, ensure_ascii=False)}\n\n"
                    f"--- 사용자 피드백 ---\n"
                    f"{user_feedback}\n\n"
                    f"--- 데이터셋의 일부 ---\n"
                    f"{df_for_prompt.head(3).to_markdown(index=False)}\n\n"
                    "참고: 이전 실험과 중복되지 않는 새롭고 창의적인 가설을 세워주세요."
                )
                
                if last_error:
                    prompt += f"\n\n이전 시도는 다음 오류로 실패했습니다:\n{last_error}\n오류의 원인을 분석하고 코드를 수정하여 다시 제안해주세요."
                
                print("[Orchestrator] AI 에이전트 실행 중...")
                agent_response_raw = feature_engineer_agent.run(prompt).content
                print(f"[Orchestrator] AI 응답 수신:\n{agent_response_raw}")

                print("[Orchestrator] AI 응답 검증 중...")
                response_json = extract_json_from_string(agent_response_raw)

                if not response_json or not all(response_json.get(k) for k in ["feature_name", "hypothesis", "python_code"]):
                    last_error = f"에이전트가 불완전한 JSON(null 또는 빈 값 포함)을 반환했습니다. Raw: {agent_response_raw}"
                    print(f"[오류] {last_error}")
                    with open(code_log_file, 'a', encoding='utf-8') as f:
                        json.dump({"timestamp": datetime.now().isoformat(), "attempt": attempt, "status": "incomplete_response", "error": last_error, "raw_response": agent_response_raw}, f, ensure_ascii=False, indent=2)
                    continue
                
                print("[Orchestrator] AI 응답 검증 완료.")
                feature_name, hypothesis, python_code = response_json["feature_name"], response_json["hypothesis"], response_json["python_code"]
                
                print(f"[Orchestrator] 생성된 피처: '{feature_name}'")
                print(f"[Orchestrator] 가설: '{hypothesis}'")
                print("[Orchestrator] 코드 실행 준비 중...")
                
                df_for_analysis = master_df.copy()
                df_modified, execution_error = execute_feature_code(python_code, df_for_analysis)

                if execution_error:
                    last_error = f"코드 실행 실패:\n{execution_error}"
                    print(f"[오류] {last_error}")
                    if "ModuleNotFoundError" not in execution_error:
                        analyze_and_log_failure(execution_error, python_code, hypothesis)
                    with open(code_log_file, 'a', encoding='utf-8') as f:
                        json.dump({"timestamp": datetime.now().isoformat(), "attempt": attempt, "status": "execution_error", "error": last_error, "code": python_code}, f, ensure_ascii=False, indent=2)
                    continue

                print(f"피처 '{feature_name}'가 성공적으로 생성 및 검증되었습니다.")
                if feature_name not in df_modified.columns:
                    last_error = f"논리적 오류: 코드는 실행됐지만 '{feature_name}' 컬럼이 생성되지 않았습니다."
                    print(f"[오류] {last_error}")
                    # 코드 로그 기록
                    with open(code_log_file, 'a', encoding='utf-8') as f:
                        json.dump({"timestamp": datetime.now().isoformat(), "attempt": attempt, "status": "logical_error", "error": last_error, "code": python_code}, f, ensure_ascii=False, indent=2)
                    continue

                # 올바른 방법: 새 피처 컬럼을 master_df에 직접 할당합니다.
                master_df[feature_name] = df_modified[feature_name]

                # 논리적 오류 검사 2: 피처의 분산이 있는지 확인 (상관관계 분석 전 필수)
                if master_df[feature_name].nunique(dropna=False) <= 1:
                    last_error = (
                        f"논리적 오류: 생성된 피처 '{feature_name}'의 값이 모두 동일하여 "
                        f"상관관계를 계산할 수 없습니다. 모든 행에 대해 다른 값을 생성하도록 코드를 수정해주세요."
                    )
                    print(f"[오류] {last_error}")
                    # 논리적 오류를 분석하고 기록합니다.
                    analyze_and_log_failure(last_error, python_code, hypothesis)
                    # 코드 로그 기록
                    with open(code_log_file, 'a', encoding='utf-8') as f:
                        json.dump({"timestamp": datetime.now().isoformat(), "attempt": attempt, "status": "logical_error", "error": last_error, "code": python_code}, f, ensure_ascii=False, indent=2)
                    continue
                
                analysis_report = analyze_correlation(master_df, feature_name, target_metric)
                history_tool.add_event({"feature_name": feature_name, "hypothesis": hypothesis, "analysis": analysis_report})
                print(f"상관관계 분석 완료: {analysis_report['interpretation']}")

                # 코드 로그 기록 (성공)
                with open(code_log_file, 'a', encoding='utf-8') as f:
                    json.dump({"timestamp": datetime.now().isoformat(), "attempt": attempt, "status": "success", "feature_name": feature_name, "hypothesis": hypothesis, "code": python_code, "analysis": analysis_report}, f, ensure_ascii=False, indent=2)
                
                if abs(analysis_report.get("correlation", 0) or 0) >= 0.5:
                     print(f"🎉 목표 달성! 유의미한 피처 '{feature_name}'를 발견했습니다.")
                
                # 성공했으므로 이 작업의 자가-수정 루프를 탈출하고 다음 작업으로 넘어갑니다.
                break 
            
            else: # for...else 구문: break 없이 루프가 끝나면 (모든 시도 실패시) 실행
                print(f"\n[최종 실패] 작업 #{task_number}이(가) 최대 시도 횟수({MAX_CORRECTION_ATTEMPTS}) 내에 성공하지 못했습니다.")
                history_tool.add_event({"status": "failed", "reason": f"Max correction attempts reached ({MAX_CORRECTION_ATTEMPTS}).", "last_error": last_error})
    
    if shutdown_flag:
        print("\n[Orchestrator] 정상적으로 종료되었습니다.")
    else:
        print(f"\n{'='*30} 모든 사이클 완료 {'='*30}")


def main():
    run_meta_orchestrator()

if __name__ == "__main__":
    main() 