import os
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

from .team import feature_engineer_agent, failure_analysis_agent
from .tools import HistoryTool, AGENTS_DIR, HumanFeedbackTool

# --- 0. Load Environment Variables ---
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# --- 1. Meta-Orchestrator Configuration ---
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

GOAL = "Discover a new feature with a Pearson correlation of at least 0.6 (or -0.6) with 'non_brand_inflow' or 'non_brand_average_ctr'."
MAX_ITERATIONS = 5
MAX_CORRECTION_ATTEMPTS = 10
# --- MODIFIED: Point to the correct, enriched master dataset ---
BASE_DATASET_PATH = os.path.abspath(os.path.join(AGENTS_DIR, '..', 'data', 'data_processed', 'master_post_data.csv'))


def extract_json_from_string(text: str) -> dict | None:
    """Extracts the first JSON object from a string and returns it as a dict."""
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
    Handles ModuleNotFoundError by attempting to pip install the missing package
    and retrying the script execution once.
    """
    # Regex to find the missing module name
    match = re.search(r"ModuleNotFoundError: No module named '(\w+)'", error_output)
    if not match:
        return None, error_output # Not a ModuleNotFoundError, return original error

    library_name = match.group(1)
    print(f"[Self-Healing] Detected missing library: '{library_name}'. Attempting to install...")
    
    try:
        # Attempt to install the library
        install_result = subprocess.run(
            [sys.executable, "-m", "pip", "install", library_name],
            capture_output=True, text=True, check=True, encoding='utf-8'
        )
        print(f"[Self-Healing] Successfully installed '{library_name}'. Retrying script execution...")

        # Retry executing the script
        retry_result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, encoding='utf-8', check=False
        )
        
        if retry_result.returncode != 0:
            return None, retry_result.stderr # Failed even after install

        if os.path.exists(output_csv):
            modified_df = pd.read_csv(output_csv)
            return modified_df, None # Success on retry
        else:
            return None, "Output file was not created after retry."

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        install_error = e.stderr if hasattr(e, 'stderr') else str(e)
        return None, f"Failed to install '{library_name}'. Pip error: {install_error}"
    except Exception as e:
        return None, f"An unexpected error occurred during the install/retry process: {str(e)}"

def execute_feature_code(feature_code: str, base_df: pd.DataFrame) -> tuple[pd.DataFrame | None, str | None]:
    """
    Saves and executes the generated python code in an isolated subprocess.
    If a ModuleNotFoundError occurs, it attempts to install the package and retries.
    """
    temp_script_path = os.path.join(AGENTS_DIR, '_temp_feature_generator.py')
    temp_input_csv_path = os.path.join(AGENTS_DIR, '_temp_input.csv')
    temp_output_csv_path = os.path.join(AGENTS_DIR, '_temp_output.csv')

    # Prepare the execution script content
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
        
        # Find the generated function dynamically
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
        # 1. Write the generated code to a temporary script file
        with open(temp_script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)

        # 2. Save the input DataFrame to a temporary CSV
        base_df.to_csv(temp_input_csv_path, index=False)

        # 3. Execute the script in a subprocess
        result = subprocess.run(
            [sys.executable, temp_script_path],
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=False
        )

        # 4. Check for errors
        if result.returncode != 0:
            # --- SELF-HEALING LOGIC ---
            # If it's a ModuleNotFoundError, try to install and retry
            if "ModuleNotFoundError" in result.stderr:
                return _install_and_retry(result.stderr, temp_script_path, temp_input_csv_path, temp_output_csv_path)
            else:
                # For all other errors, return the traceback directly
                return None, result.stderr

        # 5. Read the output DataFrame from the result CSV
        if os.path.exists(temp_output_csv_path):
            modified_df = pd.read_csv(temp_output_csv_path)
            return modified_df, None
        else:
            return None, "Output file was not created by the script."

    except Exception as e:
        return None, traceback.format_exc()
    finally:
        # 6. Clean up temporary files
        for path in [temp_script_path, temp_input_csv_path, temp_output_csv_path]:
            if os.path.exists(path):
                os.remove(path)

def interpret_correlation(correlation, p_value):
    """Interprets the statistical significance and meaning of the correlation results."""
    if correlation is None or p_value is None:
        return "Correlation could not be calculated. This often happens if the generated feature has no variance (all values are the same)."

    # Significance level
    alpha = 0.05
    is_significant = p_value < alpha

    # Strength of correlation
    abs_corr = abs(correlation)
    if abs_corr >= 0.7:
        strength = "very strong"
    elif abs_corr >= 0.5:
        strength = "strong"
    elif abs_corr >= 0.3:
        strength = "moderate"
    else:
        strength = "weak"

    # Direction
    direction = "positive" if correlation > 0 else "negative"

    # Conclusion sentence
    conclusion = f"Found a {strength}, {direction} correlation ({correlation:.4f}). "
    if is_significant:
        conclusion += f"The result is statistically significant (p-value: {p_value:.4f}), suggesting the relationship is unlikely due to chance."
    else:
        conclusion += f"However, the result is not statistically significant (p-value: {p_value:.4f}), so we cannot confidently rule out random chance."
        
    return conclusion

def analyze_correlation(df: pd.DataFrame, feature_name: str) -> dict:
    """Analyzes the correlation of the new feature and provides interpretation."""
    report = {
        "correlation_results": {
            "non_brand_inflow": {"correlation": None, "p_value": None, "interpretation": ""},
            "non_brand_average_ctr": {"correlation": None, "p_value": None, "interpretation": ""},
        },
        "overall_conclusion": "Analysis failed."
    }
    try:
        if feature_name not in df.columns:
            raise ValueError(f"Feature '{feature_name}' not found in DataFrame after execution.")

        target_cols = ['non_brand_inflow', 'non_brand_average_ctr']
        if not all(col in df.columns for col in target_cols):
            raise ValueError(f"Required target columns not found. Needed: {target_cols}.")

        for target in target_cols:
            cleaned_df = df[[feature_name, target]].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(cleaned_df) > 1 and pd.api.types.is_numeric_dtype(cleaned_df[feature_name]) and cleaned_df[feature_name].nunique() > 1:
                corr, p_value = pearsonr(cleaned_df[feature_name], cleaned_df[target])
                report["correlation_results"][target]["correlation"] = None if np.isnan(corr) else corr
                report["correlation_results"][target]["p_value"] = None if np.isnan(p_value) else p_value
            else:
                # This case handles when correlation cannot be calculated
                corr, p_value = None, None
            
            interpretation = interpret_correlation(corr, p_value)
            report["correlation_results"][target]["interpretation"] = interpretation

        report["overall_conclusion"] = "Analysis completed. See interpretations for each target."
    except Exception as e:
        report["overall_conclusion"] = f"Error during correlation analysis: {str(e)}"
    return report

def analyze_and_log_failure(error_traceback: str, code: str, hypothesis: str):
    """Analyzes the root cause of a logical failure using an LLM agent and logs it."""
    print("[Failure Analysis] Engaging FailureAnalysisAgent to determine root cause...")
    
    analysis_prompt = (
        f"Analyze the following failed code execution to identify the root logical cause.\n\n"
        f"--- HYPOTHESIS ---\n{hypothesis}\n\n"
        f"--- FAILED CODE ---\n{code}\n\n"
        f"--- ERROR TRACEBACK ---\n{error_traceback}\n\n"
        "Based on this, what was the fundamental logical error in the developer's approach?"
    )
    
    try:
        response_raw = failure_analysis_agent.run(analysis_prompt).content
        response_json = extract_json_from_string(response_raw)
        failure_reason = response_json.get("failure_reason", "Unknown Logical Error")
    except Exception as e:
        print(f"[Failure Analysis] Agent failed to analyze the error: {e}")
        failure_reason = "Unknown Logical Error: The analysis agent itself failed."

    # Log the summarized failure to a file
    log_path = os.path.join(AGENTS_DIR, 'logical_failure_history.log')
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"[{datetime.now().isoformat()}] {failure_reason}\n")
    print(f"[Failure Analysis] Logged logical failure: {failure_reason}")


def run_meta_orchestrator():
    """The main orchestrator implementing the Self-Correction and Self-Healing Loop."""
    print("--- Starting Meta-Orchestrator ---")
    print(f"Goal: {GOAL}")

    try:
        base_df = pd.read_csv(BASE_DATASET_PATH)
        print(f"Successfully loaded base dataset: {BASE_DATASET_PATH}")
    except FileNotFoundError:
        print(f"FATAL ERROR: Base dataset not found at {BASE_DATASET_PATH}. Please provide the required file.")
        return

    DATASET_CONTEXT = """
### 전략 브리핑 및 데이터 명세서 (`master_post_data.csv`)

당신은 이제 단순한 데이터셋이 아닌, '경쟁에서 승리하기 위한 전략'이 담긴 특별 데이터셋을 분석하게 됩니다. 당신의 임무는 이 데이터의 구조와 목적을 완벽히 이해하고, '성공의 핵심 요인'을 찾아내는 것입니다.

---

### Part 1: 전략적 목표 (The "Why")

#### 1. 데이터의 핵심 목적
이 데이터셋의 유일한 목표는 **"왜 특정 주제(키워드)에서 우리 포스트는 경쟁사보다 순위가 낮은가?"** 라는 질문에 답하는 것입니다. 이를 위해, 각 '대표 검색어'(`representative_keyword`)라는 전장(battlefield)마다, 우리의 플레이어('ours')와 강력한 경쟁자('competitor')를 함께 배치했습니다.

#### 2. 두 그룹의 이해: '우리' vs '경쟁자'
당신이 분석할 모든 행(row)은 `source` 컬럼을 통해 두 그룹 중 하나에 속합니다. 이 두 그룹을 이해하는 것이 분석의 첫걸음입니다.

*   **`source` == 'ours' (우리 포스트):**
    *   **식별자:** 내부 시스템에서 사용하는 고유한 **숫자 `post_id`**로 식별됩니다.
    *   **강점:** **실제 성과 데이터(`non_brand_inflow`, `non_brand_average_ctr`)**를 가지고 있습니다. 이 데이터를 통해 어떤 포스트가 '성공'했고 어떤 포스트가 '실패'했는지 알 수 있습니다.

*   **`source` == 'competitor' (경쟁사 포스트):**
    *   **식별자:** 포스트의 **URL(`post_url`)** 자체가 핵심 식별자입니다.
    *   **강점:** 이들은 이미 특정 '대표 검색어' 전장에서 **상위 랭킹을 차지한, 증명된 강자들**입니다. 그들의 특징은 곧 '성공 공식'의 단서가 됩니다.
    *   **한계:** 외부 포스트이므로, 당연히 우리의 내부 성과 데이터(`non_brand_inflow` 등)는 가지고 있지 않습니다 (값이 비어있음).

#### 3. 당신의 진짜 임무: '성공 패턴' 발견
단순히 컬럼 간의 상관관계를 보는 것을 넘어, 다음 가설을 증명할 피처를 만들어야 합니다.
> **"성과가 좋은 포스트들(경쟁사 전체 + 우리 중 상위 포스트)은 A라는 공통적인 특징을 가지고 있지만, 성과가 나쁜 우리 포스트에서는 A라는 특징이 부족하다."**
이 가설을 데이터로 증명하는 것이 당신의 최종 목표입니다.

---

### Part 2: 주요 컬럼 상세 명세 (The "What")

*   **`source`**: `string`. 이 행의 데이터 출처. 'ours'(우리) 또는 'competitor'(경쟁사) 값을 가집니다.
*   **`representative_keyword`**: `string`. 각 포스트가 어떤 검색어 필드에서 경쟁하는지를 나타내는 '대표 검색어'.
*   **`post_id`**: `int64`. 'ours' 포스트의 고유 숫자 ID. 'competitor'의 경우 비어있음.
*   **`post_url`**: `string`. 포스트의 전체 URL. 'competitor' 포스트의 핵심 식별자.
*   **`post_title`**: `string`. 포스트의 제목.
*   **`post_body`**: `string`. 포스트의 핵심 텍스트 본문.
    *   **[매우 중요!]** 이 텍스트는 원문에서 순수 텍스트만 추출한 것으로, **문단을 구분하는 줄바꿈 문자(\\n)가 포함되어 있지 않습니다.** 모든 내용은 하나의 연속된 문자열입니다. 따라서, **문단 분할을 전제로 하는 피처를 생성해서는 안 됩니다.**
*   **`category_keywords` / `morpheme_words`**: `string`. 텍스트 분석을 통해 추출된 키워드 및 형태소 목록.
*   **각종 `_score` 및 `_count` 컬럼**: `float64` / `int64`. `readability_score`, `total_image_count` 등 포스트의 품질과 정량적 특성을 나타내는 사전 계산된 피처.
*   **`non_brand_inflow` / `non_brand_average_ctr`**: `float64`. 'ours' 포스트에만 존재하는 성과 지표. 'competitor'의 경우 비어있음(NaN).

이 브리핑을 완벽히 숙지하고, 이제 분석을 시작하십시오.
"""

    history_tool = HistoryTool()
    feedback_tool = HumanFeedbackTool()
    log_file_path = os.path.join(AGENTS_DIR, 'code_log.json')
    iteration = 0

    while iteration < MAX_ITERATIONS:
        iteration += 1
        print(f"\n{'='*20} Starting Iteration {iteration}/{MAX_ITERATIONS} {'='*20}")

        # --- Attempt 1: Creation Mode ---
        print("\n[Orchestrator] Running Agent in Creation Mode...")
        past_experiments = history_tool.read_history()
        user_feedback = feedback_tool.read_feedback()
        
        creation_prompt = (
            f"{DATASET_CONTEXT}\n\n"
            "Your task is to invent a novel feature to improve SEO analysis based on the context provided above. "
            f"Review past experiments: {json.dumps(past_experiments, indent=2)}. "
            f"Review user feedback: '{user_feedback}'. "
            "Formulate a plan and generate the required JSON object. Do not use the `get_schema` tool as the context is now provided."
        )
        
        agent_response_raw = feature_engineer_agent.run(creation_prompt).content
        task_json = extract_json_from_string(agent_response_raw)
        
        final_report = None
        if not task_json:
            final_report = {"error": "Agent did not return a valid JSON object.", "raw_output": agent_response_raw}
        else:
            feature_name = task_json.get("feature_name")
            hypothesis = task_json.get("hypothesis")
            python_code = task_json.get("python_code")

            if not all([feature_name, hypothesis, python_code]):
                final_report = {"error": "JSON from agent is missing required keys.", "raw_output": str(task_json)}
            else:
                print(f"[Orchestrator] Agent proposed feature: '{feature_name}'")
                df_modified, error_traceback = execute_feature_code(python_code, base_df)

                # --- Self-Correction & Failure Analysis Loop ---
                correction_attempts = 0
                original_code = python_code
                correction_history = []
                
                # New: Check for constant feature value as a failure condition
                if df_modified is not None and feature_name in df_modified.columns:
                    if df_modified[feature_name].nunique() <= 1:
                        error_traceback = (
                            f"Code Error: The generated feature '{feature_name}' has no variance (all values are the same). "
                            "Correlation cannot be computed on a constant. "
                            "Please modify the code to produce a range of different values for different posts."
                        )

                while error_traceback and correction_attempts < MAX_CORRECTION_ATTEMPTS:
                    correction_attempts += 1
                    print(f"[Orchestrator] Code execution failed. Engaging Self-Correction Mode (Attempt {correction_attempts}/{MAX_CORRECTION_ATTEMPTS})...")
                    
                    correction_prompt = (
                        f"Your previous code attempt failed. Analyze the root cause and provide a corrected version.\n"
                        f"{DATASET_CONTEXT}\n\n"
                        f"--- PREVIOUS HYPOTHESIS ---\n{hypothesis}\n"
                        f"--- FAILED CODE ---\n{python_code}\n"
                        f"--- ERROR TRACEBACK ---\n{error_traceback}\n"
                        "--- NEW JSON WITH CORRECTED CODE ---"
                    )
                    
                    agent_response = feature_engineer_agent.run(correction_prompt).content
                    corrected_task_json = extract_json_from_string(agent_response)

                    correction_history.append({
                        "attempt": correction_attempts,
                        "prompt": correction_prompt,
                        "response_raw": agent_response,
                        "response_json": corrected_task_json
                    })

                    if not corrected_task_json or not corrected_task_json.get("python_code"):
                         final_report = {"error": "Agent failed to return valid JSON with code in correction mode.", "correction_history": correction_history}
                         error_traceback = "Agent failed to provide correct JSON." # Stop the loop
                         break 
                    
                    python_code = corrected_task_json.get("python_code") # Use new code
                    print(f"[Orchestrator] Retrying with corrected code (Attempt {correction_attempts})...")
                    df_modified, error_traceback = execute_feature_code(python_code, base_df)
                    
                    # New: Re-check for constant feature value after correction
                    if df_modified is not None and feature_name in df_modified.columns:
                         if df_modified[feature_name].nunique() <= 1:
                            error_traceback = (
                                f"Code Error: The generated feature '{feature_name}' still has no variance after correction. "
                                "Please ensure your logic can produce a range of different values across the dataset."
                            )
                         else:
                            error_traceback = None # Clear error if variance is now present

                # --- Final Analysis ---
                if not error_traceback and df_modified is not None:
                    print("[Orchestrator] Code executed successfully. Analyzing correlation...")
                    final_report = analyze_correlation(df_modified, feature_name)
                    final_report["hypothesis"] = hypothesis
                    final_report["feature_created"] = feature_name
                else:
                    # --- NEW: Log the final failure ONCE after all correction attempts fail ---
                    if error_traceback and "ModuleNotFoundError" not in error_traceback:
                         analyze_and_log_failure(error_traceback, python_code, hypothesis)
                    final_report = {"error": "Code failed after all correction attempts.", "final_traceback": error_traceback, "correction_history": correction_history}
        
        # --- Logging the entire iteration attempt ---
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "initial_agent_response": task_json,
            "correction_attempts_made": correction_attempts,
            "correction_history": correction_history if 'correction_history' in locals() else [],
            'final_report': final_report
        }
        
        try:
            with open(log_file_path, 'r+') as f:
                logs = json.load(f)
                logs.append(log_entry)
                f.seek(0)
                json.dump(logs, f, indent=2)
        except (FileNotFoundError, json.JSONDecodeError):
            with open(log_file_path, 'w') as f:
                json.dump([log_entry], f, indent=2)

        print("\n--- Iteration Result ---")
        print(json.dumps(final_report, indent=2))
        history_tool.write_history(final_report)
        print("\nSuccessfully recorded experiment summary to history.")
        
        corr_inflow = final_report.get("correlation_results", {}).get("non_brand_inflow", {}).get("correlation") or 0
        corr_avg_c = final_report.get("correlation_results", {}).get("non_brand_average_ctr", {}).get("correlation") or 0

        if abs(corr_inflow) >= 0.6 or abs(corr_avg_c) >= 0.6:
            print(f"\n{'='*20} GOAL ACHIEVED! {'='*20}")
            print(f"Discovered significant feature: '{final_report.get('feature_created')}'")
            break
        else:
            print("\nGoal not yet achieved.")

    if iteration >= MAX_ITERATIONS:
        print(f"\n{'='*20} Max Iterations Reached {'='*20}")

def main():
    # The log file should persist across runs to analyze failure patterns.
    # The deletion logic that was here was faulty and has been removed.
    run_meta_orchestrator()

if __name__ == "__main__":
    main() 