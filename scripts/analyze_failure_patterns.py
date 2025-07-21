import json
import pandas as pd
import os
from collections import Counter

def analyze_failure_patterns(log_file_path: str):
    """
    Analyzes an agent's code log to identify and categorize common failure patterns.

    Args:
        log_file_path (str): The path to the JSON code log file.
    """
    if not os.path.exists(log_file_path):
        print(f"Error: Log file not found at {log_file_path}")
        return

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {log_file_path}. The file might be corrupted.")
        return

    failure_reasons = []
    total_iterations = len(log_data)
    failed_iterations = 0

    for entry in log_data:
        is_failure = False
        reason = "Unknown Failure"

        # Criteria 1: A correction attempt was made.
        if entry.get("correction_attempts_made", 0) > 0 and entry.get("correction_history"):
            is_failure = True
            try:
                # Extract the most direct evidence of failure.
                traceback = entry["correction_history"][0].get("prompt", "")
                
                if "ModuleNotFoundError: No module named 'nltk'" in traceback:
                    reason = "Logical Error: Ignored data spec (Used nltk for sentence splitting)"
                elif "ModuleNotFoundError" in traceback:
                    error_line = [line for line in traceback.split('\\n') if "ModuleNotFoundError" in line]
                    reason = f"Environment Error: {error_line[0].strip()}"
                elif "KeyError" in traceback:
                    error_line = [line for line in traceback.split('\\n') if "KeyError" in line]
                    reason = f"Code Error: KeyError (Accessed non-existent column: {error_line[0].strip()})"
                else:
                    # A more generic reason if a specific pattern isn't found
                    reason = "Code Generation Error: Corrected after traceback"

            except (IndexError, KeyError):
                reason = "Parsing Error: Could not determine specific reason from correction_history."

        # Criteria 2: The final report indicates a failure.
        elif "Error during correlation analysis" in entry.get("final_report", {}).get("overall_conclusion", ""):
            is_failure = True
            reason = "Execution Error: Feature not found in DataFrame after execution"
            
        if is_failure:
            failed_iterations += 1
            failure_reasons.append(reason)

    print("--- Failure Analysis Report ---")
    print(f"Total Iterations Analyzed: {total_iterations}")
    print(f"Successful Iterations: {total_iterations - failed_iterations}")
    print(f"Failed Iterations: {failed_iterations}")
    
    if failed_iterations > 0:
        success_rate = ((total_iterations - failed_iterations) / total_iterations) * 100
        print(f"Success Rate: {success_rate:.2f}%")
        
        print("\n--- Top Failure Reasons ---")
        reason_counts = Counter(failure_reasons)
        for reason, count in reason_counts.most_common():
            percentage = (count / failed_iterations) * 100
            print(f"- {reason}: {count} times ({percentage:.2f}%)")
    
    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    # Define the path to the log file within the project structure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, '..', 'agents', 'post_body_semantic_analysis_code_log.json')
    
    analyze_failure_patterns(log_file) 