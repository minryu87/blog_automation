import pandas as pd
import json
import os
from contextlib import redirect_stdout
import io
import re
from tqdm import tqdm

from .inference import inference_agent # Import the new agent

# --- Path Configuration ---
# Use __file__ to make paths robust, independent of the current working directory.
# AGENTS_DIR will be the absolute path to '.../blog_automation/agents'
AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
# BLOG_AUTOMATION_DIR will be the absolute path to '.../blog_automation'
BLOG_AUTOMATION_DIR = os.path.dirname(AGENTS_DIR)


class HistoryTool:
    """A tool to read and write the history of experiments."""
    def __init__(self, history_file=None):
        if history_file is None:
            history_file = os.path.join(AGENTS_DIR, 'history.json')
        self.history_file = history_file
        # Ensure the file exists and contains a valid, empty JSON list if new.
        if not os.path.exists(self.history_file):
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump([], f)

    def read_history(self) -> list:
        """Reads the history of past experiments robustly.
        Returns a list of dictionaries.
        """
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content:
                    return []
                history = json.loads(content)
                # Ensure the content is a list
                return history if isinstance(history, list) else []
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def write_history(self, report: dict):
        """Appends a new report to the history file."""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        
        try:
            with open(self.history_file, 'r+', encoding='utf-8') as f:
                # Handle empty file case
                content = f.read()
                if not content:
                    history = []
                else:
                    f.seek(0)
                    history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            history = []
            
        analysis_report = report.get("analysis", {})
        summary = {
            "feature_created": report.get("feature_name", "N/A"),
            "hypothesis": report.get("hypothesis", "N/A"),
            "conclusion": analysis_report.get("interpretation", report.get("reason", "Conclusion not available."))
        }
        
        if 'status' in report:
            summary['status'] = report['status']
        else:
            conclusion_str = str(summary["conclusion"]).lower()
            if "error" in conclusion_str or "failed" in conclusion_str or "실패" in conclusion_str or "not available" in conclusion_str:
                summary["status"] = "failed"
            else:
                summary["status"] = "success"

        if summary['status'] == 'success':
            if "correlation" in analysis_report:
                summary["correlation_results"] = {
                    "correlation": analysis_report.get("correlation"),
                    "p_value": analysis_report.get("p_value")
                }

        history.append(summary)

        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        return "Successfully wrote the experiment summary to history."

    def add_event(self, report: dict):
        """Appends a new report to the history file. Alias for write_history."""
        return self.write_history(report)


class HumanFeedbackTool:
    """A tool to read human feedback."""
    def __init__(self, feedback_file=None):
        if feedback_file is None:
            feedback_file = os.path.join(AGENTS_DIR, 'feedback.md')
        self.feedback_file = feedback_file
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                f.write("### Human Feedback for the SEO Agent Team\n\n")

    def read_feedback(self) -> str:
        """Reads the latest feedback from the human user."""
        try:
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                feedback = f.read()
            # Find the start of the user's actual feedback
            start_index = feedback.find('**Instructions for the user:**')
            if start_index != -1:
                # Extract content after the instruction line
                content_after_instructions = feedback[start_index:]
                actual_feedback = content_after_instructions.split('**', 2)[-1].strip()
                if actual_feedback and not actual_feedback.startswith("Example Feedback:"):
                     return actual_feedback
            return "No new human feedback provided."
        except FileNotFoundError:
            return "Feedback file not found."


class SemanticAnalysisTool:
    """A tool to perform semantic analysis on a text column."""
    
    def __init__(self):
        self._cache = {} # Cache for inference results to avoid repeated API calls

    def _is_word_related(self, word: str, topic: str) -> bool:
        """Uses the InferenceAgent to check if a word is related to a topic."""
        cache_key = f"{topic}|{word}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt = f"topic='{topic}', word='{word}'"
        try:
            response = inference_agent.run(prompt)
            result = json.loads(response.content)
            is_related = result.get("is_related", False)
            self._cache[cache_key] = is_related
            return is_related
        except (json.JSONDecodeError, AttributeError):
            # If the agent fails or returns malformed JSON, assume not related.
            return False

    def calculate_topic_density(self, df: pd.DataFrame, text_column: str, topic: str = None, context_column: str = None) -> pd.DataFrame:
        """
        Calculates the density of topic-related words in a given text column.
        The topic can be a fixed string OR dynamic based on another column.

        Args:
            df: The pandas DataFrame to analyze.
            text_column: The name of the column containing the text to analyze.
            topic: The fixed central topic to check for relevance.
            context_column: The name of the column to use as a dynamic topic for each row.

        Returns:
            The DataFrame with a new topic density column added.
        """
        if not topic and not context_column:
            raise ValueError("Either 'topic' or 'context_column' must be provided.")
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in the DataFrame.")
        if context_column and context_column not in df.columns:
            raise ValueError(f"Context column '{context_column}' not found in the DataFrame.")

        new_feature_name = f'topic_density_{topic or context_column}'.replace(" ", "_")
        print(f"--- Starting Semantic Analysis for feature: '{new_feature_name}' ---")

        # --- Dynamic Topic Analysis ---
        if context_column:
            # For dynamic topics, we classify word relevance for each unique context
            unique_contexts = df[context_column].unique()
            print(f"Found {len(unique_contexts)} unique contexts to analyze.")
            
            # This can be slow, so we process row by row
            densities = []
            for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Analyzing row-by-row for '{context_column}'"):
                current_topic = str(row[context_column])
                text = str(row[text_column])
                words = set(re.findall(r'\b\w+\b', text.lower()))
                if not words:
                    densities.append(0.0)
                    continue
                
                related_word_count = 0
                for word in words:
                    if self._is_word_related(word, current_topic):
                        related_word_count += 1
                
                densities.append(related_word_count / len(words))
            
            df[new_feature_name] = densities

        # --- Fixed Topic Analysis ---
        else:
            all_text = ' '.join(df[text_column].astype(str).tolist())
            unique_words = set(re.findall(r'\b\w+\b', all_text.lower()))
            print(f"Found {len(unique_words)} unique words to analyze for the topic '{topic}'.")

            related_words = {word for word in tqdm(unique_words, desc="Classifying words") if self._is_word_related(word, topic)}
            print(f"Found {len(related_words)} topic-related words.")

            def calculate_density(text: str) -> float:
                words = set(re.findall(r'\b\w+\b', str(text).lower()))
                if not words: return 0.0
                return len(words.intersection(related_words)) / len(words)

            df[new_feature_name] = df[text_column].apply(calculate_density)
        
        print(f"--- Semantic Analysis complete. New feature '{new_feature_name}' added. ---")
        return df


class DataSchemaTool:
    """A tool to inspect the schema of the agent's base dataset."""
    def __init__(self, dataset_file=None):
        if dataset_file is None:
            dataset_file = os.path.join(AGENTS_DIR, 'agent_base_dataset.csv')
        self.dataset_file = dataset_file

    def get_schema(self) -> str:
        """Returns the first 5 rows and column dtypes of the dataset."""
        if not os.path.exists(self.dataset_file):
            return f"Error: The base dataset file was not found at {self.dataset_file}. Please ensure it exists."
        df = pd.read_csv(self.dataset_file)
        schema_info = {
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "head": df.head().to_dict('records')
        }
        return json.dumps(schema_info, indent=2, ensure_ascii=False)

class CodeExecutionTool:
    """A tool to execute Python code for data analysis."""
    def __init__(self, dataset_file=None):
        if dataset_file is None:
            dataset_file = os.path.join(AGENTS_DIR, 'agent_base_dataset.csv')
        self.dataset_file = dataset_file

    def execute(self, code: str) -> str:
        """
        Executes the given Python code in a restricted environment.
        The code has access to a pandas DataFrame named 'df' loaded from the base dataset.
        It should return a dictionary with 'status' and 'result'.
        The last expression of the code will be the 'result'.
        """
        try:
            # Prepare the environment
            df = None
            if os.path.exists(self.dataset_file):
                 df = pd.read_csv(self.dataset_file)
            
            # The 'df' can be None if the base dataset doesn't exist yet, which is fine.
            # The agent's code should handle this possibility.
            local_vars = {'df': df, 'pd': pd, 'os': os, 'BLOG_AUTOMATION_DIR': BLOG_AUTOMATION_DIR}
            
            # Capture stdout to return print statements
            f = io.StringIO()
            with redirect_stdout(f):
                # Split code into lines and execute all but the last
                lines = code.strip().split('\n')
                exec_code = "\n".join(lines[:-1])
                eval_code = lines[-1]
                
                exec(exec_code, local_vars)
                result = eval(eval_code, local_vars)

            output = f.getvalue()
            
            response = {
                "status": "success",
                "stdout": output,
                "result": str(result)
            }
            return json.dumps(response, indent=2, ensure_ascii=False)

        except Exception as e:
            response = {
                "status": "error",
                "error_message": str(e)
            }
            return json.dumps(response, indent=2, ensure_ascii=False)

# DataExplorationTool is no longer needed as the agent will focus on a single base dataset. 

class LogFileTool:
    """A tool to read a log file."""
    def __init__(self, file_path='logical_failure_history.log'):
        self.log_file_path = os.path.join(AGENTS_DIR, file_path)

    def read_log_file(self) -> str:
        """Reads the entire content of the log file."""
        if not os.path.exists(self.log_file_path):
            return "No logical failure history found. You can start fresh."
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                # Read lines and return a unique set to avoid redundancy
                unique_lines = sorted(list(set(f.readlines())))
                return "".join(unique_lines)
        except Exception as e:
            return f"Error reading log file: {str(e)}" 