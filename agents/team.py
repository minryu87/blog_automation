import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.team import Team
from agno.models.google import Gemini

from .tools import (
    HistoryTool, 
    HumanFeedbackTool,
    DataSchemaTool,
    LogFileTool
)

# --- 0. Load Environment Variables ---
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# --- 1. Configure the LLM ---
LLM = Gemini(
    id=os.getenv("GEMINI_MODEL", "gemini-2.5-pro"),
    api_key=os.getenv("GEMINI_API_KEY")
)

# --- 2. Instantiate Tools ---
history_tool = HistoryTool()
human_feedback_tool = HumanFeedbackTool()
schema_tool = DataSchemaTool()
log_tool = LogFileTool()

# --- 3. Dynamically construct instructions based on past failures ---
base_instructions = [
    "You are an expert AI developer responsible for creating and debugging Python code for new SEO features.",
    "Your output MUST be a single, complete JSON object and nothing else.",
    "The JSON must contain: `feature_name` (string), `hypothesis` (string), and `python_code` (string).",
    
    "**CRITICAL RULE**: The `python_code` field MUST be a single string containing a complete, standalone Python script. It MUST include all necessary imports at the top.",

    "--- PERFECT CODE EXAMPLE ---",
    "```json",
    "{",
    "  \"feature_name\": \"example_feature_name\",",
    "  \"hypothesis\": \"A hypothesis about the feature.\",",
    "  \"python_code\": \"import pandas as pd\\nimport numpy as np\\nfrom sentence_transformers import SentenceTransformer, util\\n\\n# 1. Lazy-load the model to avoid re-initializing it on every call.\\n_model = None\\n\\ndef get_model():\\n    global _model\\n    if _model is None:\\n        _model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')\\n    return _model\\n\\ndef generate_feature(df: pd.DataFrame) -> pd.DataFrame:\\n    model = get_model()\\n\\n    # 2. Use efficient batch processing, not .apply()\\n    titles = df['post_title'].fillna('').astype(str).tolist()\\n    bodies = df['post_body'].fillna('').astype(str).tolist()\\n\\n    title_embeddings = model.encode(titles, convert_to_tensor=True)\\n    body_embeddings = model.encode(bodies, convert_to_tensor=True)\\n\\n    # 3. Return the full DataFrame with the new column.\\n    df['example_feature_name'] = util.cos_sim(title_embeddings, body_embeddings).diag().tolist()\\n    return df\"",
    "}",
    "```",
    "--------------------------",

    "**Coding Best Practices (You MUST follow these):**",
    "1.  **Safe Model Initialization:** As shown in the example, use a global `_model = None` and a `get_model()` function.",
    "2.  **Efficient Data Processing:** As shown in the example, process data in batches (`.tolist()` then `model.encode()`). NEVER use `.apply()` for encoding.",
    "3.  **Return Value:** The `generate_feature` function MUST return the entire, modified DataFrame.",
    "4.  **DataFrame Checks:** ALWAYS use `if not df.empty:` to check for empty DataFrames. NEVER use `if df:`.",

    "**Your Two Modes of Operation:**",
    "1.  **Creation Mode:** If not given a previous error, invent a novel feature, following the best practices and JSON format above.",
    "2.  **Self-Correction Mode:** If you are given a 'Previous Attempt that Failed' with an error traceback, analyze the root cause and provide a new, corrected version of the complete Python code, following all rules.",
]

# Read past logical failures and inject them as rules
past_failures = log_tool.read_log_file()
if "No logical failure history found" not in past_failures and past_failures.strip():
    failure_rules = [
        "\n**[CRITICAL] LEARN FROM PAST MISTAKES:**\nYou MUST follow these rules, which are based on logical errors from previous attempts:\n"
    ]
    # Extract unique reasons from the log
    unique_reasons = sorted(list(set([line.split("]")[-1].strip() for line in past_failures.strip().split('\n') if line.strip()])))
    for i, reason in enumerate(unique_reasons):
        failure_rules.append(f"{i+1}. {reason}")
    
    # Prepend the failure rules to the base instructions
    final_instructions = failure_rules + ["\n--------------------------\n"] + base_instructions
else:
    final_instructions = base_instructions

# --- 4. Define the new, unified FeatureEngineerAgent ---
feature_engineer_agent = Agent(
    name="FeatureEngineerAgent",
    role="Self-Correcting AI Python Developer for SEO Feature Engineering",
    model=LLM,
    tools=[schema_tool, history_tool, human_feedback_tool],
    instructions=final_instructions,
    show_tool_calls=True,
    markdown=True,
)

# --- 4. Define the FailureAnalysisAgent ---
failure_analysis_agent = Agent(
    name="FailureAnalysisAgent",
    role="Code Failure Root Cause Analyst",
    model=LLM,
    instructions=[
        "You are an expert in Python and data science. Your task is to analyze a failed code execution and identify the root logical cause.",
        "You will be given the failed code, the error traceback, and the original hypothesis.",
        "Your goal is to provide a concise, one-sentence summary of the *logical* reason for the failure. Do not just state the error type.",
        "Focus on *why* the developer's approach was wrong based on the data or problem description.",
        "Your output MUST be a single JSON object with one key: `failure_reason` (string).",

        "--- EXAMPLE 1 ---",
        "INPUT: Traceback contains `ModuleNotFoundError: No module named 'nltk'`, Code contains `import nltk`.",
        "OUTPUT: {\"failure_reason\": \"Logical Error: Used a disallowed library ('nltk') instead of relying on permitted, pre-installed packages.\"}",
        
        "--- EXAMPLE 2 ---",
        "INPUT: Traceback contains `KeyError: 'non_brand_ctr'`, Code contains `df['non_brand_ctr']`.",
        "OUTPUT: {\"failure_reason\": \"Logical Error: Attempted to use a non-existent column 'non_brand_ctr'.\"}",

        "--- EXAMPLE 3 ---",
        "INPUT: Traceback contains error on `nltk.sent_tokenize(body)`, Data Spec says `post_body` has no newlines.",
        "OUTPUT: {\"failure_reason\": \"Logical Error: Ignored the data specification that 'post_body' is a continuous string and cannot be split into sentences.\"}",
    ],
    show_tool_calls=False,
    markdown=False
) 