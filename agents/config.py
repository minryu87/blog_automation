"""
Configuration for the Feature Engineering Agent System.

This file centralizes settings for the main execution script, allowing for easy 
adjustments to the agent's goals, iteration counts, and the selection of 
experimental tasks without modifying the core logic.
"""

# --- Main Goal ---
# Defines the ultimate objective for the agent system. The agent will continue to
# generate and test hypotheses until this correlation threshold is met or
# MAX_ITERATIONS is reached for a given task.
GOAL = "아래 제시된 핵심 목표 지표와 0.5 이상(또는 -0.5 이상)의 피어슨 상관관계를 가진 새로운 피처를 발견하세요."

# --- Execution Controls ---
# Maximum number of features to generate and test for each task in the pipeline.
MAX_ITERATIONS_PER_TASK = 1

# Maximum number of self-correction attempts the agent can make for a single feature.
MAX_CORRECTION_ATTEMPTS = 3

# The full pipeline of experiment tasks to run, as defined in the new 3-part framework.
EXPERIMENT_PIPELINE = list(range(1, 13)) # Run all 12 tasks

# Provides descriptive names for generated files based on the task number.
TASK_FILE_NAMES = {
    # PART 1: INTERNAL ANALYSIS
    1: "P1_Internal_Inflow_All_Intrinsic",
    2: "P1_Internal_CTR_All_Intrinsic",
    # PART 2: EXTERNAL BENCHMARKING - INTRINSIC FEATURES
    3: "P2_Benchmark_Inflow_QuantScore",
    4: "P2_Benchmark_CTR_QuantScore",
    5: "P2_Benchmark_Inflow_Semantic",
    6: "P2_Benchmark_CTR_Semantic",
    7: "P2_Benchmark_Inflow_Morpheme",
    8: "P2_Benchmark_CTR_Morpheme",
    # PART 3: EXTERNAL BENCHMARKING - RELATIONAL FEATURES
    9: "P3_Benchmark_Inflow_TitleBody_Relation",
    10: "P3_Benchmark_CTR_TitleBody_Relation",
    11: "P3_Benchmark_Inflow_Query_Relation",
    12: "P3_Benchmark_CTR_Query_Relation",
}

# --- DO NOT MODIFY: File Path Configuration (auto-managed) ---
# The following paths are dynamically determined and should not be edited manually.
BASE_DATASET_PATH = "blog_automation/data/data_processed/master_post_data.csv"
HISTORY_FILE_PATH = "blog_automation/agents/history.json"
CODE_LOG_FILE_PATH = "blog_automation/agents/code_log.json"
FEEDBACK_FILE_PATH = "blog_automation/agents/feedback.md"
LOGICAL_FAILURE_LOG_PATH = "blog_automation/agents/logical_failure_history.log" 