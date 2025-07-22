# --- Post Improvement System Configuration ---

# The number of lowest-performing posts to select for improvement.
NUM_POSTS_TO_IMPROVE = 5

# 각 피처를 개선하기 위한 최대 시도 횟수
MAX_ATTEMPTS_PER_FEATURE = 10

# The performance threshold that a post must meet or exceed to be considered "improved".
# We will define this as the 75th percentile of the CTR distribution.
IMPROVEMENT_THRESHOLD_PERCENTILE = 0.70

# --- File and Path Configuration ---
# Note: We are not creating new directories, so all outputs go into existing ones.

# Path to the trained champion model for CTR prediction.
CTR_MODEL_PATH = "/Users/min/codes/medilawyer_sales/blog_automation/data/modeling/trained_models/ctr_champion_model.joblib"

# Path to the master data file.
MASTER_DATA_PATH = "/Users/min/codes/medilawyer_sales/blog_automation/data/data_processed/agent_base_dataset.csv"

# Base path for feature data
DATA_PATH = "/Users/min/codes/medilawyer_sales/blog_automation/data/modeling/feature_calculate"

# Directory where all output files (analysis, edited posts, summaries) will be saved.
# A timestamp will be prepended to each filename to group them by run.
OUTPUT_DATA_PATH = "/Users/min/codes/medilawyer_sales/blog_automation/data/post_edit"

# [File Settings]
FEATURE_FILES = {
    'ctr': 'ctr_feature_value.csv',
    'inflow': 'inflow_feature_value.csv'
} 