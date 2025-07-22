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
# 원본 경쟁사 데이터 경로
COMPETITOR_DATA_PATH = "/Users/min/codes/medilawyer_sales/blog_automation/data/data_processed/master_post_data.csv"
# 개선 대상 포스트 목록 파일 경로
CANDIDATE_POSTS_PATH = "/Users/min/codes/medilawyer_sales/blog_automation/data/post_edit/20250722_143801_feature_with_prediction.csv"
# Directory where all output files (analysis, edited posts, summaries) will be saved.
# A timestamp will be prepended to each filename to group them by run.
OUTPUT_DATA_PATH = "/Users/min/codes/medilawyer_sales/blog_automation/data/post_edit" 