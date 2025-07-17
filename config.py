import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_INPUT_DIR = os.path.join(DATA_DIR, "data_input")
DATA_RAW_DIR = os.path.join(DATA_DIR, "data_raw")
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, "data_processed")

# Create directories if they don't exist
os.makedirs(DATA_INPUT_DIR, exist_ok=True)
os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

NSIDE_API_URL = "https://nside.kr/ajx/get_post_diagnosis.php"
NSIDE_COOKIE = os.getenv("NSIDE_COOKIE")
