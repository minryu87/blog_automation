import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from blog_automation.config import DATA_INPUT_DIR
from blog_automation.app.services.data_collector import collect_data_from_urls
from blog_automation.app.services.data_parser import parse_html_file
from tqdm import tqdm
from fastapi import BackgroundTasks
from blog_automation.keyword_research import main as run_keyword_research

app = FastAPI()

class FileProcessRequest(BaseModel):
    file_name: str

@app.post("/collect_and_process")
async def collect_and_process(request: FileProcessRequest):
    csv_path = os.path.join(DATA_INPUT_DIR, request.file_name)
    
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="CSV file not found")
        
    try:
        df = pd.read_csv(csv_path)
        post_urls = df['postUrl'].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading CSV file: {e}")

    # 1. 데이터 수집
    collected_html_files = collect_data_from_urls(post_urls)
    if not collected_html_files:
        raise HTTPException(status_code=500, detail="Data collection failed for all URLs.")

    # 2. 데이터 가공 및 저장
    processed_json_files = []
    print("Processing HTML files...")
    for html_file in tqdm(collected_html_files, desc="Parsing HTMLs"):
        json_path = parse_html_file(html_file)
        if json_path:
            processed_json_files.append(json_path)

    return {
        "message": "Data collection and processing complete.",
        "collected_files": collected_html_files,
        "processed_files": processed_json_files
    }

@app.post("/keyword-research")
async def keyword_research(background_tasks: BackgroundTasks):
    """
    Triggers the keyword research process in the background.
    """
    background_tasks.add_task(run_keyword_research)
    return {"message": "Keyword research process started in the background."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
