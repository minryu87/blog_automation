import requests
import os
import time
from tqdm import tqdm
from blog_automation.config import NSIDE_API_URL, NSIDE_COOKIE, DATA_RAW_DIR

def fetch_html(post_url: str):
    headers = {
        'Cookie': NSIDE_COOKIE
    }
    data = {
        'post_url': post_url,
        'sample': 'false'
    }
    
    try:
        response = requests.post(NSIDE_API_URL, headers=headers, data=data)
        response.raise_for_status() # Raise an exception for bad status codes
        
        html_content = response.text
        
        # Save the HTML content to a file
        file_name = f"{post_url.replace('/', '_').replace(':', '_')}.html"
        file_path = os.path.join(DATA_RAW_DIR, file_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return file_path
        
    except requests.exceptions.RequestException as e:
        print(f"\nError fetching data for {post_url}: {e}")
        return None

def collect_data_from_urls(post_urls: list[str]):
    collected_files = []
    print("Collecting HTML data...")
    for url in tqdm(post_urls, desc="Fetching HTMLs"):
        file_path = fetch_html(url)
        if file_path:
            collected_files.append(file_path)
        time.sleep(1) # Add a delay to avoid overwhelming the server
    return collected_files 