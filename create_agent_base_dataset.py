import pandas as pd
import os
import json
import re
from tqdm import tqdm
import csv # Import the csv module

def create_agent_base_dataset_correctly():
    """
    Loads master_analysis_dataset.csv and enriches it with data from the NSIDE
    JSON files by matching postId extracted directly from the JSON filenames.
    """
    # --- Define paths ---
    PROJ_DIR = os.getcwd()
    BLOG_AUTOMATION_DIR = os.path.join(PROJ_DIR, 'blog_automation')
    MASTER_DATASET_PATH = os.path.join(BLOG_AUTOMATION_DIR, 'data', 'data_processed', 'master_analysis_dataset.csv')
    JSON_DIR = os.path.join(BLOG_AUTOMATION_DIR, 'data', 'data_processed', 'NSIDE-postAnalysis-elza79')
    OUTPUT_PATH = os.path.join(BLOG_AUTOMATION_DIR, 'agents', 'agent_base_dataset.csv')

    print("--- Starting: Create Agent Base Dataset (Corrected Logic) ---")

    # --- 1. Load the master dataset ---
    if not os.path.exists(MASTER_DATASET_PATH):
        print(f"FATAL ERROR: Master dataset not found at {MASTER_DATASET_PATH}")
        return
    master_df = pd.read_csv(MASTER_DATASET_PATH)
    print(f"Loaded master dataset. Shape: {master_df.shape}")

    # --- 2. Iterate through JSON files and extract data ---
    if not os.path.isdir(JSON_DIR):
        print(f"FATAL ERROR: JSON directory not found at {JSON_DIR}")
        return
        
    all_posts_data = []
    
    for filename in tqdm(os.listdir(JSON_DIR), desc="Processing JSON files"):
        if not filename.endswith('.json'):
            continue

        # Extract postId from filename (e.g., ..._223127778693.json)
        match = re.search(r'_(\d+)\.json$', filename)
        if not match:
            continue
        
        post_id = int(match.group(1))
        file_path = os.path.join(JSON_DIR, filename)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract post_content
            post_content = data.get('post_content', '')

            # Extract all keywords from category_analysis, checking for None
            category_keywords = []
            category_analysis_data = data.get('category_analysis')
            if category_analysis_data is not None:
                for cat in category_analysis_data:
                    keywords_str = cat.get('keywords', '')
                    if isinstance(keywords_str, str):
                        # Extract only the keyword part, e.g., "치과(7)" -> "치과"
                        keywords = re.findall(r'([가-힣\w\s]+)\(\d+\)', keywords_str)
                        category_keywords.extend(keywords)
            all_category_keywords = ', '.join(list(set(category_keywords))) # Use set for unique keywords

            # Extract all words from morpheme_analysis_details, checking for None
            morpheme_words = []
            morpheme_analysis_data = data.get('morpheme_analysis_details')
            if morpheme_analysis_data is not None:
                for morph in morpheme_analysis_data:
                    words_str = morph.get('words', '')
                    if isinstance(words_str, str):
                        words = re.findall(r'([가-힣\w\s]+)\(\d+\)', words_str)
                        morpheme_words.extend(words)
            all_morpheme_words = ', '.join(list(set(morpheme_words))) # Use set for unique words

            all_posts_data.append({
                'postId': post_id,
                'post_body': post_content,
                'category_keywords': all_category_keywords,
                'morpheme_words': all_morpheme_words
            })
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not process file {filename}. Error: {e}")
            continue

    if not all_posts_data:
        print("Warning: No data extracted from JSON files. The output file will not be enriched.")
        final_df = master_df
    else:
        # --- 3. Merge dataframes ---
        json_df = pd.DataFrame(all_posts_data)
        print(f"Extracted data from {len(json_df)} JSON files.")
        
        final_df = pd.merge(master_df, json_df, on='postId', how='left')
        print(f"Merged master and JSON data. Final shape: {final_df.shape}")

    # --- 4. Save the final dataset ---
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    # Use QUOTE_ALL to ensure all fields are enclosed in quotes, preventing comma issues.
    final_df.to_csv(OUTPUT_PATH, index=False, quoting=csv.QUOTE_ALL)
    
    print(f"--- Finished: New dataset saved to {OUTPUT_PATH} ---")
    print(f"Final columns: {final_df.columns.tolist()}")

if __name__ == '__main__':
    create_agent_base_dataset_correctly() 