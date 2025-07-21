import pandas as pd
import json
import os
from glob import glob

def analyze_post_data():
    """
    Analyzes post data from JSON files based on a list of post IDs from a CSV file.
    It extracts relevant information from each JSON file and compiles it into a single CSV file.
    """
    # Define paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    input_csv_path = os.path.join(base_dir, 'data', 'data_input', 'post-anallysis-input.csv')
    json_folder_path = os.path.join(base_dir, 'data', 'data_processed', 'NSIDE-postAnalysis-elza79')
    output_csv_path = os.path.join(base_dir, 'data', 'data_processed', 'post_analysis_results.csv')

    # Read the input CSV to get the list of postIds
    try:
        input_df = pd.read_csv(input_csv_path)
        post_ids = input_df['postId'].tolist()
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {input_csv_path}")
        return
    except KeyError:
        print(f"Error: 'postId' column not found in {input_csv_path}")
        return

    all_posts_data = []

    # Process each postId
    for post_id in post_ids:
        # Find the corresponding JSON file
        json_files = glob(os.path.join(json_folder_path, f'*{post_id}.json'))
        if not json_files:
            print(f"Warning: No JSON file found for postId {post_id}")
            continue
        
        json_file_path = json_files[0]

        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error reading or parsing JSON for postId {post_id}: {e}")
            continue

        # Ensure data is not None before proceeding
        if data is None:
            print(f"Warning: Data is None for postId {post_id}")
            continue
            
        # Extract data safely
        text_analysis = data.get('text_analysis') or {}
        morpheme_counts = data.get('morpheme_counts') or {}
        image_analysis = data.get('image_analysis') or {}
        posting_score_details = data.get('posting_score_details') or []
        category_analysis = data.get('category_analysis') or []


        post_data = {
            'postId': post_id,
            'post_title': data.get('post_title'),
            'posting_score': data.get('posting_score'),
            'exposure_score': data.get('exposure_score'),
            'publish_time': data.get('publish_time'),
            'content_with_space': text_analysis.get('content_with_space'),
            'content_without_space': text_analysis.get('content_without_space'),
            'morpheme_count': morpheme_counts.get('morpheme_count'),
            'word_count': morpheme_counts.get('word_count'),
            'valid_images': image_analysis.get('valid_images'),
            'invalid_images': image_analysis.get('invalid_images'),
        }

        # Extract detailed posting scores
        for detail in posting_score_details:
            category = detail.get('category')
            score = detail.get('score')
            if category:
                post_data[f'score_{category}'] = score
        
        # Extract top category
        if category_analysis:
            top_category = category_analysis[0]
            post_data['top_category_name'] = top_category.get('category_name')
            post_data['top_category_percentage'] = top_category.get('percentage')

        all_posts_data.append(post_data)

    # Create DataFrame and save to CSV
    if all_posts_data:
        output_df = pd.DataFrame(all_posts_data)
        
        # Clean columns by removing commas and converting to numeric
        for col in ['content_with_space', 'content_without_space']:
            if col in output_df.columns:
                output_df[col] = pd.to_numeric(output_df[col].astype(str).str.replace(',', ''), errors='coerce')

        # Reorder columns for better readability
        score_cols = sorted([col for col in output_df.columns if col.startswith('score_')])
        other_cols = [col for col in output_df.columns if not col.startswith('score_') and col not in ['postId', 'post_title', 'posting_score', 'exposure_score']]
        
        column_order = ['postId', 'post_title', 'posting_score', 'exposure_score'] + score_cols + other_cols
        # Make sure all columns in column_order exist in the dataframe
        column_order = [col for col in column_order if col in output_df.columns]
        
        output_df = output_df[column_order]
        output_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"Analysis complete. Results saved to {output_csv_path}")
    else:
        print("No data was processed.")

if __name__ == "__main__":
    analyze_post_data() 