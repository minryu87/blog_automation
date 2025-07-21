import pandas as pd
import os

def standardize_agent_dataset_columns():
    """
    Renames the columns of the agent_base_dataset.csv to a clear, consistent
    English snake_case standard.
    """
    # --- 1. Define Paths and Column Mapping ---
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dataset_path = os.path.join(base_dir, 'agents', 'agent_base_dataset.csv')
    
    column_mapping = {
        'postId': 'post_id',
        'posting_score': 'posting_level',
        'score_가독성': 'readability_score',
        'score_주제': 'topic_focus_score',
        'score_콘텐츠': 'content_quality_score',
        'score_키워드': 'keyword_usage_score',
        'score_형태소': 'morpheme_score',
        'score_후기/정보': 'review_info_score',
        'content_with_space': 'char_count_with_space',
        'content_without_space': 'char_count_without_space',
        'valid_images': 'valid_image_count',
        'invalid_images': 'invalid_image_count',
        'sum_유입': 'non_brand_inflow',
        'sum_조회': 'total_views',
        'average_C': 'non_brand_average_ctr'
        # Columns not in the mapping will remain unchanged.
    }

    # --- 2. Load, Rename, and Save Data ---
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: The dataset file was not found at {dataset_path}")
        return

    # Get the list of columns before renaming
    original_columns = df.columns.tolist()

    df.rename(columns=column_mapping, inplace=True)
    
    # Get the list of columns after renaming
    new_columns = df.columns.tolist()

    # --- 3. Save the updated DataFrame ---
    df.to_csv(dataset_path, index=False, encoding='utf-8-sig')

    print("--- Column Standardization Complete ---")
    print(f"Dataset updated successfully at: {dataset_path}")
    print("\nColumn changes:")
    for old, new in zip(original_columns, new_columns):
        if old != new:
            print(f"  - '{old}' -> '{new}'")

if __name__ == "__main__":
    standardize_agent_dataset_columns() 