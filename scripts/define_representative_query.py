import pandas as pd
import os

def define_representative_queries():
    """
    Analyzes post-searchQuery.csv to determine the single, most impactful 
    search query for each blog post based on total search inflow.

    This simplifies the 1-to-N post-to-query relationship into a 1-to-1 
    relationship for easier comparison with competitors.
    """
    # --- 1. Define File Paths ---
    # The script is located in blog_automation/scripts. We need to navigate
    # to the parent directory (blog_automation) and then down to the data directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    blog_automation_dir = os.path.join(script_dir, '..')
    
    input_path = os.path.join(blog_automation_dir, 'data', 'data_input', 'post-searchQuery.csv')
    output_dir = os.path.join(blog_automation_dir, 'data', 'data_processed')
    output_path = os.path.join(output_dir, 'representative_queries.csv')

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading data from: {input_path}")
    df = pd.read_csv(input_path)

    # --- 2. Filter out Brand Keywords ---
    # To analyze pure SEO performance, we exclude queries containing the brand name.
    print(f"Original row count: {len(df)}")
    brand_name = '내이튼'
    df_non_brand = df[~df['searchQuery'].str.contains(brand_name, na=False)].copy()
    print(f"Row count after filtering out brand keyword '{brand_name}': {len(df_non_brand)}")

    # --- 3. Aggregate Inflow per Query ---
    # A single post can have multiple entries for the same query over different months.
    # We sum the inflow to get the total impact of each query on a post.
    print("Aggregating search inflow for each post and query...")
    query_inflow_totals = df_non_brand.groupby(['postId', 'searchQuery'])['searchInflow'].sum().reset_index()

    # --- 4. Find the Top Query for Each Post ---
    # For each postId, find the query that generated the most total inflow.
    print("Identifying the representative query with the highest inflow for each post...")
    # Get the index of the row with the maximum searchInflow for each postId
    # Drop posts that have no non-brand queries left
    idx = query_inflow_totals.groupby(['postId'])['searchInflow'].idxmax()
    df_representative = query_inflow_totals.loc[idx]

    # --- 5. Clean and Save the Results ---
    # Rename columns for clarity
    df_representative = df_representative.rename(columns={
        'searchQuery': 'representative_query',
        'searchInflow': 'total_inflow_from_representative_query'
    })

    # Save the resulting dataframe
    df_representative.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print("\n--- Success! ---")
    print(f"Processed {len(df_representative)} unique posts.")
    print(f"Representative queries saved to: {output_path}")
    print("\n--- Sample Output ---")
    print(df_representative.head())
    print("---------------------\n")


if __name__ == '__main__':
    define_representative_queries() 