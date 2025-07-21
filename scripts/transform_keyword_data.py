import json
import csv
import os

# Define file paths
# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct absolute paths for input and output files
# The data directory is two levels up from the scripts directory
data_processed_dir = os.path.abspath(os.path.join(script_dir, '..', 'data', 'data_processed'))
json_file_path = os.path.join(data_processed_dir, 'keyword_search_volume.json')
csv_file_path = os.path.join(data_processed_dir, 'keyword_search_volume_transform.csv')

def transform_json_to_csv():
    """
    Transforms the keyword search volume JSON file into a CSV file.
    The CSV file will have the columns: searchQuery, date, pc, mobile.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {json_file_path} was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file {json_file_path}.")
        return

    # Prepare data for CSV
    rows = []
    for keyword, monthly_data in data.items():
        if not monthly_data:  # Handle cases where a keyword has no data
            continue
        for date, counts in monthly_data.items():
            rows.append({
                'searchQuery': keyword,
                'date': date,
                'pc': counts.get('monthlyProgressPcQcCnt', 0),
                'mobile': counts.get('monthlyProgressMobileQcCnt', 0)
            })

    # Write to CSV
    try:
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['searchQuery', 'date', 'pc', 'mobile']
            # Using QUOTE_NONNUMERIC to quote string fields (searchQuery, date)
            # and leave numeric fields (pc, mobile) unquoted.
            writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Successfully transformed data and saved to {csv_file_path}")
    except IOError as e:
        print(f"Error writing to CSV file {csv_file_path}: {e}")

if __name__ == "__main__":
    transform_json_to_csv() 