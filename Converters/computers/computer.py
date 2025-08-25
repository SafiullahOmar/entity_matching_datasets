import pandas as pd
import re
import os
import csv

def ditto_to_deepmatcher(input_file, output_file):
    deepmatcher_data = []
    row_id = 0

    # Read the lines (tab-separated)
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip().split('\t') for line in f.readlines()]

    print(f"Processing {len(lines)} lines")

    for parts in lines:
        if not parts or len(parts) < 3:
            print(f"Skipping invalid line: {parts}")
            continue

        left_text, right_text, label = parts[0], parts[1], parts[2]

        title_left = extract_title(left_text)
        title_right = extract_title(right_text)

        record = {
            'id': row_id,
            'label': int(label.strip()),
            'title_left': title_left,
            'title_right': title_right
        }

        deepmatcher_data.append(record)
        row_id += 1

    # Convert to DataFrame
    df = pd.DataFrame(deepmatcher_data, columns=['id', 'label', 'title_left', 'title_right'])

    # Save to CSV, quoting title fields (in case they contain commas)
    df.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)

    print(f"Saved {len(df)} records to {output_file}")
    return df

def extract_title(entity_text):
    """
    Extract the 'COL title VAL ...' text, remove quotes, and return clean string.
    """
    pattern = r'COL\s+title\s+VAL\s+(.*?)(?=\s+COL\s+|\s*$)'
    match = re.search(pattern, entity_text)
    if match:
        raw_val = match.group(1)
        # Remove ALL double quotes and language tags like @en
        cleaned = raw_val.replace('"', '').replace('@en', '').replace('@NL', '').replace('@fr', '').strip()
        return cleaned
    return ""

# Example usage
if __name__ == "__main__":
    import sys

    input_file = "train.txt"
    output_file = "train.csv"

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        if not os.path.exists(input_file):
            if os.path.exists(input_file + '.csv'):
                input_file += '.csv'
            elif os.path.exists(input_file + '.txt'):
                input_file += '.txt'

    if len(sys.argv) > 2:
        output_file = sys.argv[2]
        if '.' not in output_file:
            output_file += '.csv'

    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)

    try:
        df = ditto_to_deepmatcher(input_file, output_file)
        print("\nPreview:")
        print(df.head())
    except Exception as e:
        print(f"Conversion error: {e}")
        sys.exit(1)
