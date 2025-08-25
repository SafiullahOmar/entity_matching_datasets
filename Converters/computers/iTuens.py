import pandas as pd
import re
import csv

def robust_parse_col_val(text):
    """
    Improved COL/VAL parser to avoid splitting on fake 'COL' inside values.
    Uses index-based slicing instead of greedy regex.
    """
    pattern = re.compile(r'COL\s+([^\s]+)\s+VAL')
    matches = list(pattern.finditer(text))
    record = {}

    for i in range(len(matches)):
        key = matches[i].group(1).strip()
        start = matches[i].end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        val = text[start:end].strip()
        record[key] = val

    return record

def parse_tabbed_file(lines, output_file):
    data = []
    row_id = 0
    all_fields = set()

    # First pass: collect all unique field names
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) != 3:
            continue
        left, right, _ = parts
        all_fields.update(robust_parse_col_val(left).keys())
        all_fields.update(robust_parse_col_val(right).keys())

    all_fields = sorted(all_fields)

    # Second pass: extract structured data
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) != 3:
            print(f"Skipping malformed line: {line[:80]}...")
            continue

        left_text, right_text, label = parts
        try:
            label = int(label)
        except ValueError:
            print(f"Skipping line with invalid label: {label}")
            continue

        left_fields = robust_parse_col_val(left_text)
        right_fields = robust_parse_col_val(right_text)

        row = {
            'id': row_id,
            'label': label
        }

        for field in all_fields:
            row[f'left_{field}'] = left_fields.get(field, "")
            row[f'right_{field}'] = right_fields.get(field, "")

        data.append(row)
        row_id += 1

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"✅ Saved {len(df)} records to {output_file}")
    return df

# === Entry point ===
if __name__ == "__main__":
    input_file = "test.txt"
    output_file = "test_1.csv"

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    parse_tabbed_file(lines, output_file)
