import pandas as pd
import re
import csv
import sys

# === Regex pattern for COL/VAL fields ===
FIELD_RE = re.compile(r'\bCOL\s+([A-Za-z0-9_]+)\s+VAL\b', re.IGNORECASE)

def robust_parse_col_val(text: str) -> dict:
    """
    Parse 'COL <key> VAL <value>' segments robustly.
    Handles inconsistent spacing, stray words, and embedded 'COL' tokens.
    """
    if not text:
        return {}

    matches = list(FIELD_RE.finditer(text))
    record = {}

    for i, m in enumerate(matches):
        key = m.group(1).strip().lower()  # normalize key name
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        val = text[start:end].strip()
        # Clean spacing, punctuation, and stray symbols
        val = re.sub(r'\s+', ' ', val).strip(' |;,:')
        record[key] = val

    return record


def parse_tabbed_file(lines, output_file, preferred_order=None):
    """
    Convert a tab-separated COL/VAL dataset to DeepMatcher-compatible CSV.
    Each line in input = "<left record>\t<right record>\t<label>"
    """
    data = []
    row_id = 0
    all_fields = set()

    # === First pass: find all field names ===
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        parts = re.split(r'\t+', line)
        if len(parts) != 3:
            continue
        left, right, _ = parts
        all_fields.update(robust_parse_col_val(left).keys())
        all_fields.update(robust_parse_col_val(right).keys())

    # Order fields cleanly
    if preferred_order:
        preferred = [f for f in preferred_order if f in all_fields]
        the_rest = sorted(f for f in all_fields if f not in preferred)
        ordered_fields = preferred + the_rest
    else:
        ordered_fields = sorted(all_fields)

    # === Second pass: build rows ===
    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        parts = re.split(r'\t+', line)
        if len(parts) != 3:
            print(f"Skipping malformed line (not 3 tab-separated parts): {line[:120]}...", file=sys.stderr)
            continue

        left_text, right_text, label = parts
        try:
            label = int(label)
        except ValueError:
            print(f"Skipping line with invalid label: {label!r}", file=sys.stderr)
            continue

        left_fields = robust_parse_col_val(left_text)
        right_fields = robust_parse_col_val(right_text)

        row = {'id': row_id, 'label': label}
        for field in ordered_fields:
            row[f'left_{field}']  = left_fields.get(field, "")
            row[f'right_{field}'] = right_fields.get(field, "")

        data.append(row)
        row_id += 1

    # === Build DataFrame ===
    df = pd.DataFrame(data)

    # Optional: normalize 'year' fields to numeric 4-digit form
    for side in ('left', 'right'):
        col = f'{side}_year'
        if col in df.columns:
            df[col] = df[col].astype(str).str.extract(r'(\d{4})', expand=False).fillna("")

    # === Save to CSV ===
    df.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"✅ Saved {len(df)} records to {output_file}")
    return df


# === Entry point ===
if __name__ == "__main__":
    input_file = "test.txt"   # your COL/VAL dataset
    output_file = "test.csv"  # output DeepMatcher-compatible CSV

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Field order for academic metadata
    parse_tabbed_file(
        lines,
        output_file,
        preferred_order=['title', 'authors', 'venue', 'year']
    )
