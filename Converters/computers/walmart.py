import pandas as pd
import re
import csv
import sys

FIELD_RE = re.compile(r'\bCOL\s+([A-Za-z0-9_]+)\s+VAL\b', re.IGNORECASE)

def robust_parse_col_val(text: str) -> dict:
    """
    Parse 'COL <key> VAL <value>' segments without being confused by 'COL' inside values.
    Uses explicit boundary matches and index slicing between successive COL/VAL markers.
    """
    if not text:
        return {}

    matches = list(FIELD_RE.finditer(text))
    record = {}

    for i, m in enumerate(matches):
        key = m.group(1).strip().lower()  # normalize keys
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        val = text[start:end].strip()
        # collapse internal whitespace; trim stray separators
        val = re.sub(r'\s+', ' ', val).strip(' |;,:')
        record[key] = val

    return record

def parse_tabbed_file(lines, output_file, preferred_order=None):
    """
    lines: iterable of strings. Each line = "<left>\t<right>\t<label>"
    preferred_order: an optional ordered list of expected fields to pin column order (e.g., ['title','category','brand','modelno','price'])
    """
    data = []
    row_id = 0
    all_fields = set()

    # First pass: collect all unique field names
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        parts = re.split(r'\t+', line)  # tolerate multiple tabs
        if len(parts) != 3:
            # silently skip in pass 1
            continue
        left, right, _ = parts
        all_fields.update(robust_parse_col_val(left).keys())
        all_fields.update(robust_parse_col_val(right).keys())

    # lock a stable, sensible field order
    if preferred_order:
        preferred = [f for f in preferred_order if f in all_fields]
        the_rest = sorted(f for f in all_fields if f not in preferred)
        ordered_fields = preferred + the_rest
    else:
        ordered_fields = sorted(all_fields)

    # Second pass: build rows
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

    df = pd.DataFrame(data)

    # If you want numeric price inference, uncomment next 6 lines
    # for side in ('left', 'right'):
    #     col = f'{side}_price'
    #     if col in df.columns:
    #         df[col] = (
    #             df[col].str.extract(r'([0-9]+(?:\.[0-9]+)?)', expand=False)
    #                   .astype(float, errors='ignore')
    #         )

    #df.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
    df.to_csv(output_file, index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
    print(f"✅ Saved {len(df)} records to {output_file}")
    return df

# === Entry point ===
if __name__ == "__main__":
    # Example: reading from your file
    input_file = "valid.txt"
    output_file = "valid.csv"

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Pin common field order so CSV columns look tidy
    parse_tabbed_file(lines, output_file, preferred_order=['title','category','brand','modelno','price'])
