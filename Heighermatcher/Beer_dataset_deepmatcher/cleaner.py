import pandas as pd
import os

# 🔧 INPUT: Your dataset files
dataset_files = {
    "train": "train.csv",
    "valid": "valid.csv",
    "test": "test.csv"
}

# ✅ Keep only these fields for HierMatcher
FIELDS_TO_KEEP = [
    "id", "label",
    "left_name", "right_name",
    "left_brewery", "right_brewery",
    "left_primary_style", "right_primary_style",
    "left_secondary_style", "right_secondary_style",
    "left_abv", "right_abv"
]

for split, file_path in dataset_files.items():
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        continue

    print(f"🔄 Processing {file_path}...")

    df = pd.read_csv(file_path)

    # Only keep the valid columns that exist in the DataFrame
    columns_to_keep = [col for col in FIELDS_TO_KEEP if col in df.columns]
    df_clean = df[columns_to_keep]

    output_path = f"{split}_hiermatcher_ready.csv"
    df_clean.to_csv(output_path, index=False)

    print(f"✅ Saved cleaned file: {output_path}")
