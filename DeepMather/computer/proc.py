import pandas as pd
import ollama
from tqdm import tqdm
import re
import json
import os
# from examples import Beer_Fewshot_exampels
# from output_strucutres import Beer_output

# Define the required schema
EXPECTED_KEYS = [
    "title"
]

class OllamaFeatureExtractor:
    def __init__(self, model_name="llama3.1"):
        self.llm_model = model_name

    def normalize_llm_output(self, response: dict) -> dict:
        """Ensure all expected keys are present with consistent types and names."""
        key_map = {
            "title": "title",
        }

        normalized = {}

        # Map and rename keys
        for key, value in response.items():
            std_key = key_map.get(key, key)
            normalized[std_key] = value

        # Fill in missing keys
        # for key in EXPECTED_KEYS:
        #     if key not in normalized:
        #         if key == "abv":
        #             normalized[key] = "unknown"
        #         elif key.startswith("is_"):
        #             normalized[key] = False
        #         else:
        #             normalized[key] = "unknown"

        return normalized

    def extract_standardized_attributes(self, record: dict) -> dict:
    
        print("passed dict",record)
        prompt = f"""
You are a record normalizer optimizing for BERT-based similarity matching. Your job is to transform raw records into a standardized format that maximizes match accuracy.

## CORE OBJECTIVES:
1. Maintain consistent COL/VAL structure
4. Normalize style descriptions
5. Preserve 0/1 pair matching indicators

## TRANSFORMATION RULES:

### Structure and Formatting:
- Keep the exact COL/VAL structure with ONE tab character between record pairs
- Never add trailing tags at record ends
- Ensure correct spacing around tags

### Match-Sensitive Transformations:
- Preserve distinctive terms that differentiate similar entity
- Apply consistent transformations to both records in a pair
- When entity pairs have a similarity indicator of 1, ensure matching fields have similar tagging

### Normalization Rule for the title:
-Identify and extract key attributes like brand, product type, capacity, generation/model number, and variant details.
-Normalize spelling, remove redundant vendor or website suffixes (e.g., "TWEAKERS@NL", "SCAN UK", "macofalltrades", etc.).
-Preserve exact numeric values (e.g., sizes in GB, GHz, model numbers) and format product names consistently (e.g., "HP Xeon X7560 2.26GHz").
-Translate foreign or mixed language segments to English, and remove duplicate or irrelevant phrases.
-Output only a cleaned COL/VAL title line in the same format as input.
-----

### Important:
- Do not guess missing values. Only extract fields if they are explicitly present.

### EXAMPLES OF GOOD STANDARDIZATION:


## Example 1:
Input:

title: "XTA-3510-73-GB-10K Sun 73-GB 10K HP FC-AL HDD" "Null"


Standardized Output:
{{
  "title": "XTA-3510-73-GB-10K Sun 73GB 10K HP FC-AL HDD ,MODEL_NO: XTA-3510-73-GB-10K, BRAND: Sun, STORAGE: 73GB, SPEED: 10K RPM, PRODUCT_TYPE: HDD, INTERFACES: FC-AL"
}}

## Example 2:
Input:
title: "WD Gold 10TB 3.5\" 7200RPM 256MB Cache Datacenter Hard Drive (WD101KRYZ)"@en "▷ WD … | OcUK"@en

Standardized Output:
{{
  "title": "WD Gold 10TB 3.5" 7200RPM 256MB Cache Datacenter Hard Drive (WD101KRYZ),BRAND: WD, PRODUCT_TYPE: Hard Drive, MODEL_NO: WD101KRYZ, STORAGE: 10TB, FORM_FACTOR: 3.5 inch, SPEED: 7200RPM, MEMORY: 256MB Cache"
}}

## Example 3:
Input:
title: "WD 6TB Gold Datacenter HDD/Hard Drive WD6002FRYZ"@en WD6002FRYZ LN72392 | SCAN UK"@en

Standardized Output:
{{
  "title":"WD 6TB Gold Datacenter HDD Hard Drive WD6002FRYZ,BRAND: WD, PRODUCT_TYPE: Hard Drive, MODEL_NO: WD6002FRYZ, STORAGE: 6TB"
}}

## Example 4:
Input:
title: "SanDisk Standard"@en-US "Superwarehouse - SanDisk Standard Sandisk SDSDB-016G-B35"@en-US


Standardized Output:
{{
  "title": "SanDisk Standard SDSDB-016G-B35 SD,BRAND: SanDisk, PRODUCT_TYPE: SD Card, MODEL_NO: SDSDB-016G-B35, STORAGE: 16GB"
}}

## Example 5:
Input:
title: "WD Black 2TB Performance Desktop Hard Disk Drive - 7200 RPM SATA 6 Gb/s 64MB Cache 3.5 Inch WD2003FZEX"


Standardized Output:
{{
  "title":  "WD Black 2TB 3.5\" 7200RPM SATA 6Gb/s 64MB HDD WD2003FZEX,BRAND: WD, PRODUCT_TYPE: HDD, MODEL_NO: WD2003FZEX, STORAGE: 2TB, FORM_FACTOR: 3.5 inch, SPEED: 7200RPM, INTERFACES: SATA 6Gb/s, MEMORY: 64MB Cache"
}}

## Example 6:

Input:
title: "WD Gold 10TB 3.5\" 7200RPM 256MB Cache Datacenter Hard Drive (WD101KRYZ)"@en "▷ WD … | OcUK"@en

Standardized Output:
{{
  "title": "WD Gold 10TB 3.5\" 7200RPM 256MB Cache Datacenter Hard Drive (WD101KRYZ),BRAND: WD, PRODUCT_TYPE: Hard Drive, MODEL_NO: WD101KRYZ, STORAGE: 10TB, FORM_FACTOR: 3.5 inch, SPEED: 7200RPM, MEMORY: 256MB Cache"
}}

---

Now process this record:


Record:
{json.dumps(record, indent=2)}


📘 Output JSON schema format (always follow this):

{{
  "title": string
  
}}

Return only valid JSON with standardized values. Do not include backticks, markdown, or explanations. Remember to ALWAYS split complex styles into primary_style and secondary_style components.

"""
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[ {
                        "role": "system",
                        "content": (
                            "You are entity matcher for the deepmatcher. Do not explain. "
                            "Do not describe anything. Do not say 'Output:' or '<think>'. "
                            "Do not provide reasoning, steps, formatting explanation, or notes. "
                            "If you violate this, your output will be rejected."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }]
            )
            content = response["message"]["content"].strip()
            print("output is",content)
            if content.startswith("```"):
                content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
                content = re.sub(r"```$", "", content).strip()

            
            parsed = json.loads(content)
            return self.normalize_llm_output(parsed)

        except json.JSONDecodeError as jde:
            print(f"❌ JSON decode error: {jde}")
            print("⚠️ Content that failed parsing:", repr(content))
            return record
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return record

    def split_record(self, row: dict, side: str) -> dict:
        """Extract left or right side sub-record"""
        return {col[len(f"{side}_"):]: row[col] for col in row if col.startswith(f"{side}_")}

    def process_dataset(self, input_csv, output_csv):
        print(f"📄 Reading data from {input_csv}...")
        df = pd.read_csv(input_csv)
        all_rows = []

        for _, row in tqdm(df.iterrows(), total=len(df)):
            row_dict = row.to_dict()

            left_input = self.split_record(row_dict, "left")
            right_input = self.split_record(row_dict, "right")

            left_cleaned = self.extract_standardized_attributes(left_input)
            right_cleaned = self.extract_standardized_attributes(right_input)

            # Construct the new row with normalized fields only
            new_row = {
                "id": row_dict.get("id"),
                "label": row_dict.get("label")
            }

            for k, v in left_cleaned.items():
                new_row[f"left_{k}"] = v
            for k, v in right_cleaned.items():
                new_row[f"right_{k}"] = v

            all_rows.append(new_row)

        enriched_df = pd.DataFrame(all_rows)
        print(f"💾 Saving enriched data to {output_csv}")
        enriched_df.to_csv(output_csv, index=False)

def main():
    extractor = OllamaFeatureExtractor()

    for split in ['train', 'valid', 'test']:
        input_file = f"{split}.csv"
        output_file = f"{split}_enriched.csv"
        if os.path.exists(input_file):
            print(f"\n🟡 Processing {split}...")
            extractor.process_dataset(input_file, output_file)
        else:
            print(f"⚠️  {input_file} not found, skipping...")

if __name__ == "__main__":
    main()
