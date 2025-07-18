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
    def __init__(self, model_name="llama3.1:latest"):
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
You are a record normalizer optimizing product titles for entity matching using DeepMatcher. You will receive a pair of computer product titles and a `label` indicating whether they refer to the same product (`label = 1`) or not (`label = 0`).

Your goal is to return **cleaned, normalized versions** of each title (`left_title` and `right_title`) as free-text strings, in the same style and format as the input, but cleaned for matching purposes.

---

## 🧹 General Cleaning and Normalization Rules:

- Identify and preserve key attributes like brand, product type, storage/capacity, model number, generation, and variant details.
- Preserve exact numeric values (e.g., 2TB, 7200RPM, 3.5in, E5607, 8GB, 2666MHz, 10K, 128GB).
- **Never remove or alter alphanumeric model numbers or part codes** (e.g., ST31000524NS, 658071-B21, MZ-N5E1T0BW, WD20EFRX, CT51264BF160B).
- If a model number appears multiple times, retain one clean instance.
- Remove redundant vendor or website suffixes (e.g., “macofalltrades”, “CDW.com”, “SCAN UK”, “TWEAKERS”, “OcUK”, “Superwarehouse”).
- Remove unnecessary tokens like “null”, “price”, “new”, “wholesale”, “@en”, “LNxxxxx”.
- Deduplicate repeated segments or trailing IDs.
- Translate foreign or mixed-language text to English.
- Never guess or add fields not already present in the input.
- Keep titles in free-text format, not structured.
- Return exactly **one cleaned line per title**, no markdown or extra commentary.

---

## 🧠 Match-Sensitive Rules:

- If `label = 1` (match):
  - Align the terminology, phrasing, and formatting of both records.
  - Unify units (e.g., "3.5 inch" → "3.5in", "7200 RPM" → "7200RPM").
  - Use consistent ordering and style across both sides.

- If `label = 0` (non-match):
  - Normalize lightly.
  - Preserve differences in brand, product type, model no, phrasing, etc.
  - Do **not** over-align structure or style across both sides.
  - Even in label = 0 cases, always retain product codes or model numbers, such as:
  - Part numbers (359461-007, 540-5629)
  - Internal model codes (MZ-N5E1T0BW, ST31000524NS, CMK64GX4M4A2666C16)
  - These identifiers are critical for DeepMatcher, even if records are not aligned.

---

Now process this record:


Record:
{json.dumps(record, indent=2)}


📘 Output JSON schema format (always follow this):

{{
  "left_title": string,
  "right_title": string
  
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
            record_pair = {
                    "left_title": row_dict.get("left_title", ""),
                    "right_title": row_dict.get("right_title", ""),
                    "label": row_dict.get("label", 0)
                }
            cleaned_pair = self.extract_standardized_attributes(record_pair)
            
             
            new_row = {
                "id": row_dict.get("id"),
                "label": row_dict.get("label"),
                "left_title": cleaned_pair.get("left_title", record_pair["left_title"]),
                "right_title": cleaned_pair.get("right_title", record_pair["right_title"])
            }
            all_rows.append(new_row)

            # left_input = self.split_record(row_dict, "left")
            # right_input = self.split_record(row_dict, "right")

            # left_cleaned = self.extract_standardized_attributes(left_input)
            # right_cleaned = self.extract_standardized_attributes(right_input)

            # # Construct the new row with normalized fields only
            # new_row = {
            #     "id": row_dict.get("id"),
            #     "label": row_dict.get("label")
            # }

            # for k, v in left_cleaned.items():
            #     new_row[f"left_{k}"] = v
            # for k, v in right_cleaned.items():
            #     new_row[f"right_{k}"] = v

            # all_rows.append(new_row)

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
