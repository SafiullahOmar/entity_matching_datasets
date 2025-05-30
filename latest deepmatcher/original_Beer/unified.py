import pandas as pd
import ollama
from tqdm import tqdm
import re
import json
import os
from examples import Beer_Fewshot_exampels
from output_strucutres import Beer_output

# Define the required schema
EXPECTED_KEYS = [
    "name",
    "brewery",
    "primary_style",
    "secondary_style",
    "abv",
    "is_amber",
    "is_ale",
    "is_lager",
    "is_imperial",
    "special_ingredients"
]

class OllamaFeatureExtractor:
    def __init__(self, model_name="llama3.1"):
        self.llm_model = model_name

    def normalize_llm_output(self, response: dict) -> dict:
        """Ensure all expected keys are present with consistent types and names."""
        key_map = {
            "Beer_Name": "name",
            "Brew_Factory_Name": "brewery",
            "Style": "primary_style"
        }

        normalized = {}

        # Map and rename keys
        for key, value in response.items():
            std_key = key_map.get(key, key)
            normalized[std_key] = value

        # Fill in missing keys
        for key in EXPECTED_KEYS:
            if key not in normalized:
                if key == "abv":
                    normalized[key] = "unknown"
                elif key.startswith("is_"):
                    normalized[key] = False
                else:
                    normalized[key] = "unknown"

        return normalized

    def extract_standardized_attributes(self, record: dict, output=Beer_output, fewshot=Beer_Fewshot_exampels) -> dict:
        print("dict", record)

        prompt = f"""
You are a data normalization expert. Your job is to clean and standardize structured data records for entity matching:

Instructions:
1. Keep the original field names (keys), but normalize all values.
2. Standardize entity names by removing branding fluff, descriptors, repeated words, and unnecessary modifiers.
3. Normalize category, type, or style fields to common canonical forms (e.g., “rock/pop” → “Rock”).
4. Convert numeric fields (e.g., percentages, prices, weights, counts) into plain numbers (e.g., "5.6%" → 5.6, "$3.00" → 3.0).
5. For missing, invalid, or placeholder values (e.g., "-", "", "unknown", "N/A"), replace them with the string "unknown".
6. Fix inconsistent formatting, punctuation, and casing. Ensure proper capitalization of names, titles, and categories.
7. Normalize abbreviations to full form (e.g., "st" → "Street", "dept" → "Department", "intl" → "International").
8. Canonicalize and normalize entity values (e.g., merge variants like "google", "Google LLC", and "Google Inc." into "Google"). Expand abbreviations (e.g., "Co." → "Company") and remove irrelevant, duplicated, or marketing-heavy terms.


-------------------------

Record:
{json.dumps(record, indent=2)}



📘 Output JSON schema format (always follow this):

{{
  "name": string,
  "brewery": string,
  "primary_style": string,
  "secondary_style": string or null,
  "abv": float or "unknown",
  "is_amber": boolean,
  "is_ale": boolean,
  "is_lager": boolean,
  "is_imperial": boolean,
  "special_ingredients": string or null
}}


__________________

⚠️ OUTPUT RULES — STRICTLY FOLLOW:
- Output must be valid JSON.
- Do NOT include backticks, explanations, markdown, or anything outside the JSON object.
- Do NOT say "Here is the output" ,"Note: I've normalized" or anything similar.
- Just return the JSON object. No comments, headers, or notes.

"""
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response["message"]["content"].strip()

            if content.startswith("```"):
                content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
                content = re.sub(r"```$", "", content).strip()

            print("record:", content)
            parsed = json.loads(content)
            return self.normalize_llm_output(parsed)

        except json.JSONDecodeError as jde:
            print(f"❌ JSON decode error: {jde}")
            print("⚠️ Content that failed parsing:", repr(content))
            return self.normalize_llm_output({})
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return self.normalize_llm_output({})

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
