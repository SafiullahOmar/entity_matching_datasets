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
    "name",
    "addr",
    "city",
    "phone",
    "category",
    "class"
]

class OllamaFeatureExtractor:
    def __init__(self, model_name="llama3.1"):
        self.llm_model = model_name

    def normalize_llm_output(self, response: dict) -> dict:
        """Ensure all expected keys are present and standardized."""
        normalized = {}
        for key in EXPECTED_KEYS:
            if key in response:
                normalized[key] = response[key]
            else:
                normalized[key] = "unknown" if key == "class" else ""

        return normalized

    def extract_standardized_attributes(self, record: dict) -> dict:
        print("dict", record)

        prompt = f"""
You are a data normalization expert. Your job is to clean and standardize structured data records to improve entity matching performance in machine learning systems.

Apply the following universal rules consistently to all fields:

1. **Preserve Keys and Structure**:
   - Keep the original field names and schema exactly as-is.
   - Do not remove or add any keys.

2. **Text Normalization**:
   - Remove escape characters, unmatched quotes, backslashes, and formatting artifacts.
   - Strip leading/trailing punctuation and fix irregular spacing (e.g., double spaces).
   - Normalize punctuation and casing across fields.
   - Apply title case to names, places, products, and entities; lowercase for standard types/categories if applicable.

3. **Abbreviation & Synonym Expansion**:
   - Expand or standardize common abbreviations (e.g., "Co." → "Company", "Intl" → "International", "St." → "Street").
   - Normalize synonyms and variants (e.g., "hip-hop / rap" → "Hip-Hop", "coffee shops/diners" → "Coffee Shop").

4. **Name and Title Cleanup**:
   - Remove marketing fluff, descriptors, and edition tags (e.g., "Deluxe Version", "Explicit", "Ltd.").
   - Remove redundant mentions of artist/brand.
   - Drop bracketed or parenthetical content unless critical for identity.

5. **Categorical Fields (e.g., genre, cuisine, product types)**:
   - Collapse long lists into the dominant category.
   - Normalize to canonical, singular forms with consistent casing.

6. **Numerical Fields**:
   - Convert formatted values to raw numbers (e.g., "$1.29" → 1.29, "5.6%" → 5.6).
   - Standardize unknown or invalid values (e.g., "Album Only", "N/A", "-", "—") to `"unknown"`.

7. **Date Fields**:
   - Normalize all dates to `YYYY-MM-DD` format (e.g., "6-May-14" → "2014-05-06").

8. **Time Durations**:
   - Keep time fields in `M:SS` or `MM:SS` format as applicable.

9. **Location and Address Fields**:
   - Normalize street suffixes and directionals (e.g., "Blvd." → "Boulevard", "NW" → keep only if meaningful).
   - Title-case street and city names, and resolve variants (e.g., "new york" vs "new york city" → "New York").

10. **Phone Numbers**:
    - Standardize to the format `+1-XXX-XXX-XXXX` (or regional equivalent).
    - Replace invalid or incomplete numbers with `"unknown"`.

11. **Legal and Copyright Strings**:
    - Strip prefixes like `(C)`, `©`, `℗`.
    - Retain only the normalized label, publisher, or brand name.

12. **Missing or Corrupted Values**:
    - Replace corrupted encodings (e.g., `"‰ ãÑ"`, `""`, `"?"`, `"null"`) with `"unknown"`.
    
---------------

### Few Shot EXAMPLES 

## Example 1 :

Input:
{{
  "name": "bugsy \\ 's diner",
  "addr": "3555 las vegas blvd. s",
  "city": "las vegas",
  "phone": "702/733 -3111",
  "category": "coffee shops/diners",
  "class": 431
}}

Standardized OUTPUT:
{{
  "name": "Bugsy's Diner",
  "addr": "3555 Las Vegas Boulevard South",
  "city": "Las Vegas",
  "phone": "+1-702-733-3111",
  "category": "Coffee Shop",
  "class": 431
}}


Input:
{{
  "name": "moongate",
  "addr": "3400 las vegas blvd. s.",
  "city": "las vegas",
  "phone": "702-791-7352",
  "category": "chinese",
  "class": 666
}}

Standardized OUTPUT:
{{
  "name": "Moongate",
  "addr": "3400 Las Vegas Boulevard South",
  "city": "Las Vegas",
  "phone": "+1-702-791-7352",
  "category": "Chinese",
  "class": 666
}}


Input:
{{
 "name": "ritz-carlton restaurant",
  "addr": "181 peachtree st.",
  "city": "atlanta",
  "phone": "404-659-0400",
  "category": "french ( classic )",
  "class": 91
}}

Standardized OUTPUT:
{{
   "name": "Ritz-Carlton Restaurant",
  "addr": "181 Peachtree Street",
  "city": "Atlanta",
  "phone": "+1-404-659-0400",
  "category": "French",
  "class": 91
}}


Input:
{{
  "name": "carmine \\ 's",
  "addr": "2450 broadway between 90th and 91st sts .",
  "city": "new york",
  "phone": "212/362 -2200",
  "category": "italian",
  "class": 28
}}

Standardized OUTPUT:
{{
   "name": "Carmine's",
  "addr": "2450 Broadway Between 90th And 91st Streets",
  "city": "New York",
  "phone": "+1-212-362-2200",
  "category": "Italian",
  "class": 28
}}



---------------

Input Record:
{json.dumps(record, indent=2)}



📘 Output JSON schema format (always follow this):

{{
  "name": string,
  "addr": string,
  "city": string,
  "phone": string,
  "category": string,
  "class": int
}}


__________________
⚠️ STRICT FORMAT RULES:
You must respond ONLY with a SINGLE valid JSON object matching the format shown above.

DO NOT include:
- Any explanation, summary, description, or markdown formatting
- Multiple outputs or surrounding text

If multiple examples are shown above, IGNORE them — only process the current input record.

❌ INVALID RESPONSES:
- Do not say “Here is the output”
- Do not return multiple JSONs
- Do not wrap in backticks

✅ VALID RESPONSE:
Return exactly ONE JSON object as the output, following the required schema.
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
