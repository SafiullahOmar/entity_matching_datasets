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
    "Song_Name",
    "Artist_Name",
    "Album_Name",
    "Genre",
    "Price",
    "CopyRight",
    "Time",
    "Released"
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
                normalized[key] = "unknown" if key in ["Price", "CopyRight", "Released"] else ""

        return normalized

    def extract_standardized_attributes(self, record: dict) -> dict:
        print("dict", record)

        prompt = f"""
You are a data normalization expert. Your job is to clean and standardize structured data records to improve entity matching performance in machine learning systems.

Apply the following rules consistently:

1. **Preserve Keys**: Keep the original field names and structure.
2. **Standardize Names**:
   - Remove branding fluff, descriptors, marketing suffixes (e.g., "[Deluxe Version]", "(Explicit)").
   - Remove bracketed or parenthetical details unless essential to identity.
   - Strip repetitive artist/brand mentions in titles.
3. **Canonicalize Types/Categories**:
   - Reduce long lists (e.g., "Pop, Rock, Soul, Teen Pop") to the dominant category ("Pop").
   - Normalize naming variants and casing (e.g., "hip-hop / rap" → "Hip-Hop").
   - Use singular, title-case forms.
4. **Numerical Cleanup**:
   - Convert formatted numbers to raw values (e.g., "$1.29" → 1.29, "5.6%" → 5.6).
   - Replace `"Album Only"`, `"N/A"`, `"unknown"`, or `"-"` with `"unknown"`.
5. **Dates**:
   - Normalize release/published dates to `YYYY-MM-DD` format (e.g., "6-May-14" or "May 6 , 2014" → "2014-05-06").
6. **Time Durations**:
   - Preserve time values in `"M:SS"` or `"MM:SS"` format.
7. **Copyrights & Labels**:
   - Remove prefixes like `(C)`, `(P)`, `©`, `℗`, etc.
   - Retain only the label or publisher (e.g., "Sony Music", "Capitol Records").
   - Normalize all company suffixes (e.g., "LLC", "Ltd.") to a canonical form.
8. **Abbreviation Expansion**:
   - Expand common terms (e.g., "Co." → "Company", "Intl" → "International", "St." → "Street").
   - Clean common acronyms and unify to canonical form.
9. **Missing or Corrupted Fields**:
   - Replace empty strings, invalid encodings, or placeholders like `"‰ ãÑ"` with `"unknown"`.
10. **Casing and Punctuation**:
    - Apply title casing to names and proper nouns.
    - Lowercase standard field categories.
    - Remove double spaces, trailing punctuation, or irregular spacing.


_________________________

Input Record:
{json.dumps(record, indent=2)}



📘 Output JSON schema format (always follow this):

{{
  "Song_Name": string,
  "Artist_Name": string,
  "Album_Name": string,
  "Genre": string,
  "Price": float or "unknown",
  "CopyRight": string,
  "Time": string,
  "Released": string (YYYY-MM-DD)
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
