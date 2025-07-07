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
    "manufacturer",
    "price"
]

class OllamaFeatureExtractor:
    def __init__(self, model_name="llama3.1"):
        self.llm_model = model_name

    def normalize_llm_output(self, response: dict) -> dict:
        """Ensure all expected keys are present with consistent types and names."""
        key_map = {
            "name": "name",
            "manufacturer": "manufacturer",
            "price": "price"
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
    

        prompt = f"""
You are a data normalization expert. Your job is to clean and standardize structured data records to improve entity matching performance in machine learning systems.
---
Your Task:
Clean the input record according to the following universal rules. Then return the normalized version using the exact same schema and field names (keys).
---
### Normalization Rules

#### 1. Preserve Schema
- Use the exact input field names and structure.
- Do NOT add, remove, or rename keys.

#### 2. Text Normalization
- Remove escape characters, unmatched quotes, slashes, and formatting artifacts.
- Strip leading/trailing punctuation and fix irregular spacing (e.g., double spaces).
- Fix punctuation and casing (e.g., "john doe " → "John Doe").
- Apply Title Case to names, entities, brands, and places. Use lowercase for generic types (e.g., categories, genres).
-Eliminate repeated words or phrases (e.g., "Google Google" → "Google").

#### 3. Abbreviations & Synonyms
- Expand common abbreviations (e.g., “Co.” → “Company”, “St.” → “Street”).
- Standardize synonyms (e.g., “coffee shops/diners” → “Coffee Shop”, “hip-hop / rap” → “Hip-Hop”).

#### 4. Canonicalization
- Normalize brands and entities (e.g., "Google LLC", "google inc." → "Google").
- Remove branding fluff, edition tags (e.g., “Ltd.”, “Deluxe”), and bracketed content unless necessary.
#### 3. Product Title Cleaning (`*_title`)
- Remove platform tags in brackets or parentheses (e.g., "[Mac]", "(Win/Mac)", "(DVD-ROM)").
- Remove licensing or packaging fluff such as:
  - "Upgrade", "Edition", "Package", "PC", "CD", "Version", "Software", "Box", "Retail", "OEM", "Download"
  - Example: "Microsoft Office 2007 Upgrade Package CD" → "Microsoft Office 2007"
- Expand common short forms:
  - "Pro" → "Professional"
  - "CS3" → "Creative Suite 3"
- Remove promotional/educational tags like "Educational Discount", "Student/Teacher", unless critical to product identity.

#### 4. Manufacturer Canonicalization (`*_manufacturer`)
- Normalize variations to canonical names using this dictionary (extend as needed):
  - "Adobe", "Adobe Inc.", "Adobe Systems" → "Adobe"
  - "Microsoft", "Microsoft Licenses", "Microsoft Software" → "Microsoft"
  - "Steinberg", "Steinberg GmbH" → "Steinberg"
  - "Topics Entertainment" → "Topics Entertainment"
  - "Intuit Inc." → "Intuit"
  - "Nolo Press" → "Nolo"
- If missing or empty, set to `"unknown"`

#### 5. Price Cleaning (`*_price`)
- Convert prices like "$29.99" or "29.0" to float format: `29.99`
- Ensure two decimal precision for all prices
- If price is missing, empty, or invalid, set to `"unknown"`

#### 6. Handling Unknown or Missing Values
- For any empty field, or field with values like "", null, "n/a", "-", replace with `"unknown"`.
-----

### EXAMPLES OF GOOD STANDARDIZATION:

## Example 1:
Input:

title: microsoft visio standard 2007 version upgrade
manufacturer: microsoft
price: 129.95

Standardized Output:
{{
  "title": "Microsoft Visio Standard 2007",
  "manufacturer": "Microsoft",
  "price": 129.95
}}

## Example 2:
Input:
title: adobe after effects professional 7.0
manufacturer: adobe
price: 999.0

Standardized Output:
{{
  "title": "Adobe After Effects Professional 7.0",
  "manufacturer": "Adobe",
  "price": 999.00
}}

## Example 3:
Input:
title: motu digital performer 5 digital audio software competitive upgrade ( mac only )
manufacturer: motu
price: 395.0

Standardized Output:
{{
  "title": "MOTU Digital Performer 5",
  "manufacturer": "MOTU",
  "price": 395.00
}}

## Example 4:
Input:
title: illustrator cs3 13 mac ed 1u
manufacturer: adobe-education-box
price: 199.0

Standardized Output:
{{
  "title": "Adobe Illustrator Creative Suite 3 For Mac",
  "manufacturer": "Adobe Education",
  "price": 199.00
}}

---

Now process this record:


Record:
{json.dumps(record, indent=2)}


📘 Output JSON schema format (always follow this):

{{
  "title": string,
  "manufacturer": string,
  "price": number,
  
}}

Return only valid JSON with standardized values. Do not include backticks, markdown, or explanations. Remember to ALWAYS split complex styles into primary_style and secondary_style components.

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
