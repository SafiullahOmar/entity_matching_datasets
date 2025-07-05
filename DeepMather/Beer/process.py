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

#### 5. Categorical Values
- Collapse compound categories to a single dominant form (e.g., "rock/pop" → "Rock").
- Use consistent singular and canonical category values.
- If applicable, split into primary_style and secondary_style based on specificity.

#### 6. Numerical Fields
- Convert percentages (e.g., "5.6%") → numeric float (5.6).
- Convert prices (e.g., "$3.00") → float (3.0).
- Replace invalid or placeholder values ("-", "", "N/A", "unknown") with "unknown".

#### 7. Dates and Times
- Dates: Format to YYYY-MM-DD (e.g., “5-Jul-25” → “2025-07-05”).
- Times: Use M:SS or MM:SS (e.g., "2:34").

#### 8. Phone Numbers
- Format as +1-XXX-XXX-XXXX (for US).
- Replace invalid, partial, or corrupted numbers with "unknown".

#### 9. Address Fields
- Normalize street suffixes (e.g., “Blvd.” → “Boulevard”).
- Use title case and fix spacing/punctuation.

#### 10. Missing or Corrupt Values
- Replace malformed entries like null, ?, ‰, "" with "unknown".

#### 11. Style Field Normalization
- Always split complex styles into `primary_style` and `secondary_style`:
  - Choose the most specific or defining type as `primary_style`
  - Move descriptors (e.g., nationality, strength, barrel-aging) to `secondary_style`
- Examples:
  - "American Amber / Red Ale" → primary_style: "Red Ale", secondary_style: "Amber"
---

### EXAMPLES OF GOOD STANDARDIZATION:

## Example 1: Style breakdown for Red Ales
Input 1:
Beer Name: Red Amber Ale
Brewery: Example Brewing
Style: American Amber / Red Ale
ABV: 5.5%

Input 2:
Beer Name: Red Amber
Brewery: Example
Style: Amber Ale
ABV: 5.5%

Standardized Output (for both):
{{
  "name": "Red Amber Ale",
  "brewery": "Example Brewing",
  "primary_style": "Red Ale",
  "secondary_style": "Amber",
  "abv": 5.5,
  "is_amber": "true",
  "is_ale": "true",
  "is_lager": "false",
  "is_imperial": "false",
  "special_ingredients": "none"
}}

## Example 2: Style breakdown for IPAs
Input 1:
Beer Name: Hazy Double IPA
Brewery: Craft Brewers
Style: New England Double India Pale Ale
ABV: 8.2%

Input 2:
Beer Name: Hazy DIPA
Brewery: Craft
Style: Hazy Imperial IPA
ABV: 8.0%

Standardized Output (for both):
{{
  "name": "Hazy Double IPA",
  "brewery": "Craft Brewers",
  "primary_style": "IPA",
  "secondary_style": "Double Hazy",
  "abv": 8.1,
  "is_amber": "false",
  "is_ale": "true",
  "is_lager": "false",
  "is_imperial": "true",
  "special_ingredients": "none"
}}

## Example 3: Style breakdown for Stouts
Input 1:
Beer Name: Chocolate Coffee Stout
Brewery: Dark Brewing Co.
Style: American Imperial Stout
ABV: 10.5%

Input 2:
Beer Name: Chocolate Coffee Imperial Stout
Brewery: Dark Brewing
Style: Russian Imperial Stout
ABV: 10.5%

Standardized Output (for both):
{{
  "name": "Chocolate Coffee Stout",
  "brewery": "Dark Brewing",
  "primary_style": "Stout",
  "secondary_style": "Imperial",
  "abv": 10.5,
  "is_amber": "false",
  "is_ale": "true",
  "is_lager": "false",
  "is_imperial": "true",
  "special_ingredients": "chocolate, coffee"
}}

## Example 4: Matching beers with different style descriptions
Input 1:
Beer Name: Sanibel Red Island Ale
Brewery: Point Ybel Brewing Company
Style: American Amber / Red Ale
ABV: 5.60%

Input 2:
Beer Name: Point Ybel Sanibel Red Island Ale
Brewery: Point Ybel Brewing
Style: Irish Ale
ABV: 5.60%

Standardized Output (for both):
{{
  "name": "Sanibel Red Island Ale",
  "brewery": "Point Ybel Brewing",
  "primary_style": "Red Ale",
  "secondary_style": "Amber",
  "abv": 5.6,
  "is_amber": "true",
  "is_ale": "true",
  "is_lager": "false",
  "is_imperial": "false",
  "special_ingredients": "none"
}}

## Example 5: Style breakdown for barrel-aged beers
Input:
Beer Name: Buffalo Trace Bourbon Barrel Aged Amber Ale
Brewery: Wolf Hills Brewing Company
Style: American Amber / Red Ale
ABV: 7.5%

Standardized Output:
{{
  "name": "Buffalo Trace Bourbon Barrel Aged Amber Ale",
  "brewery": "Wolf Hills Brewing",
  "primary_style": "Amber Ale",
  "secondary_style": "Barrel-Aged",
  "abv": 7.5,
  "is_amber": "true",
  "is_ale": "true",
  "is_lager": "false",
  "is_imperial": "false",
  "special_ingredients": "bourbon"
}}


---

Now process this beer record:


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
