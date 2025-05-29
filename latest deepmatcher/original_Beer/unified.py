import pandas as pd
import ollama
from tqdm import tqdm
import re
import json
import os

class OllamaFeatureExtractor:
    def __init__(self, model_name="llama3.1"):
        self.llm_model = model_name

    def extract_standardized_attributes(self, record: dict) -> dict:
        print("dict",record)
        prompt = f"""
You are a data normalization expert. Your job is to clean and standardize structured data records for entity matching:

Instructions:
1. Keep the original keys, but normalize all values.
2. Standardize names by removing branding fluff, promotional terms, and descriptors.
3. Normalize style, category, or genre fields to canonical names.
4. Convert numeric fields (e.g., ABV, weight, pages) to float or integer as appropriate.
5. For missing or invalid values (e.g., "-", "", "unknown"), replace with "unknown".
6. Remove special characters, repeated words, and irrelevant tokens.
7. Remove special characters, repeated words, and irrelevant tokens.
8.Fix inconsistent formatting and punctuation.
9.Standardize naming, capitalization, and abbreviations (e.g., "st" → "Street", "IPA" styles, etc.).
10.Normalize numbers (e.g., "5.6%" → 5.6), and clean phone numbers to digits only.

Critical Important :
-Output only valid JSON. No markdown or explanation.

______________________________________________________________
Few Shot Examples:

### EXAMPLES OF GOOD STANDARDIZATION:

## Example 1:
Input 1:
Beer Name: Red Amber Ale
Brewery: Example Brewing
Style: American Amber / Red Ale
ABV: 5.5%

Standardized Output:
{{
  "name": "Red Amber Ale",
  "brewery": "Example Brewing",
  "primary_style": "American Amber",
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

Standardized Output :
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


-------------------------

Record:
{json.dumps(record, indent=2)}

Return only valid JSON with standardized values. Do not include backticks, markdown, or explanations. 

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
            print("record:",content)
            parsed = json.loads(content)
            return parsed
        except json.JSONDecodeError as jde:
            print(f"❌ JSON decode error: {jde}")
            print("⚠️ Content that failed parsing:", repr(content))
            return {}
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return {}

    def split_record(self, row: dict, side: str) -> dict:
        return {col[len(f"{side}_"):]: row[col] for col in row if col.startswith(f"{side}_")}

    def merge_record(self, row: dict, side: str, enriched: dict) -> None:
        for k, v in enriched.items():
            row[f"{side}_{k}"] = v

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

            self.merge_record(row_dict, "left", left_cleaned)
            self.merge_record(row_dict, "right", right_cleaned)

            all_rows.append(row_dict)

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
