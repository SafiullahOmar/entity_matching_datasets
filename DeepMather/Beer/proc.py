import pandas as pd
import numpy as np
import ollama
from tqdm import tqdm
import re
import json
import ast
import os

class OllamaFeatureExtractor:
    def __init__(self, model_name="llama3.1"):
        self.llm_model = model_name

    def extract_product_attributes(self, beer_name, brew_factory, style, abv):
        prompt = f"""
You are an expert in beer product classification and entity resolution.

Given information about a beer, standardize and normalize the values to help with entity matching. Your task is to create consistent representations that would match between different listings of the same product.

### STYLE STANDARDIZATION IS CRITICAL
- ALWAYS split complex styles into primary_style and secondary_style
- For example: "American Amber / Red Ale" → primary_style: "Red Ale", secondary_style: "Amber"
- For example: "Imperial Russian Stout" → primary_style: "Stout", secondary_style: "Imperial"
- For example: "Belgian-Style Tripel" → primary_style: "Tripel", secondary_style: "Belgian"
- For example: "Barrel Aged Double IPA" → primary_style: "IPA", secondary_style: "Double Barrel-Aged"

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

Beer Name: {beer_name}  
Brewery: {brew_factory}  
Style: {style}  
ABV: {abv}  

Return only valid JSON with standardized values. Do not include backticks, markdown, or explanations. Remember to ALWAYS split complex styles into primary_style and secondary_style components.
"""
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )

            content = response["message"]["content"].strip()
            print("🔍 response content:", content)

            # Remove Markdown code formatting if present
            if content.startswith("```"):
                content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
                content = re.sub(r"```$", "", content)
                content = content.strip()

            # Attempt to parse as JSON
            parsed = json.loads(content)
            print("✅ Parsed:", parsed)
            return parsed
        except json.JSONDecodeError as jde:
            print(f"❌ JSON decode error: {jde}")
            print("⚠️ Content that failed parsing:", repr(content))
            return {}

        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return {}

    def process_dataset(self, input_csv, output_csv):
        print(f"Reading data from {input_csv}...")
        df = pd.read_csv(input_csv)

        all_rows = []

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            left = self.extract_product_attributes(
                row['left_Beer_Name'],
                row['left_Brew_Factory_Name'],
                row['left_Style'],
                row['left_ABV']
            )
            right = self.extract_product_attributes(
                row['right_Beer_Name'],
                row['right_Brew_Factory_Name'],
                row['right_Style'],
                row['right_ABV']
            )

            combined = {
                **row.to_dict(),
                **{f"left_{k}": v for k, v in left.items()},
                **{f"right_{k}": v for k, v in right.items()},
            }

            all_rows.append(combined)

        enriched_df = pd.DataFrame(all_rows)
        print(f"Saving enriched data to {output_csv}")
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
