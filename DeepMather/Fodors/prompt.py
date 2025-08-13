import pandas as pd
import ollama
from tqdm import tqdm
import re
import json
import os
from typing import Dict, Any, Tuple

# Expected output keys for each side
EXPECTED_KEYS = [
    "name",
    "addr",
    "city",
    "phone",
    "category",
    "class",
]

class OllamaFeatureExtractor:
    def __init__(self, model_name: str = "llama3.1") -> None:
        self.llm_model = model_name


    def normalize_llm_output(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all expected keys exist, map variants, and coerce types."""
        key_map = {
            
            "name":"name",
            "addr":"addr",
            "city":"city",
            "phone":"phone",
            "category":"category",
            "class":"class"
        }
        normalized: Dict[str, Any] = {}
        for key, value in response.items():
            std_key = key_map.get(key, key)
            normalized[std_key] = value
        return normalized

    # -------------------- LLM prompt (pair) --------------------
    def _build_pair_prompt(self, left: Dict[str, Any], right: Dict[str, Any]) -> str:
        
        return f"""
You are a data normalization expert. Your job is to clean and standardize structured data records for entity matching:

You are a data normalization expert. Clean and standardize TWO structured restaurant records at once.
Return a SINGLE valid JSON object with exactly two top-level keys: "left" and "right".
Each side must follow the schema below .
---
## Normalization Rule for the title:
## General Cleaning
- Remove surrounding quotes/backticks from values.
- Strip extra whitespace, collapse multiple spaces into one.
- Convert phone numbers to standard format: NNN-NNN-NNNN (U.S. style), remove slashes/spaces.
- Expand abbreviations in address (St. → Street, Ave. → Avenue, Rd. → Road, Blvd. → Boulevard, NE/NW/SE/SW stays).
- Normalize city names to their common full form ("la" → "Los Angeles", "nyc" → "New York City").
- Remove state names or zip codes from address if already represented in the city field.

## Restaurant Name
- Remove location suffixes in parentheses unless part of the official name ("Le Chardonnay (Los Angeles)" → "Le Chardonnay").
- Remove extra quotes/backslashes.
- Keep brand or chain identifiers if part of the official name.
- Preserve distinctive tokens (e.g., "Cafe", "Grill", "Bistro").

## Category & Class
- Category: normalize to a consistent lowercase-with-spaces form (e.g., "french bistro", "american (new)").
- Class: keep as-is if numeric; else normalize to title case for text classes.

## Missing Values
- If a field is missing, use an empty string "" (never null).



Output JSON schema (MUST follow):
{{
  "left": {{
    "name": string ,
    "address": string, 
    "city": string ,
    "phone": string ,
    "category": string,
    "class":string
  }},
  "right": {{
    
    "name": string ,
    "address": string, 
    "city": string ,
    "phone": string ,
    "category": string,
    "class":string
  }}
}}


---
## FEW‑SHOT EXAMPLES ( nested left/right)


Example A: **label 0 — different restaurants**

Left input ⟶  
  "name": "Le Chardonnay (Los Angeles)",  
  "address": "6703 Melrose Ave.",  
  "city": "los angeles",  
  "phone": "213/857 -0034",  
  "category": "californian",  
  "class": "6"  

Right input ⟶  
  "name": "French Bistro",  
  "address": "8284 Melrose Ave.",  
  "city": "los angeles",  
  "phone": "213-655-8880",  
  "category": "french bistro",  
  "class": "12"

label: 0

Standardized Output:
{{
 "left":  {{ "name": "Le Chardonnay", "address": "6703 Melrose Avenue", "city": "Los Angeles", "phone": "213-857-0034", "category": "californian", "class": 6 }},
 "right": {{ "name": "French Bistro", "address": "8284 Melrose Avenue", "city": "Los Angeles", "phone": "213-655-8880", "category": "french bistro", "class": 12 }}
}}

Example B: **label 1 — same restaurant**

Left input ⟶  
  "name": "Yujean Kang's Gourmet Chinese Cuisine",  
  "address": "67 N. Raymond Ave.",  
  "city": "los angeles",  
  "phone": "818/585 -0855",  
  "category": "asian",  
  "class": "22"  

Right input ⟶  
  "name": "Yujean Kang's",  
  "address": "67 N. Raymond Ave.",  
  "city": "pasadena",  
  "phone": "818-585-0855",  
  "category": "chinese",  
  "class": "22" 
label: 1

Standardized Output:
{{
  "left":  {{ "name": "Yujean Kang's Gourmet Chinese Cuisine", "address": "67 North Raymond Avenue", "city": "Pasadena", "phone": "818-585-0855", "category": "asian", "class": 22 }},
  "right": {{ "name": "Yujean Kang's", "address": "67 North Raymond Avenue", "city": "Pasadena", "phone": "818-585-0855", "category": "chinese", "class": 22 }}
}}

Example C:  **label 0 — different restaurants**

Left input ⟶  
  "name": "Bone's",  
  "address": "3130 Piedmont Road",  
  "city": "atlanta",  
  "phone": "404/237 -2663",  
  "category": "american",  
  "class": "76"  

Right input ⟶  
  "name": "Joe's",  
  "address": "1023 Abbot Kinney Blvd.",  
  "city": "venice",  
  "phone": "310-399-5811",  
  "category": "american (new)",  
  "class": "560"
  
label: 0

Standardized Output:
Output  
{{
  "left":  {{ "name": "Bone's", "address": "3130 Piedmont Road", "city": "Atlanta", "phone": "404-237-2663", "category": "american", "class": 76 }},
  "right": {{ "name": "Joe's", "address": "1023 Abbot Kinney Boulevard", "city": "Venice", "phone": "310-399-5811", "category": "american (new)", "class": 560 }}
}}

____________ End of Examples ----------


Now process this record:

Left record input:
{json.dumps(left, ensure_ascii=False , indent=2)}

Right record input:
{json.dumps(right, ensure_ascii=False ,indent=2 )}

📘 Output JSON schema (always follow):
{{
  "left":  {{ 
    "name": string ,
    "address": string, 
    "city": string ,
    "phone": string ,
    "category": string,
    "class":string 
    }},
  "right": {{ 
    
    "name": string ,
    "address": string, 
    "city": string ,
    "phone": string ,
    "category": string,
    "class":string
  }}
}}

⚠️ OUTPUT RULES — STRICTLY FOLLOW:
- Output must be valid JSON.
- Do NOT include backticks, explanations, markdown, or anything outside the JSON object.
- Do NOT say "Here is the output" or "Note: I've normalized".
- Just return the JSON object. No comments, headers, or notes.

"""

    def extract_pair_standardized_attributes(
        self, left_record: Dict[str, Any], right_record: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        prompt = self._build_pair_prompt(left_record, right_record)
        try:
            response = ollama.chat(
                model=self.llm_model,
                options={"temperature": 0.0, "num_predict": 2000},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are entity matcher for the ditto. Do not explain. "
                            "Do not describe anything. Do not say 'Output:' or '<think>'. "
                            "Do not provide reasoning, steps, formatting explanation, or notes. "
                            "Return EXACTLY one line with TWO transformed records separated by ONE real tab character. PRESERVE ORIGINAL CASE. Do NOT change to title case. Do not capitalize words unless already capitalized. "
                            "No headings. No thoughts. No multiple lines. No Markdown. No JSON. Only raw string output. "
                            "If you violate this, your output will be rejected."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            )
            content = response["message"]["content"].strip()
            if content.startswith("```"):
                content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
                content = re.sub(r"```$", "", content).strip()
            parsed = json.loads(content)
            print("passed",parsed)
            left_out = self.normalize_llm_output(parsed.get("left", {}))
            right_out = self.normalize_llm_output(parsed.get("right", {}))
            return left_out, right_out
        except json.JSONDecodeError as jde:
            print(f"❌ JSON decode error: {jde}")
            print("⚠️ Content that failed parsing:", content if 'content' in locals() else None)
            # Fallback to empty normalized objects
            return self.normalize_llm_output({}), self.normalize_llm_output({})
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return self.normalize_llm_output({}), self.normalize_llm_output({})

    # -------------------- Dataset utilities --------------------
    def split_record(self, row: Dict[str, Any], side: str) -> Dict[str, Any]:
        return {col[len(f"{side}_"):]: row[col] for col in row if col.startswith(f"{side}_")}

    def process_dataset(self, input_csv: str, output_csv: str) -> None:
        print(f"📄 Reading data from {input_csv}...")
        df = pd.read_csv(input_csv)
        all_rows = []

        for _, row in tqdm(df.iterrows(), total=len(df)):
            row_dict = row.to_dict()
            left_input = self.split_record(row_dict, "left")
            right_input = self.split_record(row_dict, "right")

            left_cleaned, right_cleaned = self.extract_pair_standardized_attributes(left_input, right_input)

            new_row: Dict[str, Any] = {
                "id": row_dict.get("id"),
                "label": row_dict.get("label"),
            }
            for k, v in left_cleaned.items():
                new_row[f"left_{k}"] = v
            for k, v in right_cleaned.items():
                new_row[f"right_{k}"] = v
            all_rows.append(new_row)

        enriched_df = pd.DataFrame(all_rows)
        print(f"💾 Saving enriched data to {output_csv}")
        enriched_df.to_csv(output_csv, index=False)


def main() -> None:
    extractor = OllamaFeatureExtractor()

    for split in ["train", "valid", "test"]:
        input_file = f"{split}.csv"
        output_file = f"{split}_enriched.csv"
        if os.path.exists(input_file):
            print(f"\n🟡 Processing {split}...")
            extractor.process_dataset(input_file, output_file)
        else:
            print(f"⚠️  {input_file} not found, skipping...")


if __name__ == "__main__":
    main()
