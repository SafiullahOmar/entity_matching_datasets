import pandas as pd
import ollama
from tqdm import tqdm
import re
import json
import os
from typing import Dict, Any, Tuple
 
# Expected output keys for each side
EXPECTED_KEYS = [
    "title",
    "manufacturer",
    "price"
    
]

class OllamaFeatureExtractor:
    def __init__(self, model_name: str = "llama3.1") -> None:
        self.llm_model = model_name


    def normalize_llm_output(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all expected keys exist, map variants, and coerce types."""
        key_map = {
            "title": "title",
            "manufacturer":"manufacturer",
            "price": "price",
        }
        normalized: Dict[str, Any] = {}
        for key, value in response.items():
            std_key = key_map.get(key, key)
            normalized[std_key] = value
        return normalized

    # -------------------- LLM prompt (pair) --------------------
    def _build_pair_prompt(self, left: Dict[str, Any], right: Dict[str, Any]) -> str:
        return f"""
You are a product normalization expert. Clean and standardize **two** Amazon software/product records at once for entity matching with DeepMatcher.
Return a SINGLE valid JSON object with exactly two top-level keys: "left" and "right".
Each side must follow the schema below .

## Rules:
## Normalization Rules

1. **Preserve Specificity**  
Keep version numbers (e.g., 2007, CS3), editions (e.g., Standard, Professional), packaging (DVD, CD), and licensing (Upgrade, OEM, TLP) when present.

2. **Fix Casing and Whitespace**  
Apply Title Case: capitalize major words in the title. Collapse extra spaces.

3. **Manufacturer Canonicalization**  
Standardize manufacturer names (e.g., "Microsoft Corporation" → "Microsoft"). Strip suffixes like Inc., Ltd., Corp. unless needed to disambiguate.

4. **Missing Values (One-Sided Only)**  
If one side is missing a field, leave it as-is. Do NOT blank both sides. DeepMatcher learns from asymmetry.

   Use these values only when field is truly missing:
   - Title / Manufacturer → `""` (empty string)
   - Price → `"unknown"`

5. **Price Formatting**  
If price is a numeric value, return as float. If missing, non-numeric, or noisy → return `"unknown"`.


## Remove duplicate words.

If the best CoreName is a two‑part franchise + function, keep both (e.g., SpongeBob SquarePants Typing).
If a listing is a compilation “featuring” a work, use the central work as CoreName (e.g., Sunset Boulevard).
---
## FEW‑SHOT EXAMPLES 

1. Adobe suites — different variants — Label: 0
Left input:
   "title": "Adobe Creative Suite CS3 Web Premium Upgrade [Mac]",
  "manufacturer": "Adobe",
  "price": 499.0
Right input:
  "title": "Adobe CS3 Design Standard Upgrade Windows",
  "manufacturer": "",
  "price": "unknown"

{{
  "left": {{
    "title": "Adobe Creative Suite CS3 Web Premium",
    "manufacturer": "",
    "price": "unknown"
  }},
  "right": {{
    "title": "Adobe Creative Suite CS3 Design Standard",
    "manufacturer": "",
    "price": "unknown"
  }}
}}

2. Language learning — Japanese vs Italian — Label: 0
Left input:
  "title": "Instant Immersion Japanese Deluxe 2.0",
  "manufacturer": "Topics Entertainment",
  "price": 39.99
Right input:
  "title": "Instant Immersion Italian Audio (Audio Book)",
  "manufacturer": "",
  "price": "unknown"

Standardized Output:
{{
  "left": {{
    "title": "Instant Immersion Japanese Deluxe 2.0",
    "manufacturer": "",
    "price": "unknown"
  }},
  "right": {{
    "title": "Instant Immersion Italian Audio",
    "manufacturer": "",
    "price": "unknown"
  }}
}}

3. Reader Rabbit variants — Label: 0
Left input:
  "title": "Reader Rabbit Learn to Read Phonics Pre Kindergarten",
  "manufacturer": "The Learning Company",
  "price": 9.99
Right input:
   "title": "Reader Rabbit Kindergarten Special 2-CD Edition",
  "manufacturer": "",
  "price": "unknown"

Standardized Output:
{{
  "left": {{
    "title": "Reader Rabbit Learn to Read Phonics",
    "manufacturer": "",
    "price": "unknown"
  }},
  "right": {{
    "title": "Reader Rabbit Kindergarten Special Edition",
    "manufacturer": "",
    "price": "unknown"
  }}
}}

4. Same product name; both have price — Label: 0
Left input:
  "title": "Professor Teaches Windows XP",
  "manufacturer": "",
  "price": 19.99
Right input:
    "title": "Individual Software Professor Teaches Windows XP",
  "manufacturer": "Individual Software",
  "price": 24.99

Standardized Output:
{{
  "left": {{
    "title": "Professor Teaches Windows XP",
    "manufacturer": "",
    "price": 19.99
  }},
  "right": {{
    "title": "Professor Teaches Windows XP",
    "manufacturer": "",
    "price": 24.99
  }}
}}




____________ End of Examples ----------

Now process this record:

Left record input:
{json.dumps(left, ensure_ascii=False, indent=2)}

Right record input:
{json.dumps(right, ensure_ascii=False, indent=2)}

📘 Output JSON schema (always follow exactly):
{{
  "left":  {{"title": string, "manufacturer": string, "price": float or "unknown"}},
  "right": {{"title": string, "manufacturer": string, "price": float or "unknown"}}
}}

⚠️ OUTPUT RULES — STRICTLY FOLLOW
- Return **exactly one** JSON object.
- No code fences, markdown, comments, logs, or repeated JSON.
- Do not add or omit keys. Use only float or "unknown" for price.

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
