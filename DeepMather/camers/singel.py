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
]

class OllamaFeatureExtractor:
    def __init__(self, model_name: str = "llama3.1") -> None:
        self.llm_model = model_name


    def normalize_llm_output(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all expected keys exist, map variants, and coerce types."""
        key_map = {
            "title": "title",
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

You are a data normalization expert. Clean and standardize TWO structured camera records at once.
Return a SINGLE valid JSON object with exactly two top-level keys: "left" and "right".
Each side must follow the schema below .
---

### Normalization Rule for the title:
- Extract key attributes such as:
- Brand, Camera Type (e.g., DSLR, Mirrorless, GoPro), Accessory Type (e.g., Bag, Mount, Tripod, Lens), Model Number, Specifications (e.g., zoom range, resolution, edition)
- Normalize spelling and remove redundant vendor/website suffixes (e.g., “TWEAKERS@NL”, “SCAN UK”, “Foto Erhardt”)
- Translate foreign/mixed language segments to English when relevant
- Remove promotional or unrelated words (e.g., “Black Friday 2017”, “Come As You Arts”)
- Preserve key specs like focal length (e.g., 24-70mm), aperture (e.g., f/2.8), edition (e.g., “Silver Edition”), sensor type, and battery model
- Ensure accessory types like battery grip, mounts, cases, kits, remotes are clearly stated
- Maintain consistent format: e.g., GoPro Hero3+ Silver Edition, Canon EF 24-70mm f/4L IS USM Zoom Lens
### Canonical Structure Order:
-Always structure titles in this order:
-[Brand] [Product Type] [Model Number/Name] [Key Specs] [Edition] [Accessory Type]
-Examples:
Canon DSLR EOS 80D Camera with 18-135mm IS USM Lens Kit
GoPro Hero3+ Silver Edition Mount Strap
Sigma 8-16mm f/4.5-5.6 DC HSM Ultra-Wide Zoom Lens

### Normalize Brand Names:
-Standardize known brands: Canon, Nikon, Sigma, GoPro, Panasonic, Sony, etc.

###Normalize and Expand Abbreviations:
“EF-S” → “EF-S Mount”
“USM” → “Ultrasonic Motor”
“SLR” → “Single Lens Reflex”

### Normalize Accessory Types:
Standardize to: Battery, Charger, Strap, Tripod, Grip, Lens, Kit, Bag, Remote, Mount

### De-clutter and Translate:
Remove vendor noise (e.g., “@Tweakers”, “| Fumfie.com”)
Translate or drop foreign metadata
Remove promotional terms (e.g., “Black Friday Deal”, “Hot Buy”)


Output JSON schema (MUST follow):
{{
  "left": {{
    "title": string,
  }},
  "right": {{
    "title": string
  }}
}}


---


Now process this record:

Left record input:
{json.dumps(left.get("title", ""), ensure_ascii=False)}

Right record input:
{json.dumps(right.get("title", ""), ensure_ascii=False)}

📘 Output JSON schema (always follow):
{{
  "left":  {{"title": string}},
  "right": {{"title": string}}
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
        print(f"�� Saving enriched data to {output_csv}")
        enriched_df.to_csv(output_csv, index=False)


def main() -> None:
    extractor = OllamaFeatureExtractor()

    for split in ["train","valid","test"]:
        input_file = f"{split}.csv"
        output_file = f"{split}_enriched.csv"
        if os.path.exists(input_file):
            print(f"\n🟡 Processing {split}...")
            extractor.process_dataset(input_file, output_file)
        else:
            print(f"⚠️  {input_file} not found, skipping...")


if __name__ == "__main__":
    main()
