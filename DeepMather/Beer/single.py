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
    "brewery",
    "primary_style",
    "secondary_style",
    "abv",
    "is_amber",
    "is_ale",
    "is_lager",
    "is_imperial",
    "special_ingredients",
]

class OllamaFeatureExtractor:
    def __init__(self, model_name: str = "llama3.1") -> None:
        self.llm_model = model_name

    # -------------------- Helpers --------------------
    def _coerce_types(self, d: Dict[str, Any]) -> Dict[str, Any]:
        # Coerce boolean-like strings for is_* keys
        for k in list(d.keys()):
            v = d[k]
            if k.startswith("is_"):
                if isinstance(v, str):
                    lv = v.strip().lower()
                    if lv in {"true", "yes", "1"}:
                        d[k] = True
                    elif lv in {"false", "no", "0"}:
                        d[k] = False
                elif isinstance(v, (int, float)):
                    d[k] = bool(v)
                elif v is None:
                    d[k] = False
        # Coerce abv to float or "unknown"
        if "abv" in d:
            v = d["abv"]
            if isinstance(v, str):
                s = v.strip().lower()
                if s in {"", "n/a", "na", "none", "unknown", "-"}:
                    d["abv"] = "unknown"
                else:
                    # extract first number
                    m = re.search(r"[-+]?[0-9]*\.?[0-9]+", s)
                    if m:
                        try:
                            d["abv"] = float(m.group(0))
                        except Exception:
                            d["abv"] = "unknown"
                    else:
                        d["abv"] = "unknown"
            elif isinstance(v, (int, float)):
                d["abv"] = float(v)
            else:
                d["abv"] = "unknown"
        return d

    def normalize_llm_output(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all expected keys exist, map variants, and coerce types."""
        key_map = {
            "Beer_Name": "name",
            "Brew_Factory_Name": "brewery",
            "Style": "primary_style",
        }
        normalized: Dict[str, Any] = {}
        for key, value in response.items():
            std_key = key_map.get(key, key)
            normalized[std_key] = value
        for key in EXPECTED_KEYS:
            if key not in normalized:
                if key == "abv":
                    normalized[key] = "unknown"
                elif key.startswith("is_"):
                    normalized[key] = False
                else:
                    normalized[key] = "unknown"
        return self._coerce_types(normalized)

    # -------------------- LLM prompt (pair) --------------------
    def _build_pair_prompt(self, left: Dict[str, Any], right: Dict[str, Any]) -> str:
        return f"""
You are a data normalization expert. Clean and standardize TWO structured beer records at once.
Return a SINGLE valid JSON object with exactly two top-level keys: "left" and "right".
Each side must follow the schema below and use booleans for is_* fields and a float or "unknown" for abv.

General rules:
1. Keep original field names (keys) conceptually, but output in the canonical schema.
2. Standardize names (remove fluff, duplicates, marketing terms) and ensure proper capitalization.
3. Normalize styles to canonical forms (e.g., "American Amber / Red Ale" → primary_style "Red Ale", secondary_style "Amber").
4. Convert numbers: "5.6%" → 5.6.
5. Missing/invalid values → "unknown".
6. Expand abbreviations where useful (Co. → Company) and compress entities to a single canonical form (e.g., Google LLC → Google).
7. Set flags: is_ale True for ales and stouts/porters; is_lager True for lagers/pilsners/helles; is_imperial True for imperial/double/strong variants; is_amber True when explicitly amber/red amber.

Output JSON schema (MUST follow):
{{
  "left": {{
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
  }},
  "right": {{
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
}}

STRICT OUTPUT RULES:
- Return ONLY the JSON object. No backticks, no explanations.
- Ensure booleans are true/false literals (not strings).
- Ensure the top-level object has exactly the keys "left" and "right".

Left record input:
{json.dumps(left, indent=2)}

Right record input:
{json.dumps(right, indent=2)}
"""

    def extract_pair_standardized_attributes(
        self, left_record: Dict[str, Any], right_record: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        prompt = self._build_pair_prompt(left_record, right_record)
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
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
