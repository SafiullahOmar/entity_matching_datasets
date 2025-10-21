import pandas as pd
import ollama
from tqdm import tqdm
import re
import json
import os
from typing import Dict, Any, Tuple

# ====== DBLP/ACM schema ======
EXPECTED_KEYS = [
    "title",
    "authors",
    "venue",
    "year"
]




class OllamaFeatureExtractor:
    def __init__(self, model_name: str = "gemma3:12b") -> None:
        self.llm_model = model_name

    def normalize_llm_output(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Map variants, coerce types, and ensure all EXPECTED_KEYS exist."""
        key_map = {
            "title": "title",
            "authors": "authors",
            "venue": "venue",
            "year": "year",
         
          
        }

        normalized: Dict[str, Any] = {}
        for key, value in (response or {}).items():
            std_key = key_map.get(key, key)
            normalized[std_key] = value

        # ensure all keys exist
        for k in EXPECTED_KEYS:
            normalized.setdefault(k, "VAL -")

      

        return normalized

    # -------------------- LLM prompt (pair) --------------------
    def _build_pair_prompt(self, left: Dict[str, Any], right: Dict[str, Any]) -> str:
        return f"""
     You are a data normalization expert for bibliographic records (DBLP/ACM style).
Clean and standardize TWO records at once. 
Return ONE valid JSON object with exactly two top-level keys: "left" and "right".
Each side MUST include these fields (all strings): title, authors, venue, year, vldb.

Normalization rules:
- Title: trim whitespace; remove venue/year fragments accidentally appended to title; retain original punctuation/casing where reasonable; do not invent subtitles.
- Authors: preserve order if known; emit a single comma-separated string "First Last, First Last, ..."; normalize spaces; keep diacritics; if unknown → "VAL -".
- Venue: normalize obvious variants:
  * "International Conference on Management of Data", "SIGMOD", "SIGMOD Conference" → "SIGMOD Conference"
  * "ACM SIGMOD Record", "SIGMOD Record" → "ACM SIGMOD Record"
  * "The VLDB Journal — The International Journal on Very Large Data Bases", "VLDB J.", "VLDB Journal" → "VLDB Journal"
  * "Very Large Data Bases", "VLDB", "VLDB Conference" → "VLDB"
  * Keep other venues as-is but cleaned (e.g., "TODS", "PODS", "ICDE", "KDD").
- Year: four digits "YYYY" when present; otherwise "VAL -".
- VLDB flag (vldb):
  * If venue/title indicates VLDB Journal → "VLDB Journal"
  * Else if venue/title indicates VLDB conference/series or “Very Large Data Bases” → "VLDB"
  * Else → "VAL -"
- Do not hallucinate missing information; if unknown → "VAL -".
- Output exactly one JSON object; no code fences, no extra text.



Now process this pair.

Left record input:
{json.dumps(left, ensure_ascii=False, indent=2)}

Right record input:
{json.dumps(right, ensure_ascii=False, indent=2)}

Output JSON — MUST follow exactly:
{{
  "left": {{
    "title": string,
    "authors": string,
    "venue": string,
    "year": string,
  }},
  "right": {{
    "title": string,
    "authors": string,
    "venue": string,
    "year": string,
  }}
}}

STRICT OUTPUT RULES
- Output exactly one JSON object; no code fences, no Markdown, no surrounding text.
- All fields are strings. Use "VAL -" for unknowns.
- Year must be four digits when known.
""".strip()

    def extract_pair_standardized_attributes(
        self, left_record: Dict[str, Any], right_record: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        prompt = self._build_pair_prompt(left_record, right_record)
        try:
            response = ollama.chat(
                model=self.llm_model,
                options={"temperature": 0.0, "num_predict": 768},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an entity matching normalizer. "
                            "Return ONLY one valid JSON object as specified. "
                            "No code fences, no extra text, no explanations."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            )
            content = response["message"]["content"].strip()
            # Strip accidental fences if any
            if content.startswith("```"):
                content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
                content = re.sub(r"```$", "", content).strip()

            parsed = json.loads(content)
            left_out = self.normalize_llm_output(parsed.get("left", {}))
            right_out = self.normalize_llm_output(parsed.get("right", {}))
            print("reponse",left_out,right_out)
            return left_out, right_out

        except json.JSONDecodeError as jde:
            print(f"❌ JSON decode error: {jde}")
            print("⚠️ Content that failed parsing:", content if 'content' in locals() else None)
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

            # final safety net: ensure all expected keys present
            for side in (left_cleaned, right_cleaned):
                for k in EXPECTED_KEYS:
                    side.setdefault(k, "VAL -")

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
