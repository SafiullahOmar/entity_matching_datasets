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

Return **one** valid JSON object with **exactly two top‑level keys**: "left" and "right".
Normalize each side **independently**. **Never** rewrite fields to make left and right equal. Preserve differences (version, edition, OS, license, packaging) **inside the title string**.

Each side must include **exactly** these fields and types:
{{
  "left":  {{"title": string, "manufacturer": string, "price": float or "unknown"}},
  "right": {{"title": string, "manufacturer": string, "price": float or "unknown"}}
}}

**Symmetry rule**: If a field is missing/unknown on either side, set the same field missing on both sides — title/manufacturer → "" and price → "unknown".

---
### Normalization goals
1. **Preserve key tokens** for alignment; remove vendor noise, SKUs, trailing site names.
2. **Title** canonical order ⇒ [Brand] [Product/Family] [Version] [Edition] [OS/Media/Packaging] [Key Terms].
3. **Manufacturer**: canonical brand (e.g., "microsoft software" → "Microsoft").
4. **Acronyms, non‑destructive**: append long form once → "CS3 (Creative Suite 3)", "CAL (Client Access License)", "TLP (Transactional License Program)".
5. **Keep discriminators**: Educational vs Professional, Upgrade vs Full, Legal vs Professional, X vs IX, Vista Business vs SBS CALs, Media (CD/DVD/Download), OS list.
6. **Price**: float (USD, no symbol) or "unknown".

---
## FEW‑SHOT EXAMPLES (nested left/right)

# 1. Professional vs Educational — keep editions different
Left input:
  title: "Sibelius 4 Professional Edition"
  manufacturer: "Sibelius Software Ltd."
  price: 599.99
Right input:
  title: "Sibelius - Sibelius 4 Educational Discount Music Production Software"
  manufacturer: "Sibelius Software Ltd."
  price: 249.95

**Standardized Output:**
{{
  "left": {{
    "title": "Sibelius 4 Professional Edition",
    "manufacturer": "Sibelius Software Ltd.",
    "price": 599.99
  }},
  "right": {{
    "title": "Sibelius 4 Educational Edition (Educational Discount)",
    "manufacturer": "Sibelius Software Ltd.",
    "price": 249.95
  }}
}}

---
# 2. Different versions & vertical — do not unify
Left input:
  title: "Nuance Dragon NaturallySpeaking Pro Solution 9.0"
  manufacturer: "Nuance Communications Inc."
  price: 399.54
Right input:
  title: "Nuance Dragon NaturallySpeaking V8 Legal Solutions from Professional V6 and Up (Govt) OLPA590A-SF2-8.0"
  manufacturer: "Nuance Communications Inc."
  price: 314.00

**Standardized Output:**
{{
  "left": {{
    "title": "Nuance Dragon NaturallySpeaking 9.0 Professional",
    "manufacturer": "Nuance Communications Inc.",
    "price": 399.54
  }},
  "right": {{
    "title": "Nuance Dragon NaturallySpeaking 8 Legal (Government Upgrade from Professional 6+)",
    "manufacturer": "Nuance Communications Inc.",
    "price": 314.0
  }}
}}

---
# 3. Version X vs IX — keep version tokens
Left input:
  title: "Corel Painter X for PC/Mac"
  manufacturer: "Corel"
  price: 155.22
Right input:
  title: "Corel Painter IX Win/Mac"
  manufacturer: "Corel Corporation"
  price: 89.99

**Standardized Output:**
{{
  "left": {{
    "title": "Corel Painter X for Windows/Mac",
    "manufacturer": "Corel",
    "price": 155.22
  }},
  "right": {{
    "title": "Corel Painter IX for Windows/Mac",
    "manufacturer": "Corel Corporation",
    "price": 89.99
  }}
}}

---
# 4. Upgrade vs Media of prior version — keep both signals
Left input:
  title: "CorelDRAW Graphics Suite X3 (Upgrade)"
  manufacturer: "Corel"
  price: 179.00
Right input:
  title: "CorelDRAW Graphics Suite V.12 Media, 1 User CD Win Multi-Lingual"
  manufacturer: "Unknown"
  price: 22.59

**Standardized Output:**
{{
  "left": {{
    "title": "CorelDRAW Graphics Suite X3 Upgrade",
    "manufacturer": "Corel",
    "price": 179.0
  }},
  "right": {{
    "title": "CorelDRAW Graphics Suite 12 Media, 1-User CD, Windows, Multilingual",
    "manufacturer": "Unknown",
    "price": 22.59
  }}
}}

---
# 5. Exact title text, different manufacturer strings — keep canonical manufacturer
Left input:
  title: "Punch! Super Home Suite"
  manufacturer: "Punch!"
  price: 49.99
Right input:
  title: "Punch! Super Home Suite"
  manufacturer: "Punch Software"
  price: 45.99

**Standardized Output:**
{{
  "left": {{
    "title": "Punch! Super Home Suite",
    "manufacturer": "Punch Software",
    "price": 49.99
  }},
  "right": {{
    "title": "Punch! Super Home Suite",
    "manufacturer": "Punch Software",
    "price": 45.99
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
