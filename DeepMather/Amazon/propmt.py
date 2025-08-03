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
You are a product‑normalization expert.  Clean and standardize **two**
Amazon software/product records at once for entity matching with DeepMatcher.

Return a SINGLE valid JSON object with exactly two top‑level keys: "left"
and "right".  
Each side must follow this schema:  
  • "title"  (string)  
  • "manufacturer"  (string)  
  • "price"  (float **or** "unknown")

────────────────────────────────────────
NORMALIZATION RULES
────────────────────────────────────────

## Noise Removal

Delete SKU/catalog codes (mixed letters + digits such as 19600061dm, SF9006).

Strip brackets/parentheses that name only platform or media ([Mac], (Win 95/98/ME), (DVD)).

## Trim generic trailer phrases (case-insensitive, stop at first match):
Full Version of .* Software · .* Production Software · Sound Editing S/?W ·
Photo Editing Software for Windows · Complete (Package|Product) ·
Standard English PC · Scientific Brain Training · Music Production ·
Qualification · Contact Management .* · No Limit Texas Hold 'Em · similar.

## Abbreviation & Spelling Expansion
CS1/2/3 → Creative Suite 1/2/3 · CAL → Client Access License · Svr → Server ·
Upg → Upgrade · OEM → OEM · AV → Anti-Virus · S/W → Software · Win → Windows ·
Propack → Pro Pack · Host Only stays as is · feel free to add obvious forms.

## Preserve Specificity
Keep version tokens (CS3, XI, X3, 11.0, 7.3, 2007) exactly.
Keep edition/licence words (Professional, Home, Standard, Upgrade, 3-User, Host Only, Boxed).
Never delete or alter these discriminative tokens.

## Casing & Whitespace
Convert to Title Case; collapse consecutive spaces.

## Manufacturer Canonicalisation
Shortest unambiguous form: “Microsoft Corporation” → Microsoft; delete Inc., Ltd., Corp., Software unless needed.

## Missing Values
Empty title / manufacturer → ""; price handled above.

## Price Formatting
Valid number → float with two decimals; else "unknown".

## Duplicate-Word Collapse
Remove consecutive duplicate words inside the title (“Home Home Design” → “Home Design”).
---


## FEW‑SHOT EXAMPLES 
1. **Adobe suites — label 0**

Left input ⟶  
  "title": "Adobe Photoshop CS3",
  "manufacturer": "Adobe",
  "price": Adobe Systems Inc

Right input ⟶  
  "title": "Adobe Photoshop CS2 Mac OS X v10.2.8 to 10.3",
  "manufacturer": "Adobe Systems Inc",
  "price": 788.63

Output  
{{  
  "left":  {{ "title": "Adobe Photoshop CS3", "manufacturer": "Adobe", "price": 649.00 }},
  "right": {{ "title": "Adobe Photoshop CS2", "manufacturer": "Adobe", "price": 788.63 }}

}}

2. **Example2 — label 1**

Left input ⟶  
  "title": "Microsoft Digital Image Suite Plus" ,
  "manufacturer": "Microsoft",
  "price": 129.95

Right input ⟶  
  "title": "Microsoft Digital Image Suite Plus Full Version of Photo Editing Software for Windows",
  "manufacturer": "",
  "price":  89.95

Output  
{{  
  "left":  {{ "title": "Microsoft Digital Image Suite Plus", "manufacturer": "Microsoft", "price": 129.95 }},
  "right": {{ "title": "Microsoft Digital Image Suite Plus", "manufacturer": "", "price": 89.95 }}

}}


3. ** Example 3 (label 0) 


Left input ⟶
"title":  "Adobe Creative Suite CS3 Production Premium Upsell",
"manufacturer": "Adobe",
"price": 1199.0
Right input ⟶
"title":  "Adobe Creative Suite 3 Production Premium Media TLP Download Mac World",
"manufacturer": <NULL>,
"price": 20.97

Output
{{
  "left": {{
    "title": "Adobe Creative Suite 3 Production Premium Upsell",
    "manufacturer": "Adobe",
    "price": 1199.00
  }},
  "right": {{
    "title": "Adobe Creative Suite 3 Production Premium",
    "manufacturer": "",
    "price": 20.97
  }}
}}


4.** Example 4 (label 0)



Left input ⟶
"title": "M-Audio Pro Tools M-Powered 7.3",
"manufacturer": "M-Audio",
"price":  299.99
Right input ⟶
"title":  "Make Finale 2007 Software Music Production Software",
"manufacturer": <NULL>,
"price": 429.95

Output
{{
  "left": {{
    "title": "M-Audio Pro Tools M-Powered 7.3",
    "manufacturer": "M-Audio",
    "price": 299.99
  }},
  "right": {{
    "title": "Make Finale 2007",
    "manufacturer": "",
    "price": 429.95
  }}
}}


5.** Example 5 (label 1)

Left input ⟶
"title":  "Checkmark Multiledger" ,
"manufacturer": " Checkmark Software",
"price":  399.0
Right input ⟶
"title":  "Channel Sources Distribution Co Mlw6.0 Checkmark Multiledger for PC/Mac"  ,
"manufacturer":" Channel Sources Distribution Co",
"price": 294.58

Output
{{
  "left": {{
    "title": "Checkmark Multiledger",
    "manufacturer": "Checkmark",
    "price": 399.00
  }},
  "right": {{
    "title": "Checkmark Multiledger",
    "manufacturer": "Channel Sources",
    "price": 294.58
  }}
}}



6.** Example 6 (label 1)


Left input ⟶
"title":  "The Sims 2: Open for Business Expansion Pack" ,
"manufacturer": "Aspyr Media",
"price":  34.99
Right input ⟶
"title":  "The Sims 2 Open for Business" ,
"manufacturer": <NULL> ,
"price": 34.99

Output
{{
   "left": {{
    "title": "The Sims 2 Open For Business Expansion Pack",
    "manufacturer": "Aspyr",
    "price": 34.99
  }},
  "right": {{
    "title": "The Sims 2 Open For Business",
    "manufacturer": "",
    "price": 34.99
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
