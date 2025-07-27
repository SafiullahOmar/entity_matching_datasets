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
You are a data normalization expert. Your job is to clean and standardize structured data records for entity matching:

You are a data normalization expert. Clean and standardize TWO structured camera records at once.
Return a SINGLE valid JSON object with exactly two top-level keys: "left" and "right".
Each side must follow the schema below .
---

### Normalization goals
1. **Preserve key tokens** for alignment; remove vendor noise, SKUs, trailing site names.
2. **Title**: canonical order → [Brand] [Product Type] [Model/Version] [Edition/Platform] [Key Terms].
3. **Manufacturer**: normalize to canonical brand (e.g., "microsoft software" → "Microsoft"); expand legal suffixes compactly ("Inc." → "Inc.").
4. **Acronym expansion, non‑destructive**: append the first time they appear in parentheses. Examples:
   - "CS3" → "Creative Suite 3 (CS3)"
   - "TLP" → "Transactional License Program (TLP)"
   - "CAL" → "Client Access License (CAL)"
   - "SBS" → "Small Business Server (SBS)"
5. **Platform markers**: keep "Mac", "Windows", "DVD", "Upgrade", "Academic", "OEM", "Builder" when present.
6. **Price**: parse to a **float** in USD (no currency symbol). If missing/unparseable → "unknown".
7. **label usage**: If **label = 1** (same product), unify **title** and **manufacturer** to the same canonical phrasing on both sides. Price may differ by seller; keep each side’s parsed price. If **label = 0**, normalize each independently.



Output JSON schema (MUST follow):

{{
  "left": {{
    "title": string,
    "manufacturer": string,
    "price": string,
  }},
  "right": {{
    "title": string,
    "manufacturer": string,
    "price": string,
  }}
}}

## FEW‑SHOT EXAMPLES (Beer‑style; nested left/right)

### Example 1 — Different products (label = 0)
Left input:
  title: "adobe creative suite cs3 production premium upsell"
  manufacturer: "adobe"
  price: 1199.0
Right input:
  title: "19600061dm adobe creative suite 3 production premium media tlp download mac world"
  manufacturer: ""
  price: 20.97
label: 0

**Standardized Output:**
{{
  "left": {{
    "title": "Adobe Creative Suite 3 (CS3) Production Premium Upsell",
    "manufacturer": "Adobe",
    "price": 1199.0
  }},
  "right": {{
    "title": "Adobe Creative Suite 3 (CS3) Production Premium, Media TLP Download for Mac",
    "manufacturer": "Adobe",
    "price": 20.97
  }}
}}

---
### Example 2 — Same product; unify title/manufacturer, keep prices (label = 1)
Left input:
  title: "the sims 2 : open for business expansion pack"
  manufacturer: "aspyr media"
  price: 34.99
Right input:
  title: "sims 2 open for business"
  manufacturer: ""
  price: 34.99
label: 1

**Standardized Output:**
{{
  "left": {{
    "title": "The Sims 2: Open for Business Expansion Pack",
    "manufacturer": "Aspyr Media",
    "price": 34.99
  }},
  "right": {{
    "title": "The Sims 2: Open for Business Expansion Pack",
    "manufacturer": "Aspyr Media",
    "price": 34.99
  }}
}}

---
### Example 3 — Same product with different prices; unify textual fields (label = 1)
Left input:
  title: "checkmark multiledger"
  manufacturer: "checkmark software"
  price: 399.0
Right input:
  title: "channel sources distribution co mlw6 .0 checkmark multiledger for pc/mac"
  manufacturer: "channel sources distribution co"
  price: 294.58
label: 1

**Standardized Output:**
{{
  "left": {{
    "title": "CheckMark MultiLedger 6.0 for PC/Mac",
    "manufacturer": "CheckMark Software",
    "price": 399.0
  }},
  "right": {{
    "title": "CheckMark MultiLedger 6.0 for PC/Mac",
    "manufacturer": "CheckMark Software",
    "price": 294.58
  }}
}}

---
### Example 4 — CAL / license packs (label = 0)
Left input:
  title: "microsoft windows small business server cal 2003 license pack 20 client addpack device"
  manufacturer: "microsoft software"
  price: ""
Right input:
  title: "microsoft ( r ) windows vista ( tm ) business"
  manufacturer: ""
  price: 199.99
label: 0

**Standardized Output:**
{{
  "left": {{
    "title": "Microsoft Windows Small Business Server 2003 Client Access License Pack, 20 Device CALs (Client Access License)",
    "manufacturer": "Microsoft",
    "price": "unknown"
  }},
  "right": {{
    "title": "Microsoft Windows Vista Business",
    "manufacturer": "Microsoft",
    "price": 199.99
  }}
}}

---
### Example 5 — QuickBooks vs QuickBooks (different editions; label = 0)
Left input:
  title: "quickbooks premier non-profit edition 2005"
  manufacturer: "intuit inc."
  price: 499.95
Right input:
  title: "quickbooks ( r ) premier : accountant edition 2007"
  manufacturer: ""
  price: 399.99
label: 0

**Standardized Output:**
{{
  "left": {{
    "title": "QuickBooks Premier 2005 Nonprofit Edition",
    "manufacturer": "Intuit",
    "price": 499.95
  }},
  "right": {{
    "title": "QuickBooks Premier 2007 Accountant Edition",
    "manufacturer": "Intuit",
    "price": 399.99
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
- Return values exactly as strings; format Price as `USD X.XX`, Time as `MM:SS`, Released as `YYYY-MM-DD`, and use `VAL -` when unknown.
- Do not invent fields or omit required ones.

### SINGLE‑OBJECT EMISSION GUARD — CRITICAL
- Return **exactly one** JSON object.
- **No code fences**, markdown, comments, or explanations.
- Output **compact single‑line JSON** if possible.

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

