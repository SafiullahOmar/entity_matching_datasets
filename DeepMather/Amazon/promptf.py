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
Schema lock: Only the three fields per side: title, manufacturer, price. No extra keys.
  - Core‑only title: Set title to the CoreName only. Remove version, edition, license, packaging, media, OS, upgrade/source, seat counts, program names, marketing fluff, SKUs, and legal marks.
  - Independence via deterministic mapping: Compute CoreName independently for each side using the same rules. Do not copy a value from one side to the other. If both map to the same string, that’s acceptable because the mapping is deterministic.
  - Strict symmetry: If any field is missing/unknown on either side, set that same field missing/unknown on both sides:
  - title / manufacturer → "" (empty string)
  - price → "unknown"
Manufacturer canonicalization: Use the canonical brand (e.g., "Microsoft", "Adobe", "Apple", "Encore Software"). Strip Inc., Corp., Ltd. unless needed for disambiguation. If blank on either side → blank on both (Rule 4).
Price: Parse numeric USD as float; otherwise "unknown". If either side is non‑numeric/missing → "unknown" on both.
Valid JSON only. No comments, no trailing commas, correct quotes.

## CoreName Extraction
Produce the shortest, most distinctive family/app/game/work name that identifies the product without variant attributes.

#Keep:
  - Canonical family/app name: e.g., Visio, Access, Windows Vista, Creative Suite, Final Cut Express, iLife, Mozy, PrintMaster, The Print Shop, Instant Immersion Spanish, SpongeBob SquarePants Typing, Sunset Boulevard.
  - Language/topic that defines the product identity (e.g., Instant Immersion Spanish vs Instant Immersion Japanese).
  - Franchise or character names integral to the product (SpongeBob SquarePants).

## Remove (strip entirely)
  - Version: numbers and tokens like 2007, CS3, 3.5, v. 4.1, '06.
  - Edition/variant: Standard, Professional, Educational, Academic, Web Premium, Design Standard, Production Premium, Home Basic, Platinum, HD, Deluxe, Pro, Express only if it’s a pure edition; keep if it is inseparable from the common name (e.g., Final Cut Express is a distinct product line → keep Express).
  - License/program/seat: Upgrade, Full, OEM, Retail, Download, DVD, CD, TLP, CAL, 20 User, Single User.
  - OS/media/packaging: Windows, Mac, Mac DVD, PC, Box, Download, Complete Package.
  - Source/qualifiers: for Terminal Services, for Mac, for PC, Nonprofit, Academic, Older Version, English, MLP.
  - Legal/formatting noise: (R), (TM), Software, AV Production Software, seller/site names, SKUs/ASINs.
## Formatting
  - Title Case (capitalize main words) with ASCII characters; collapse whitespace.

## Remove duplicate words.

If the best CoreName is a two‑part franchise + function, keep both (e.g., SpongeBob SquarePants Typing).
If a listing is a compilation “featuring” a work, use the central work as CoreName (e.g., Sunset Boulevard).
---
## FEW‑SHOT EXAMPLES (nested left/right)
1. Adobe suites — different variants — Label: 0
Left input:
  title: "Adobe Creative Suite CS3 Web Premium Upgrade [Mac]"
  manufacturer: "Adobe"
  price: 499.0
Right input:
  title: "Adobe CS3 Design Standard Upgrade Windows"
  manufacturer: ""
  price: "unknown"

Standardized Output:
{{
"left": {{
"title": "Adobe Creative Suite Web Premium",
"manufacturer": "",
"price": "unknown"
}},
"right": {{
"title": "Adobe Creative Suite Design Standard",
"manufacturer": "",
"price": "unknown"
}}
}}

2. Language learning — Japanese vs Italian — Label: 0
Left input:
  title: "Instant Immersion Japanese Deluxe 2.0"
  manufacturer: "Topics Entertainment"
  price: 39.99
Right input:
  title: "Instant Immersion Italian Audio (Audio Book)"
  manufacturer: ""
  price: "unknown"

Standardized Output:
{{
"left": {{
"title": "Instant Immersion Japanese",
"manufacturer": "",
"price": "unknown"
}},
"right": {{
"title": "Instant Immersion Italian",
"manufacturer": "",
"price": "unknown"
}}
}}

3. Reader Rabbit variants — Label: 0
Left input:
  title: "Reader Rabbit Learn to Read Phonics Pre Kindergarten"
  manufacturer: "The Learning Company"
  price: 9.99
Right input:
  title: "Reader Rabbit Kindergarten Special 2-CD Edition"
  manufacturer: ""
  price: "unknown"

Standardized Output:
{{
"left": {{
"title": "Reader Rabbit Learn to Read Phonics",
"manufacturer": "",
"price": "unknown"
}},
"right": {{
"title": "Reader Rabbit Kindergarten",
"manufacturer": "",
"price": "unknown"
}}
}}

4. Same product name; both have price — Label: 0
Left input:
  title: "Professor Teaches Windows XP"
  manufacturer: ""
  price: 19.99
Right input:
  title: "Individual Software Professor Teaches Windows XP"
  manufacturer: "Individual Software"
  price: 24.99

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

5. Instant Immersion audio languages — Label: 0
Left input:
  title: "Instant Immersion Spanish Audio Deluxe"
  manufacturer: "Topics Entertainment"
  price: 39.95
Right input:
  title: "Topics Entertainment Instant Immersion German Audio"
  manufacturer: ""
  price: "unknown"

Standardized Output:
{{
"left": {{
"title": "Instant Immersion Spanish Audio",
"manufacturer": "",
"price": "unknown"
}},
"right": {{
"title": "Instant Immersion German Audio",
"manufacturer": "",
"price": "unknown"
}}
}}

6. JumpStart grades — Label: 0
Left input:
  title: "Jumpstart Kindergarten"
  manufacturer: "Knowledge Adventure"
  price: 19.99
Right input:
  title: "Jumpstart (R) Advanced 1st Grade"
  manufacturer: ""
  price: "unknown"

Standardized Output:
{{
"left": {{
"title": "JumpStart Kindergarten",
"manufacturer": "",
"price": "unknown"
}},
"right": {{
"title": "JumpStart Advanced 1st Grade",
"manufacturer": "",
"price": "unknown"
}}
}}

7. QuickVerse vs QuickBooks — Label: 0
Left input:
  title: "QuickVerse Bible Premier Suite"
  manufacturer: "Individual Software"
  price: 39.99
Right input:
  title: "QuickBooks (R) Premier 2003"
  manufacturer: ""
  price: "unknown"

Standardized Output:
{{
"left": {{
"title": "Individual Software QuickVerse Bible",
"manufacturer": "",
"price": "unknown"
}},
"right": {{
"title": "Intuit QuickBooks Premier",
"manufacturer": "",
"price": "unknown"
}}
}}

8. OneNote vs Outlook — Label: 0
Left input:
  title: "Microsoft OneNote 2007 Upgrade"
  manufacturer: "Microsoft"
  price: 79.95
Right input:
  title: "Outlook 2007 Educational Microsoft"
  manufacturer: ""
  price: "unknown"

Standardized Output:
{{
"left": {{
"title": "Microsoft OneNote",
"manufacturer": "",
"price": "unknown"
}},
"right": {{
"title": "Microsoft Outlook",
"manufacturer": "",
"price": "unknown"
}}
}}

9. Audio production — different vendors — Label: 0
Left input:
  title: "M-Audio Pro Tools M-Powered 7.3"
  manufacturer: "M-Audio"
  price: 299.99
Right input:
  title: "Steinberg Wavelab Studio 6 Audio Editing Software Music Production Software"
  manufacturer: "Steinberg Media Technologies"
  price: 299.95

Standardized Output:
{{
"left": {{
"title": "M‑Audio Pro Tools M‑Powered",
"manufacturer": "M-Audio",
"price": 299.99
}},
"right": {{
"title": "Steinberg WaveLab Studio",
"manufacturer": "Steinberg Media Technologies",
"price": 299.95
}}
}}

10. Excel vs Word — Label: 0
Left input:
  title: "Microsoft Office Excel 2007"
  manufacturer: "Microsoft"
  price: 229.95
Right input:
  title: "Microsoft (R) Office Word 2004 for Mac Full Version"
  manufacturer: ""
  price: "unknown"

Standardized Output:
{{
"left": {{
"title": "Microsoft Excel",
"manufacturer": "",
"price": "unknown"
}},
"right": {{
"title": "Microsoft Word",
"manufacturer": "",
"price": "unknown"
}}
}}

11. Same game; duplicate brand wording — Label: 1
Left input:
  title: "Alliance Future Combat"
  manufacturer: "Strategy First"
  price: 19.99
Right input:
  title: "Strategy First Inc. Alliance Future Combat"
  manufacturer: ""
  price: "unknown"

Standardized Output:
{{
"left": {{
"title": "Strategy First Alliance Future Combat",
"manufacturer": "",
"price": "unknown"
}},
"right": {{
"title": "Strategy First Alliance Future Combat",
"manufacturer": "",
"price": "unknown"
}}
}}

12. Dreamweaver vs InDesign — Label: 0
Left input:
  title: "Adobe Dreamweaver CS3 Upgrade"
  manufacturer: "Adobe"
  price: 199.0
Right input:
  title: "Adobe InDesign CS3 for Mac Upgrade"
  manufacturer: ""
  price: "unknown"

Standardized Output:
{{
"left": {{
"title": "Adobe Dreamweaver",
"manufacturer": "",
"price": "unknown"
}},
"right": {{
"title": "Adobe InDesign",
"manufacturer": "",
"price": "unknown"
}}
}}

13. LoJack subscriptions — 1‑yr vs 4‑yr — Label: 0
Left input:
  title: "Computrace LoJack for Laptops : 1 Year Subscription"
  manufacturer: "Absolute Software"
  price: 49.99
Right input:
  title: "Absolute Software Computrace LoJack for Laptops 4-Year License"
  manufacturer: ""
  price: "unknown"

Standardized Output:
{{
"left": {{
"title": "Absolute Software Computrace LoJack for Laptops",
"manufacturer": "",
"price": "unknown"
}},
"right": {{
"title": "Absolute Software Computrace LoJack for Laptops",
"manufacturer": "",
"price": "unknown"
}}
}}

14. Preschool title match — Label: 1
Left input:
  title: "Land Before Time : Preschool"
  manufacturer: "Brighter Minds Media Inc."
  price: 9.99
Right input:
  title: "The Land Before Time : Preschool Adventure"
  manufacturer: "Brighter Minds Media Inc."
  price: 12.9

Standardized Output:
{{
"left": {{
"title": "Brighter Minds The Land Before Time Preschool",
"manufacturer": "Brighter Minds Media Inc.",
"price": 9.99
}},
"right": {{
"title": "Brighter Minds The Land Before Time Preschool",
"manufacturer": "Brighter Minds Media Inc.",
"price": 12.9
}}
}}

15. Punch! Pro Home Design — Label: 1
Left input:
  title: "Punch! Professional Home Design"
  manufacturer: "Punch Software"
  price: 89.99
Right input:
  title: "Punch Software Punch ! Professional Home Design Suite for Windows"
  manufacturer: "Punch Software"
  price: 62.99

Standardized Output:
{{
"left": {{
"title": "Punch Professional Home Design",
"manufacturer": "Punch Software",
"price": 89.99
}},
"right": {{
"title": "Punch Professional Home Design",
"manufacturer": "Punch Software",
"price": 62.99
}}
}}

16. Office Small Business vs SBS CALs — Label: 0
Left input:
  title: "Microsoft Office Small Business 2007"
  manufacturer: "Microsoft"
  price: 735.33
Right input:
  title: "Microsoft Windows Small Business Server 2003 License (20 Additional User CALs)"
  manufacturer: ""
  price: "unknown"

Standardized Output:
{{
"left": {{
"title": "Microsoft Office Small Business",
"manufacturer": "",
"price": "unknown"
}},
"right": {{
"title": "Microsoft Windows Small Business Server CALs",
"manufacturer": "",
"price": "unknown"
}}
}}

17. CRM CAL pack vs Adobe upgrade — Label: 0
Left input:
  title: "Microsoft CRM Professional CAL 3.0 Product Upgrade License Pack User CAL"
  manufacturer: "Microsoft"
  price: 9980.0
Right input:
  title: "Adobe Creative Suite 3 Design Premium Product Upgrade Package 1 User Upgrade"
  manufacturer: "Unknown"
  price: "unknown"

Standardized Output:
{{
"left": {{
"title": "Microsoft CRM Professional CAL Pack",
"manufacturer": "",
"price": "unknown"
}},
"right": {{
"title": "Adobe Creative Suite Design Premium Upgrade",
"manufacturer": "",
"price": "unknown"
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
