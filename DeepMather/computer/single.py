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
    def __init__(self, model_name: str = "mixtral:latest") -> None:
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

You are a data normalization expert. Clean and standardize TWO structured computer records at once.
Return a SINGLE valid JSON object with exactly two top-level keys: "left" and "right".
Each side must follow the schema below .
---
## Normalization Rule for the title:

## Extract key attributes:
Brand, Product Type (SSD, HDD, USB Flash Drive, RAM Kit, CPU, GPU, Motherboard, PSU, Case, Cooler, Server Part, etc.), Series/Family, Model/Part No., Capacity, Form Factor, Interface/Bus, Speed/Perf, Kit/Count, Edition/Color.
Normalize spelling/casing; remove vendor/site noise (e.g., "CDW.com", "PCPartPicker", "SCAN UK", "Tweakers", pipes |, parentheses with store info), and tokens like “Null”, “Price”, “Wholesale”, “US/UK”.
Translate foreign/mixed segments to English when relevant.
Preserve distinguishing specs (e.g., 7200RPM, DDR4‑2400 CL15, SATA 6Gb/s, NVMe PCIe 3.0 x4, M.2 2280) so different items stay different.
Keep part numbers exact (dashes/case intact).

## Canonical Structure Order:
[Brand] [Product Type] [Series] [Model/Part] [Capacity] [Form Factor] [Interface/Bus] [Key Specs] [Kit/Count] [Edition/Notes]
Examples:
Samsung 840 EVO SSD MZ-7TE250BW 250GB 2.5" SATA 6Gb/s
Seagate Barracuda ST2000DM006 2TB 3.5" SATA 6Gb/s 7200RPM HDD
Western Digital Red WD60EFRX 6TB 3.5" SATA 6Gb/s HDD


##Normalize Brand Names:
Apple, AMD, Intel, Samsung, Seagate, Western Digital (WD), Kingston, Corsair, Crucial, G.Skill, Transcend, SanDisk, Adata, Kioxia, Toshiba, HGST, OCZ, ASUS, MSI, Gigabyte, ASRock, NZXT, Fractal Design, be quiet!, Cooler Master, EVGA, Seasonic, HP/HPE, Dell, Lenovo.
## Normalize and Expand Abbreviations:
SATA III / SATA 3 → SATA 6Gb/s
USB 3.1 Gen1 → USB 3.0; USB 3.1 Gen2 → USB 3.1 Gen2 (keep if explicitly Gen2)

## De‑clutter and Translate:
Remove vendor noise ("@Tweakers", domain names), pricing/status words ("Price", "Wholesale", "New", "Discontinued"), marketing fluff ("Hot Buy", "Black Friday"), and locale suffixes. Translate or drop foreign metadata; do not invent specs.



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
## FEW‑SHOT EXAMPLES ( nested left/right)


Example A: Different HP server parts (should not match)

Left input:
left_title: "293461-B21 PL BL40p 1.5X Null"

Right input:
right_title: "359557-B21 BL20P G2 Xeon 2.8GHz (2P), Null Price 359557-B21 (2P) Wholesale 359557-B21"

label: 0

Standardized Output:
{{
"left":  {{"title": "HP 293461-B21 BL40p 1.5X Server Part"}},
"right": {{"title": "HP 359557-B21 BL20p G2 Xeon 2.8GHz (2P) Server Part"}}
}}

Example B: Same HP memory kit (should match)

Left input:
left_title: "Null 461826-B21 HP 2GB PC5300 (2x1GB) Kit"

Right input:
right_title: "461826-B21 HP 2GB PC5300 (2x1GB) Kit, Null Price 461826-B21 New 461826-B21 Kit Wholesale"

label: 1

Standardized Output:
{{
"left":  {{"title": "HP 461826-B21 2GB (2x1GB) PC2-5300 DDR2 Server Memory Kit"}},
"right": {{"title": "HP 461826-B21 2GB (2x1GB) PC2-5300 DDR2 Server Memory Kit"}}
}}

Example C: USB flash drives, different products

Left input:
left_title: "Transcend 8GB USB Flash Drive 2.0  Memory & Storage | Unique Photo"

Right input:
right_title: "Rugged Corsair Survivor Stealth USB 3.0 64GB Flash Drive V2 V2 LN65056 - CMFSS3B-64GB | SCAN UK"

label: 0

Standardized Output:
{{
"left":  {{"title": "Transcend USB 2.0 8GB Flash Drive"}},
"right": {{"title": "Corsair Survivor Stealth USB 3.0 64GB Flash Drive CMFSS3B-64GB"}}
}}

Example D: NAS HDD vs desktop HDD (different)

Left input:
left_title: "Western Digital Red 6 TB Internal HDD  Western HDD - WD60EFRX Desktop Hard Drives CDW.com"

Right input:
right_title: "Seagate Barracuda ST2000DM006 - hard drive 2 TB SATA 6Gb/s  Seagate 6Gb/s Internal Desktop Hard Drives CDW.com"

label: 0

Standardized Output:
{{
"left": {{"title": "Western Digital Red WD60EFRX 6TB 3.5\" SATA 6Gb/s HDD"}},
"right": {{"title": "Seagate Barracuda ST2000DM006 2TB 3.5\" SATA 6Gb/s 7200RPM HDD"}}
}}

Example E: Same SSD model (should match)

Left input:
left_title: "DISCONTINUED Samsung 840 EVO 250GB 2.5-Inch SATA III Internal SSD (MZ-7TE250BW)-US Data Storage - Page 6 | Laptops Outlet Direct-US"

Right input:
right_title: "Samsung - 840 EVO 250GB 2.5 Solid State Drive Drive (MZ-7TE250BW) PCPartPicker United Kingdom"

label: 1

Standardized Output:
{{
"left":  {{"title": "Samsung 840 EVO SSD MZ-7TE250BW 250GB 2.5\" SATA 6Gb/s"}},
"right": {{"title": "Samsung 840 EVO SSD MZ-7TE250BW 250GB 2.5\" SATA 6Gb/s"}}
}}

Example F: RAM kit, same product (should match)

Left input:
left_title: "8GB 1333MHZ DDR3 DIMM KIT OF 2 2 | Tradineur.com"

Right input:
right_title: "Kingston ValueRAM KVR13N9S8HK2/8 - Prijzen  Tweakers"

label: 1

Standardized Output:
{{
"left":  {{"title": "Kingston ValueRAM KVR13N9S8HK2/8 8GB (2x4GB) DDR3-1333 CL9 240-Pin DIMM Kit"}},
"right": {{"title": "Kingston ValueRAM KVR13N9S8HK2/8 8GB (2x4GB) DDR3-1333 CL9 240-Pin DIMM Kit"}}
}}

Example G: Different Apple iMac configurations (different)

Left input:
left_title: "Apple 27 2.7GHz Intel Quad-Core i5 iMac Desktop Computer Accessories For Apple Computer - Z0M6TBD1 Abt"

Right input:
right_title: "Apple 21.5 2.7GHz Intel Quad-Core i5 iMac Desktop Computer Computer - Z0M5TBD3 Abt"

label: 0

Standardized Output:
{{
"left":  {{"title": "Apple iMac Z0M6TBD1 27\" 2.7GHz Quad‑Core i5"}},
"right": {{"title": "Apple iMac Z0M5TBD3 21.5\" 2.7GHz Quad‑Core i5"}}
}}

____________ End of Examples ----------


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
