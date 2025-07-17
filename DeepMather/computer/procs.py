import pandas as pd
import ollama
from tqdm import tqdm
import re
import json
import os
# from examples import Beer_Fewshot_exampels
# from output_strucutres import Beer_output

# Define the required schema
EXPECTED_KEYS = [
    "title"
]

class OllamaFeatureExtractor:
    def __init__(self, model_name="llama3.1"):
        self.llm_model = model_name

    def normalize_llm_output(self, response: dict) -> dict:
        """Ensure all expected keys are present with consistent types and names."""
        key_map = {
            "title": "title",
        }

        normalized = {}

        # Map and rename keys
        for key, value in response.items():
            std_key = key_map.get(key, key)
            normalized[std_key] = value

        # Fill in missing keys
        # for key in EXPECTED_KEYS:
        #     if key not in normalized:
        #         if key == "abv":
        #             normalized[key] = "unknown"
        #         elif key.startswith("is_"):
        #             normalized[key] = False
        #         else:
        #             normalized[key] = "unknown"

        return normalized

    def extract_standardized_attributes(self, record: dict) -> dict:
    
        print("passed dict",record)
        prompt = f"""
You are a record normalizer optimizing product titles for entity matching using DeepMatcher. You will receive a pair of computer product titles and a `label` indicating whether they refer to the same product (`label = 1`) or not (`label = 0`).

Your goal is to return **cleaned, normalized versions** of each title (`left_title` and `right_title`) as free-text strings, in the same style and format as the input, but cleaned for matching purposes.

---

## 🧹 General Cleaning and Normalization Rules:

- Identify and preserve key attributes like brand, product type, storage/capacity, model number, generation, and variant details.
- Preserve exact numeric values (e.g., 2TB, 7200RPM, 3.5in, E5607, 8GB, 2666MHz, 10K, 128GB).
- **Never remove or alter alphanumeric model numbers or part codes** (e.g., ST31000524NS, 658071-B21, MZ-N5E1T0BW, WD20EFRX, CT51264BF160B).
- If a model number appears multiple times, retain one clean instance.
- Remove redundant vendor or website suffixes (e.g., “macofalltrades”, “CDW.com”, “SCAN UK”, “TWEAKERS”, “OcUK”, “Superwarehouse”).
- Remove unnecessary tokens like “null”, “price”, “new”, “wholesale”, “@en”, “LNxxxxx”.
- Deduplicate repeated segments or trailing IDs.
- Translate foreign or mixed-language text to English.
- Never guess or add fields not already present in the input.
- Keep titles in free-text format, not structured.
- Return exactly **one cleaned line per title**, no markdown or extra commentary.

---

## 🧠 Match-Sensitive Rules:

- If `label = 1` (match):
  - Align the terminology, phrasing, and formatting of both records.
  - Unify units (e.g., "3.5 inch" → "3.5in", "7200 RPM" → "7200RPM").
  - Use consistent ordering and style across both sides.

- If `label = 0` (non-match):
  - Normalize lightly.
  - Preserve differences in brand, product type, model no, phrasing, etc.
  - Do **not** over-align structure or style across both sides.
  - Even in label = 0 cases, always retain product codes or model numbers, such as:
  - Part numbers (359461-007, 540-5629)
  - Internal model codes (MZ-N5E1T0BW, ST31000524NS, CMK64GX4M4A2666C16)
  - These identifiers are critical for DeepMatcher, even if records are not aligned.

---

## ✅ FEW-SHOT EXAMPLES

### Example 1 (label = 1)

**Input**  
left_title: "Corsair Vengeance LPX Black 64GB (4x16GB) DDR4 PC4-21300 2666MHz Quad Channel Kit"  
right_title: "Corsair Vengeance LPX CMK64GX4M4A2666C16"  
label: 1

**Standardized Output:**  
{{
  "left_title": "Corsair Vengeance LPX 64GB (4x16GB) DDR4 2666MHz Quad Channel Kit CMK64GX4M4A2666C16",
  "right_title": "Corsair Vengeance LPX 64GB DDR4 2666MHz Kit CMK64GX4M4A2666C16"
}}

---

### Example 2 (label = 0)

**Input**  
left_title: "388504-B21 HP Storageworks Internal, Null Price 388504-B21 Internal Wholesale 388504-B21"  
right_title: "Null, 449363-B21 HP SC40Ge Host Bus Adapter Adapter Wholesale 449363-B21 Price 449363-B21"  
label: 0

**Standardized Output:**  
{{
  "left_title": "HP Storageworks Internal 388504-B21",
  "right_title": "HP SC40Ge Host Bus Adapter 449363-B21"
}}

---

### Example 3 (label = 0 – Intel SSD vs. Samsung SSD)

**Input**  
left_title: "Intel Solid-State Drive 540S Series - solid state drive 240 GB SATA 6Gb Intel 6Gb SSDSCKKW240H6X1 Solid State Drives (SSDs) CDW.com"  
right_title: "Samsung 850 EVO Series M.2 1TB SATA 6Gbps Solid State Drive (MZ-N5E1T0BW) ▷ Samsung … | OcUK"  
label: 0

**Standardized Output:**  
{{
  "left_title": "Intel 540S Series 240GB SATA 6Gbps SSD SSDSCKKW240H6X1",
  "right_title": "Samsung 850 EVO Series 1TB M.2 SATA 6Gbps SSD MZ-N5E1T0BW"
}}

---

### Example 4 (label = 0 – Two different Z270 motherboards)

**Input**  
left_title: "Gigabyte GA-Z270N-WIFI Intel Z270 (Socket 1151) DDR4 Mini-ITX Motherboard ▷ Gigabyte Mini-IT… | OcUK"  
right_title: "ASUS PRIME Z270M-PLUS microATX LGA1151 Intel Z270 DDR4 SATA 3"  
label: 0

**Standardized Output:**  
{{
  "left_title": "Gigabyte GA-Z270N-WIFI Intel Z270 DDR4 Mini-ITX Motherboard",
  "right_title": "ASUS PRIME Z270M-PLUS Intel Z270 DDR4 microATX Motherboard"
}}

### Example 5  (label = 0 — part numbers must be retained)

**Input** 
left_title: "XTA-3510-73-GB-10K (540-5629) Sun 73-GB, Null 73-GB Wholesale XTA-3510-73-GB-10K Price XTA-3510-73GB-10K"
right_title: "359461-007 HP 300-GB 10K FC-AL HDD Null"
label: 0

**Standardized Output:**  
{{
  "left_title": "Sun 73GB 10K XTA-3510-73GB-10K (540-5629)",
  "right_title": "HP 300GB 10K FC-AL HDD 359461-007"
}}

### Example 6  (label = 0 — model numbers inside long vendor strings)

**Input** 
left_title: "ST31000524NS Seagate 1-TB 7.2K 3.5 3G SATA, Null New ST31000524NS SATA 10 Pack Wholesale Price ST31000524NS-10Pack"
right_title: "Null, 658071-B21 HP G8 G9 500-GB 6G 7.2K 3.5 SATA SC New 658071-B21 5 Pack Wholesale Price 658071-B21-5Pack"
label: 0

**Standardized Output:**  
{{
  "left_title": "Seagate ST31000524NS 1TB 7.2K 3.5in SATA",
  "right_title": "HP G8 G9 500GB 7.2K 3.5in SATA 658071-B21"
}}


---

Now process this record:


Record:
{json.dumps(record, indent=2)}


📘 Output JSON schema format (always follow this):

{{
  "left_title": string,
  "right_title": string
  
}}

Return only valid JSON with standardized values. Do not include backticks, markdown, or explanations. Remember to ALWAYS split complex styles into primary_style and secondary_style components.

"""
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[ {
                        "role": "system",
                        "content": (
                            "You are entity matcher for the deepmatcher. Do not explain. "
                            "Do not describe anything. Do not say 'Output:' or '<think>'. "
                            "Do not provide reasoning, steps, formatting explanation, or notes. "
                            "If you violate this, your output will be rejected."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }]
            )
            content = response["message"]["content"].strip()
            print("output is",content)
            if content.startswith("```"):
                content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
                content = re.sub(r"```$", "", content).strip()

            
            parsed = json.loads(content)
            return self.normalize_llm_output(parsed)

        except json.JSONDecodeError as jde:
            print(f"❌ JSON decode error: {jde}")
            print("⚠️ Content that failed parsing:", repr(content))
            return record
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return record

    def split_record(self, row: dict, side: str) -> dict:
        """Extract left or right side sub-record"""
        return {col[len(f"{side}_"):]: row[col] for col in row if col.startswith(f"{side}_")}

    def process_dataset(self, input_csv, output_csv):
        print(f"📄 Reading data from {input_csv}...")
        df = pd.read_csv(input_csv)
        all_rows = []

        for _, row in tqdm(df.iterrows(), total=len(df)):
            row_dict = row.to_dict()
            record_pair = {
                    "left_title": row_dict.get("left_title", ""),
                    "right_title": row_dict.get("right_title", ""),
                    "label": row_dict.get("label", 0)
                }
            cleaned_pair = self.extract_standardized_attributes(record_pair)
            
             
            new_row = {
                "id": row_dict.get("id"),
                "label": row_dict.get("label"),
                "left_title": cleaned_pair.get("left_title", record_pair["left_title"]),
                "right_title": cleaned_pair.get("right_title", record_pair["right_title"])
            }
            all_rows.append(new_row)

            # left_input = self.split_record(row_dict, "left")
            # right_input = self.split_record(row_dict, "right")

            # left_cleaned = self.extract_standardized_attributes(left_input)
            # right_cleaned = self.extract_standardized_attributes(right_input)

            # # Construct the new row with normalized fields only
            # new_row = {
            #     "id": row_dict.get("id"),
            #     "label": row_dict.get("label")
            # }

            # for k, v in left_cleaned.items():
            #     new_row[f"left_{k}"] = v
            # for k, v in right_cleaned.items():
            #     new_row[f"right_{k}"] = v

            # all_rows.append(new_row)

        enriched_df = pd.DataFrame(all_rows)
        print(f"💾 Saving enriched data to {output_csv}")
        enriched_df.to_csv(output_csv, index=False)

def main():
    extractor = OllamaFeatureExtractor()

    for split in ['train', 'valid', 'test']:
        input_file = f"{split}.csv"
        output_file = f"{split}_enriched.csv"
        if os.path.exists(input_file):
            print(f"\n🟡 Processing {split}...")
            extractor.process_dataset(input_file, output_file)
        else:
            print(f"⚠️  {input_file} not found, skipping...")

if __name__ == "__main__":
    main()
