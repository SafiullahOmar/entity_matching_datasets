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
You are a record normalizer optimizing product titles for entity matching using DeepMatcher. You will receive a pair of camera product titles and a `label` indicating whether they refer to the same product (`label = 1`) or not (`label = 0`).

Your goal is to return **cleaned, normalized versions** of each title (`left_title` and `right_title`) as free-text strings, in the same style and format as the input, but cleaned for matching purposes.

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
---

## ✅ FEW-SHOT EXAMPLES

### Example 1

**Input**  
left_title: "Canon EF 70-300mm f/4-5.6 IS II USM Telephoto Zoom Lens"@en "by Canon | Amazon.com"  
right_title: "Canon 70-300mm IS II USM Lens EF Mount"@en "Canon Lens for EOS SLR"  
label: 1

**Standardized Output:**  
{{
  "left_title": "Canon EF 70-300mm f/4-5.6 IS II USM Telephoto Zoom Lens",
  "right_title": "Canon EF 70-300mm f/4-5.6 IS II USM Telephoto Zoom Lens"
}}

---

### Example 2

**Input**  
left_title: "GoPro HERO9 Black Bundle with Extra Battery + Tripod Mount"@en-US "@MediaMarkt NL"  
right_title: "GoPro HERO9 Black Edition Camera Kit with Battery and Mount"@en"  
label: 0

**Standardized Output:**  
{{
  "left_title": "GoPro HERO9 Black Camera Kit with Extra Battery and Tripod Mount",
  "right_title": "GoPro HERO9 Black Camera Kit with Extra Battery and Tripod Mount"
}}

---


### Example 3 

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

### Example 4 

**Input**  
left_title: "Canon EOS 90D DSLR Camera with 18-135mm IS USM Lens"@en 
right_title: "Canon EOS Rebel T7 with 18-55mm Lens Kit for Beginners"@en  
label: 0

**Standardized Output:**  
{{
  "left_title": "Canon EOS 90D DSLR Camera with 18-135mm IS USM Lens",
  "right_title": "Canon EOS Rebel T7 DSLR Camera with 18-55mm Lens Kit"
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
