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
    "Song_Name",
    "Artist_Name",
    "Album_Name",
    "Genre",
    "Price",
    "CopyRight",
    "Time",
    "Released",
]

class OllamaFeatureExtractor:
    def __init__(self, model_name="llama3.1"):
        self.llm_model = model_name

    def normalize_llm_output(self, response: dict) -> dict:
        """Ensure all expected keys are present with consistent types and names."""
        key_map = {
            "Song_Name": "Song_Name",
            "Artist_Name": "Artist_Name",
            "Album_Name": "Album_Name",
            "Genre": "Genre",
            "Price": "Price",
            "CopyRight": "CopyRight",
            "Time": "Time",
            "Released": "Released"
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
    

        prompt = f"""
You are a data normalization expert. Your job is to clean and standardize structured data records to improve entity matching performance in machine learning systems.
---
Your Task:
Clean the input record according to the following universal rules. Then return the normalized version using the exact same schema and field names (keys).
---

## CORE OBJECTIVES:

1. Preserve original tokens where possible to support token-level alignment.
2. Expand acronyms *non-destructively* by **appending normalized forms in parentheses** (e.g., "cs3 (creative suite 3)").
3. Normalize OS, versions, editions consistently but avoid over-rewriting.
4. Standardize price formatting.

## NORMALIZATION RULES:
##Song_Name:
Retain featured artists, remix titles, version info, and punctuation such as parentheses or brackets exactly as written.
e.g., Illusion (feat. Echosmith), Titanium (Spanish Version), Still Down [Explicit].
Preserve original casing — do not apply title casing.
Do not add or remove descriptors like [Live], (Remix), etc., unless they are redundant and already represented in other fields.

##Artist_Name:
If Artist_Name is missing but clearly appears in the raw input (e.g., embedded in Song_Name or after title), extract it accurately.
e.g., "Illusion (feat. Echosmith) Zedd" → Artist_Name: Zedd
Keep artist names exactly as written, including groupings like David Guetta & Sia.
Do not modify order or formatting.
If still not present, use VAL -.

##Album_Name:
Retain full album name if found in the input or embedded in the Song_Name value.
e.g., "Titanium David Guetta Listen (Deluxe Version)" → Album_Name: Listen (Deluxe Version)
Strip only platform-specific extras like [iTunes Live], [Bonus Track], etc. if already reflected elsewhere.

##Genre:
If missing but genre-like terms (e.g., Dance & Electronic, Country, Hip-Hop) appear in the raw input, extract and assign them.
e.g., "...True Colors Dance, Music, Electronic..." → Genre: Dance, Music, Electronic
Preserve the delimiter structure (e.g., commas, slashes).
Normalize spacing but never fabricate genres.

##Price:
Extract prices from raw strings like $1.29, 1.29, or USD 0.99, and format as: USD X.XX.
e.g., "Still Down... $1.29..." → Price: USD 1.29
Always prefix with USD, even if missing in the original.
Use VAL - if price cannot be parsed.

##CopyRight:
If a copyright string is embedded in the input (e.g., 2015 Interscope Records, (C) 2014 Atlantic Recording Corporation), extract it verbatim.
Preserve symbols like (C) or ©, and keep full label chain.
If not found, use VAL -.

##Time:
Extract time values like 3:29, 4:02, 05:30, and normalize as MM:SS.
If seconds or minutes are single-digit, pad to two digits: 3:5 → 03:05
Do not infer durations — only assign if explicitly present.

##Released:
Extract recognizable date expressions like May 18, 2015 or 26-Aug-11 and convert to YYYY-MM-DD.
August 26, 2011 → 2011-08-26
26-Aug-11 → 2011-08-26
If multiple date formats exist, choose the most explicit/complete.
If unavailable, use VAL -.

##Missing Field Inference (General Rule):
When a field (e.g., Artist_Name, Album_Name, Genre, Price, Time, Released) is missing but clearly present elsewhere in the input (often inside Song_Name or after it), extract it and populate the correct field.
Never fabricate values — infer only when the information is unambiguous.

-----

### EXAMPLES OF GOOD STANDARDIZATION:

## Example 1:
Input:

Song_Name: Illusion ( feat . Echosmith ) Zedd True Colors Dance , Music , Electronic 2015 Interscope Records 6:30
Artist_Name: ""
Album_Name: ""
Genre: ""
price: $ 1.29
CopyRight: "2015 Interscope Records"
Time: ""
Released: "18-May-15"

Standardized Output:
{{
  "Song_Name": "Illusion (feat. Echosmith)",
  "Artist_Name": "Zedd",
  "Album_Name": "True Colors",
  "Genre": "Dance, Music, Electronic",
  "Price": 1.29,
  "CopyRight": "2015 Interscope Records",
  "Time": "06:30",
  "Released": "2015-05-18"
}}

## Example 2:
Input:

Song_Name: Transmission [ feat . X Ambassadors ] Dance & Electronic $ 1.29 ( C ) 2015 Interscope Records 4:02
Artist_Name: "Zedd"
Album_Name: "True Colors"
Genre: ""
Price: "$ 1.29"
CopyRight: "2015 Interscope Records"
Time: ""
Released: "May 18 , 2015"

Standardized Output:
{{
  "Song_Name": "Transmission (feat. X Ambassadors)",
  "Artist_Name": "Zedd",
  "Album_Name": "True Colors",
  "Genre": "Dance & Electronic",
  "Price": 1.29,
  "CopyRight": "(C) 2015 Interscope Records",
  "Time": "04:02",
  "Released": "2015-05-18"
}}

## Example 3:
Input:
Song_Name: I 'm a Machine ( feat . Crystal Nicole and Tyrese Gibson ) Dance , Music , Rock , House , Electronic 26-Aug-11
Artist_Name: "David Guetta"
Album_Name: "Nothing But the Beat"
Genre: ""
Price: "$ 1.29"
CopyRight: "2011 What A Music Ltd , Licence exclusive Parlophone Music France"
Time: "3:34"
Released: "May 18 , 2015"

Standardized Output:
{{
  "Song_Name": "I'm a Machine (feat. Crystal Nicole and Tyrese Gibson)",
  "Artist_Name": "David Guetta",
  "Album_Name": "Nothing But the Beat",
  "Genre": "Dance, Music, Rock, House, Electronic;",
  "Price": "1.29",
  "CopyRight": "2011 What A Music Ltd, Licence exclusive Parlophone Music France",
  "Time": "03:34",
  "Released": "2011-08-26"
}}

---

Now process this record:


Record:
{json.dumps(record, indent=2)}


📘 Output JSON schema format (always follow this):

{{
  "Song_Name": string,
  "Artist_Name": string,
  "Album_Name": string,
  "Genre": string,
  "Price": string,
  "CopyRight": string,
  "Time": string,
  "Released": string,
  
}}

Return only valid JSON with standardized values. Do not include backticks, markdown, or explanations. Remember to ALWAYS split complex styles into primary_style and secondary_style components.

"""
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response["message"]["content"].strip()

            if content.startswith("```"):
                content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
                content = re.sub(r"```$", "", content).strip()

            print("record:", content)
            parsed = json.loads(content)
            return self.normalize_llm_output(parsed)

        except json.JSONDecodeError as jde:
            print(f"❌ JSON decode error: {jde}")
            print("⚠️ Content that failed parsing:", repr(content))
            return self.normalize_llm_output({})
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return self.normalize_llm_output({})

    def split_record(self, row: dict, side: str) -> dict:
        """Extract left or right side sub-record"""
        return {col[len(f"{side}_"):]: row[col] for col in row if col.startswith(f"{side}_")}

    def process_dataset(self, input_csv, output_csv):
        print(f"📄 Reading data from {input_csv}...")
        df = pd.read_csv(input_csv)
        all_rows = []

        for _, row in tqdm(df.iterrows(), total=len(df)):
            row_dict = row.to_dict()

            left_input = self.split_record(row_dict, "left")
            right_input = self.split_record(row_dict, "right")

            left_cleaned = self.extract_standardized_attributes(left_input)
            right_cleaned = self.extract_standardized_attributes(right_input)

            # Construct the new row with normalized fields only
            new_row = {
                "id": row_dict.get("id"),
                "label": row_dict.get("label")
            }

            for k, v in left_cleaned.items():
                new_row[f"left_{k}"] = v
            for k, v in right_cleaned.items():
                new_row[f"right_{k}"] = v

            all_rows.append(new_row)

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
