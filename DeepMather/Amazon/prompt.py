import pandas as pd
import ollama
from tqdm import tqdm
import re
import json
from textwrap import dedent
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
        return dedent(f"""You are a product-normalization expert. Clean and standardize TWO Amazon software/product records for entity matching with DeepMatcher.

Return a SINGLE valid JSON object with exactly two top-level keys: "left" and "right".
Each side must follow this schema:
• "title" (string)
• "manufacturer" (string)
• "price" (float OR "unknown")

────────────────────────────────────────
GLOBAL RULES (read carefully)
────────────────────────────────────────
• Normalize EACH SIDE INDEPENDENTLY. Never copy, infer, or harmonize from the other side.
• Keep discriminative tokens (versions, editions/licences, platform/OS, language). Do NOT paraphrase names.
• Casing: Title Case. Collapse duplicate spaces and consecutive duplicate words.
• Manufacturer: Canonical minimal form (drop “Inc.”, “Corp.”, “Ltd.”, “Software”, “Systems”), but keep the core brand (e.g., “Adobe Systems Inc.” → “Adobe”).
• If manufacturer is blank, leave "" (don’t guess from title or the other side).
• Price: Parse numeric → float with two decimals. If missing/invalid, use "unknown". Never invent or copy across sides.

────────────────────────────────────────
TOKEN PRESERVATION (SKU-LEVEL)
────────────────────────────────────────
KEEP ALL of the following when present:
• Versions: CS2/CS3, X3, XI, 11.0, 7.3, 2007, etc. (keep token as written; don’t change CS→“Creative Suite”.)
• Editions/Licences: Professional, Pro, Home, Standard, Enterprise, Academic/Education/Student, OEM, Retail, Upgrade/Update, Single-User/1-User/3-User/5-User, Volume, Site, Server, CAL, ESD, Boxed, Download, Subscription, License/Licence, Host Only, Expansion Pack, Add-on, Plug-in, Trial, Beta.
• Platform/OS & Language (when SKU-differentiating): Windows/Win, macOS/Mac, Linux; Spanish, English, French, Multilingual.
• Product-family words: Photoshop, WordPerfect, Pro Tools, Finale, TaxCut, Visual Studio, QuickBooks, etc.

────────────────────────────────────────
NOISE REMOVAL
────────────────────────────────────────
• Remove SKU/catalog codes like “19600061dm”, “SF9006”, “11052”.
• Remove pure packaging media ONLY: (DVD), (CD-ROM), [CD], [DVD].
• Remove generic trailers (case-insensitive) ONLY if they don’t remove SKU info:
  “Production Software”, “Photo Editing Software for Windows”, “Complete Package/Product”, “Standard English PC”, “Qualification”.
  ⚠️ Do NOT remove “Expansion Pack”, “Server”, “CAL”, or platform/edition/language tokens.

────────────────────────────────────────
ABBREVIATION / SPELLING
────────────────────────────────────────
• Win → Windows; S/W → Software; AV → Anti-Virus; Svr → Server; Upg → Upgrade; Propack → Pro Pack.
• Retain “Host Only”. Do NOT expand CS tokens.

────────────────────────────────────────
DUPLICATE-WORD COLLAPSE
────────────────────────────────────────
Collapse consecutive duplicate words inside the title (“Home Home Design” → “Home Design”).

────────────────────────────────────────
MISSING VALUES
────────────────────────────────────────
Empty title/manufacturer → ""; price per rules above.

────────────────────────────────────────
FEW-SHOT EXAMPLES (use as guidance)
────────────────────────────────────────

1) Different vendors — label 0
Left input ⟶
  "title": "microsoft visio standard 2007 version upgrade",
  "manufacturer": "microsoft",
  "price": 129.95
Right input ⟶
  "title": "adobe cs3 design standard upgrade",
  "manufacturer": "",
  "price": 413.99
Output
{{
  "left":  {{"title": "Microsoft Visio Standard 2007 Version Upgrade", "manufacturer": "Microsoft", "price": 129.95}},
  "right": {{"title": "Adobe CS3 Design Standard Upgrade",             "manufacturer": "",          "price": 413.99}}
}}

2) Same product family & version (Mac note on left) — label 1
Left input ⟶
  "title": "motu digital performer 5 digital audio software competitive upgrade ( mac only )",
  "manufacturer": "motu",
  "price": 395.0
Right input ⟶
  "title": "motu digital performer dp5 software music production software",
  "manufacturer": "",
  "price": 319.95
Output
{{
  "left":  {{"title": "Motu Digital Performer 5 Competitive Upgrade (Mac Only)", "manufacturer": "Motu", "price": 395.00}},
  "right": {{"title": "Motu Digital Performer 5",                                "manufacturer": "",     "price": 319.95}}
}}

3) Illustrator CS3 Academic for Mac (1-User vs Academic) — label 1
Left input ⟶
  "title": "illustrator cs3 13 mac ed 1u",
  "manufacturer": "adobe-education-box",
  "price": 199.0
Right input ⟶
  "title": "adobe illustrator cs3 for mac academic",
  "manufacturer": "adobe-education-box",
  "price": 199.99
Output
{{
  "left":  {{"title": "Adobe Illustrator CS3 13 For Mac Academic 1-User", "manufacturer": "Adobe", "price": 199.00}},
  "right": {{"title": "Adobe Illustrator CS3 For Mac Academic",           "manufacturer": "Adobe", "price": 199.99}}
}}

4) Brand mismatch within “Mavis Beacon” line — label 0
Left input ⟶
  "title": "mavis beacon typing 17 ( win/mac )",
  "manufacturer": "encore software",
  "price": 19.99
Right input ⟶
  "title": "broderbund mavis beacon teaches typing standard17",
  "manufacturer": "",
  "price": 22.99
Output
{{
  "left":  {{"title": "Mavis Beacon Typing 17 (Windows/Mac)", "manufacturer": "Encore", "price": 19.99}},
  "right": {{"title": "Mavis Beacon Teaches Typing Standard 17", "manufacturer": "", "price": 22.99}}
}}

5) Different Adobe apps (Premiere Pro vs Soundbooth) — label 0
Left input ⟶
  "title": "adobe premiere pro cs3",
  "manufacturer": "adobe",
  "price": 799.0
Right input ⟶
  "title": "adobe soundbooth cs3 academic",
  "manufacturer": "",
  "price": 95.99
Output
{{
  "left":  {{"title": "Adobe Premiere Pro CS3",       "manufacturer": "Adobe", "price": 799.00}},
  "right": {{"title": "Adobe Soundbooth CS3 Academic", "manufacturer": "",     "price": 95.99}}
}}

6) Training media vs software SKU — label 0
Left input ⟶
  "title": "adobe photoshop cs2 advanced techniques by julieanne kost",
  "manufacturer": "software cinema",
  "price": 
Right input ⟶
  "title": "adobe photoshop cs3 ( v10 .0 ) mac adobe 13102488",
  "manufacturer": "",
  "price": 537.65
Output
{{
  "left":  {{"title": "Adobe Photoshop CS2 Advanced Techniques By Julieanne Kost", "manufacturer": "Software Cinema", "price": "unknown"}},
  "right": {{"title": "Adobe Photoshop CS3 v10.0 For Mac",                         "manufacturer": "",                "price": 537.65}}
}}

7) Enterprise licensing (quantities differ) — label 0
Left input ⟶
  "title": "microsoft crm professional cal 3.0 product upgrade license pack user cal",
  "manufacturer": "microsoft software",
  "price": 9980.0
Right input ⟶
  "title": "c8a-00066 microsoft dynamics crm professional v. 3.0 product upgrade license 20",
  "manufacturer": "",
  "price": 9676.92
Output
{{
  "left":  {{"title": "Microsoft CRM Professional CAL 3.0 Product Upgrade License Pack User CAL", "manufacturer": "Microsoft", "price": 9980.00}},
  "right": {{"title": "Microsoft Dynamics CRM Professional 3.0 Product Upgrade License 20",       "manufacturer": "",          "price": 9676.92}}
}}

8) Academic vs retail; Mac noted — label 0
Left input ⟶
  "title": "adobe creative suite cs3 web standard",
  "manufacturer": "adobe",
  "price": 999.0
Right input ⟶
  "title": "adobe creative suite 3 web standard complete package academic cd mac",
  "manufacturer": "",
  "price": 369.0
Output
{{
  "left":  {{"title": "Adobe Creative Suite CS3 Web Standard",              "manufacturer": "Adobe", "price": 999.00}},
  "right": {{"title": "Adobe Creative Suite 3 Web Standard Academic For Mac", "manufacturer": "",    "price": 369.00}}
}}

9) Edition mismatch (Gold vs Platinum) — label 0
Left input ⟶
  "title": "printmaster gold v 17.0",
  "manufacturer": "encore software",
  "price": 19.99
Right input ⟶
  "title": "print master platinum v17",
  "manufacturer": "",
  "price": 29.9
Output
{{
  "left":  {{"title": "PrintMaster Gold 17.0",    "manufacturer": "Encore", "price": 19.99}},
  "right": {{"title": "Print Master Platinum 17", "manufacturer": "",       "price": 29.90}}
}}

10) Same upgrade SKU wording — label 1
Left input ⟶
  "title": "microsoft word 2007 version upgrade",
  "manufacturer": "microsoft",
  "price": 109.95
Right input ⟶
  "title": "microsoft word 2007 upgrade ( pc )",
  "manufacturer": "",
  "price": 109.95
Output
{{
  "left":  {{"title": "Microsoft Word 2007 Version Upgrade", "manufacturer": "Microsoft", "price": 109.95}},
  "right": {{"title": "Microsoft Word 2007 Upgrade (PC)",    "manufacturer": "",          "price": 109.95}}
}}

11) Same product; remove SKU code, keep OS — label 1
Left input ⟶
  "title": "hoyle : classic collection 2006",
  "manufacturer": "encore",
  "price": 19.99
Right input ⟶
  "title": "encore software 11052 hoyle : classic collection 2006 win 98 me 2000 xp",
  "manufacturer": "",
  "price": 18.97
Output
{{
  "left":  {{"title": "Hoyle: Classic Collection 2006",                           "manufacturer": "Encore", "price": 19.99}},
  "right": {{"title": "Hoyle: Classic Collection 2006 (Windows 98/ME/2000/XP)",   "manufacturer": "",       "price": 18.97}}
}}

12) Different Visual Studio editions (MSDN vs Pro Upgrade) — label 0
Left input ⟶
  "title": "microsoft visual studio team edition for software developers 2005 with msdn premium",
  "manufacturer": "microsoft",
  "price": 5479.0
Right input ⟶
  "title": "visual studio pro 2005 upgrade ( pc ) microsoft",
  "manufacturer": "",
  "price": 549.0
Output
{{
  "left":  {{"title": "Microsoft Visual Studio Team Edition For Software Developers 2005 With MSDN Premium", "manufacturer": "Microsoft", "price": 5479.00}},
  "right": {{"title": "Microsoft Visual Studio Pro 2005 Upgrade (PC)",                                       "manufacturer": "",          "price": 549.00}}
}}

────────────────────────────────────────
OUTPUT (JSON ONLY, EXACTLY THIS SHAPE)
────────────────────────────────────────
{{
  "left":  {{"title": string, "manufacturer": string, "price": float or "unknown"}},
  "right": {{"title": string, "manufacturer": string, "price": float or "unknown"}}
}}

Do not output anything except this single JSON object.

Now process this record:

Left record input:
{json.dumps(left, ensure_ascii=False, indent=2)}

Right record input:
{json.dumps(right, ensure_ascii=False, indent=2)}
""")

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
