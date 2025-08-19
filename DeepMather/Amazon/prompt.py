import pandas as pd
import ollama
from tqdm import tqdm
import re
import json
from textwrap import dedent
import os
from typing import Dict, Any, Tuple, Optional

# Expected output keys for each side
EXPECTED_KEYS = [
    "title",
    "manufacturer",
    "price"
]


class OllamaFeatureExtractor:
    def __init__(self, model_name: str = "llama3.1") -> None:
        self.llm_model = model_name

    # -------------------- Coercion & Validation --------------------
    def _coerce_price(self, value: Any) -> Any:
        """Coerce price to float with two decimals, or the literal string 'unknown'."""
        if value is None:
            return "unknown"
        if isinstance(value, (int, float)):
            return float(f"{float(value):.2f}")
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"", "n/a", "na", "none", "null", "unknown"}:
                return "unknown"
            # Remove currency symbols and commas
            v = re.sub(r"[,$]", "", v)
            try:
                return float(f"{float(v):.2f}")
            except Exception:
                return "unknown"
        return "unknown"

    def normalize_llm_output(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all expected keys exist, coerce types, and sanitize values."""
        out: Dict[str, Any] = {}
        # Defaults
        out["title"] = str(response.get("title", "") or "").strip()
        out["manufacturer"] = str(response.get("manufacturer", "") or "").strip()
        out["price"] = self._coerce_price(response.get("price", "unknown"))
        return out

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Robustly extract a single JSON object from the model output."""
        # Strip code fences if present
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"```$", "", text).strip()
        # Heuristic: take the outermost JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
        return json.loads(text)

    # -------------------- LLM prompts (two variants) --------------------
    def _build_prompt_match(self, left: Dict[str, Any], right: Dict[str, Any]) -> str:
        """Prompt A — Label = 1 (MATCH): strong alignment-oriented normalization."""
        return dedent(f"""
        You are a product-normalization expert for software titles. Normalize and ALIGN two Amazon software/product records for DeepMatcher.

        Return a SINGLE valid JSON object with exactly two top-level keys: "left" and "right".
        Each side must follow this schema:
          • "title" (string)
          • "manufacturer" (string)
          • "price" (float or "unknown")

        ALIGNMENT & NORMALIZATION FOR MATCHED PAIRS (label = 1)
        - Aggressively remove noise:
          • Delete alphanumeric SKUs / catalog codes (e.g., "19600061dm", "SF9006").
          • Remove brackets/parentheses that ONLY specify platform/media (e.g., "[Mac]", "(Win 95/98/ME)", "(DVD)").
          • Trim generic trailer phrases (case-insensitive; stop at first match):
            "Full Version of .* Software" · ".* Production Software" · "Sound Editing S/?W" ·
            "Photo Editing Software for Windows" · "Complete (Package|Product)" · "Standard English PC" ·
            "Scientific Brain Training" · "Music Production" · "Qualification" · "Contact Management .*" ·
            "No Limit Texas Hold 'Em" · similar marketing tails.
        - Expand abbreviations/spellings:
          CS1/2/3 → Creative Suite 1/2/3 · CAL → Client Access License · Svr → Server ·
          Upg → Upgrade · OEM → OEM · AV → Anti-Virus · S/W → Software · Win → Windows ·
          Propack → Pro Pack · keep “Host Only”.
        - PRESERVE SPECIFICITY:
          • Keep version/edition/license tokens exactly (CS3, XI, X3, 11.0, 7.3, 2007, Professional, Home, Standard, Upgrade, 3-User, Host Only, Boxed).
          • If a version/edition appears on only one side and there is NO conflicting version/edition on the other side, COPY it so both sides align to the most specific shared product.
        - Casing & whitespace: Title Case; collapse multiple spaces; dedupe consecutive duplicate words.
        - Manufacturer canonicalization: shortest unambiguous form (e.g., “Adobe Systems Inc” → “Adobe”; “Microsoft Corporation” → “Microsoft”); drop Inc., Ltd., Corp., Software unless needed to disambiguate.
        - Missing values: empty title/manufacturer → ""; price: valid number → float with two decimals; else "unknown".
        - NEVER invent prices. Do not copy a price from one side to the other.

        FEW-SHOT EXAMPLES
        A1. Microsoft Digital Image Suite Plus (match)
        Left ⟶  {{"title":"Microsoft Digital Image Suite Plus","manufacturer":"Microsoft","price":129.95}}
        Right ⟶ {{"title":"Microsoft Digital Image Suite Plus Full Version of Photo Editing Software for Windows","manufacturer":"","price":89.95}}
        Output
        {{
          "left":  {{"title":"Microsoft Digital Image Suite Plus","manufacturer":"Microsoft","price":129.95}},
          "right": {{"title":"Microsoft Digital Image Suite Plus","manufacturer":"","price":89.95}}
        }}

        A2. M-Audio Pro Tools M-Powered 7.3 (match)
        Left ⟶  {{"title":"M-Audio Pro Tools M-Powered 7.3","manufacturer":"M-Audio","price":299.99}}
        Right ⟶ {{"title":"M-Audio Pro Tools M-Powered Software Music Production","manufacturer":"","price":249.00}}
        Output
        {{
          "left":  {{"title":"M-Audio Pro Tools M-Powered 7.3","manufacturer":"M-Audio","price":299.99}},
          "right": {{"title":"M-Audio Pro Tools M-Powered 7.3","manufacturer":"","price":249.00}}
        }}

        A3. Symantec pcAnywhere 11.0 Host Only (match)
        Left ⟶  {{"title":"Symantec Pcanywhere 11.0 Host Only Cd-Rom Xp 98 Nt W2k Me","manufacturer":"Symantec","price":"unknown"}}
        Right ⟶ {{"title":"Symantec Pcanywhere 11.0 Windows","manufacturer":"","price":19.99}}
        Output
        {{
          "left":  {{"title":"Symantec Pcanywhere 11.0 Host Only","manufacturer":"Symantec","price":"unknown"}},
          "right": {{"title":"Symantec Pcanywhere 11.0 Host Only","manufacturer":"","price":19.99}}
        }}

        OUTPUT RULES — STRICT
        - Return exactly one JSON object.
        - No code fences/markdown/comments/logs.
        - Keys must be exactly: left.title, left.manufacturer, left.price, right.title, right.manufacturer, right.price.
        - Price must be float (two decimals) or "unknown".

        Now process this record:
        
        Left record input:
        {json.dumps(left, ensure_ascii=False, indent=2)}

        Right record input:
        {json.dumps(right, ensure_ascii=False, indent=2)}
        """)

    def _build_prompt_nonmatch(self, left: Dict[str, Any], right: Dict[str, Any]) -> str:
        """Prompt B — Label = 0 (NON-MATCH): light, conservative cleanup without alignment."""
        return dedent(f"""
        You are a product-normalization expert for software titles. Lightly CLEAN two Amazon software/product records for DeepMatcher WITHOUT aligning them. Preserve discriminative tokens and platform/media cues.

        Return a SINGLE valid JSON object with exactly two top-level keys: "left" and "right".
        Each side must follow this schema:
          • "title" (string)
          • "manufacturer" (string)
          • "price" (float or "unknown")

        LIGHT NORMALIZATION FOR NON-MATCHED PAIRS (label = 0)
        - DO NOT remove platform/media tags in brackets/parentheses (e.g., "[Mac]", "(Windows)", "(DVD)").
        - DO NOT trim generic trailer phrases; keep marketing tails and qualifiers.
        - DO NOT delete alphanumeric SKUs / catalog codes; keep them.
        - DO NOT propagate or copy version/edition/license tokens across sides.
        - Abbreviation expansion: avoid expanding (Win, CS3, Pro, etc.) to keep original distinctions, except simple punctuation/casing fixes.
        - Preserve specificity: keep all version/edition/license tokens exactly as given.
        - Casing & whitespace: convert to Title Case; collapse multiple spaces; remove consecutive duplicate words.
        - Manufacturer canonicalization: shorten obvious suffixes (Inc., Ltd., Corp., Software) when unambiguous; do NOT force two different brands to match.
        - Missing values: empty title/manufacturer → ""; price: valid number → float with two decimals; else "unknown".
        - NEVER invent prices.

        FEW-SHOT EXAMPLES
        B1. Adobe Photoshop CS3 vs CS2 (non-match)
        Left ⟶  {{"title":"Adobe Photoshop CS3","manufacturer":"Adobe","price":649.00}}
        Right ⟶ {{"title":"Adobe Systems Inc Adobe Photoshop CS2 Mac OS X v10.2.8 to 10.3","manufacturer":"Adobe Systems Inc","price":788.63}}
        Output
        {{
          "left":  {{"title":"Adobe Photoshop CS3","manufacturer":"Adobe","price":649.00}},
          "right": {{"title":"Adobe Photoshop CS2 Mac OS X 10.2.8 To 10.3","manufacturer":"Adobe","price":788.63}}
        }}

        B2. Corel WordPerfect Office X3 Editions (non-match)
        Left ⟶  {{"title":"Corel WordPerfect Office X3 Professional Edition","manufacturer":"Corel","price":399.99}}
        Right ⟶ {{"title":"Corel WordPerfect Office X3 Home Edition","manufacturer":"","price":99.99}}
        Output
        {{
          "left":  {{"title":"Corel Wordperfect Office X3 Professional Edition","manufacturer":"Corel","price":399.99}},
          "right": {{"title":"Corel Wordperfect Office X3 Home Edition","manufacturer":"","price":99.99}}
        }}

        B3. QuickBooks Pro 2007 Mac vs Windows (non-match)
        Left ⟶  {{"title":"QuickBooks Pro 2007 for Mac","manufacturer":"Intuit","price":199.95}}
        Right ⟶ {{"title":"Intuit QuickBooks Pro 2007 Software for Windows Tax & Finance Software","manufacturer":"","price":179.95}}
        Output
        {{
          "left":  {{"title":"Quickbooks Pro 2007 For Mac","manufacturer":"Intuit","price":199.95}},
          "right": {{"title":"Intuit Quickbooks Pro 2007 Software For Windows Tax & Finance Software","manufacturer":"","price":179.95}}
        }}

        OUTPUT RULES — STRICT
        - Return exactly one JSON object.
        - No code fences/markdown/comments/logs.
        - Keys must be exactly: left.title, left.manufacturer, left.price, right.title, right.manufacturer, right.price.
        - Price must be float (two decimals) or "unknown".

        Now process this record:
        
        Left record input:
        {json.dumps(left, ensure_ascii=False, indent=2)}

        Right record input:
        {json.dumps(right, ensure_ascii=False, indent=2)}
        """)

    # -------------------- LLM call --------------------
    def _chat_json(self, prompt: str) -> Dict[str, Any]:
        response = ollama.chat(
            model=self.llm_model,
            options={"temperature": 0.0, "num_predict": 1024},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful information extractor. Output only valid JSON. "
                        "Do not include explanations, markdown fences, comments, or extra text. "
                        "Return exactly one JSON object conforming to the requested schema."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = response["message"]["content"].strip()
        try:
            return self._extract_json(content)
        except json.JSONDecodeError as jde:
            # Try a second pass by removing everything before first '{' and after last '}'
            try:
                return self._extract_json(content)
            except Exception:
                print(f"❌ JSON decode error: {jde}")
                print("⚠️ Content that failed parsing:", content)
                raise

    # -------------------- Main extraction API --------------------
    def extract_pair_standardized_attributes(
        self,
        left_record: Dict[str, Any],
        right_record: Dict[str, Any],
        label: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Choose prompt by label when available:
          - label == 1 → strong, alignment-oriented normalization (match)
          - label == 0 → light, conservative cleanup (non-match)
          - label is None → default to non-match prompt (safer at inference)
        """
        if label == 1:
            prompt = self._build_prompt_match(left_record, right_record)
        else:
            prompt = self._build_prompt_nonmatch(left_record, right_record)

        try:
            parsed = self._chat_json(prompt)
            left_out = self.normalize_llm_output(parsed.get("left", {}))
            right_out = self.normalize_llm_output(parsed.get("right", {}))
            print("left :",left_out,"---- right:",right_out)
            return left_out, right_out
        except Exception as e:
            print(f"❌ Extraction error: {e}")
            # Fallback to minimally cleaned original inputs
            return self.normalize_llm_output(left_record), self.normalize_llm_output(right_record)

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

            raw_label = row_dict.get("label", None)
            try:
                label_val: Optional[int] = int(raw_label) if pd.notna(raw_label) else None
            except Exception:
                label_val = None

            left_cleaned, right_cleaned = self.extract_pair_standardized_attributes(
                left_input, right_input, label=label_val
            )

            new_row: Dict[str, Any] = {
                "id": row_dict.get("id"),
                "label": raw_label,
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
