# walmart_llm_normalizer.py
import pandas as pd
import ollama
from tqdm import tqdm
import re
import json
from textwrap import dedent
import os
from typing import Dict, Any, Tuple, Optional

# Walmart/Amazon record schema we want the LLM to output (per side)
EXPECTED_KEYS = ["title", "category", "brand", "modelno", "price"]


class OllamaFeatureExtractor:
    def __init__(self, model_name: str = "llama3.1") -> None:
        self.llm_model = model_name

    # -------------------- Coercion & Validation (no manual normalization) --------------------
    def _coerce_price(self, value: Any) -> Any:
        """Coerce price to float with two decimals, or the literal string 'unknown'."""
        if value is None:
            return "unknown"
        if isinstance(value, (int, float)):
            try:
                return float(f"{float(value):.2f}")
            except Exception:
                return "unknown"
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"", "n/a", "na", "none", "null", "unknown"}:
                return "unknown"
            v = re.sub(r"[,$]", "", v)
            try:
                return float(f"{float(v):.2f}")
            except Exception:
                return "unknown"
        return "unknown"

    def normalize_llm_output(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Only enforce presence and types; do NOT apply domain rules here.
        All normalization happens INSIDE the prompt.
        """
        out: Dict[str, Any] = {}
        out["title"] = str(response.get("title", "") or "").strip()
        out["category"] = str(response.get("category", "") or "").strip()
        out["brand"] = str(response.get("brand", "") or "").strip()
        out["modelno"] = str(response.get("modelno", "") or "").strip()
        out["price"] = self._coerce_price(response.get("price", "unknown"))
        return out

    # -------------------- JSON extraction --------------------
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Robustly extract a single JSON object from the model output."""
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"```$", "", text).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
        return json.loads(text)

    # -------------------- LLM prompts (Walmart dataset, Amazon-style brevity) --------------------
    def _build_prompt_match(self, left: Dict[str, Any], right: Dict[str, Any]) -> str:
        """Prompt A — Label = 1 (MATCH): strong alignment-oriented normalization."""
        return dedent(f"""
        You are a product-normalization expert. Normalize and ALIGN two Walmart/Amazon product records for DeepMatcher.

        Return a SINGLE valid JSON object with exactly two top-level keys: "left" and "right".
        Each side must follow this schema:
          • "title" (string)
          • "category" (string)
          • "brand" (string)
          • "modelno" (string)
          • "price" (float or "unknown")

        ALIGNMENT & NORMALIZATION FOR MATCHED PAIRS (label = 1)
        - Titles: remove marketing fluff/tails; keep concrete identifiers (series, capacity, size/diagonal, edition, kit names).
        - Casing/spacing: Title Case; collapse spaces; dedupe consecutive duplicate words.
        - Brand: canonicalize to shortest unambiguous form shared by both sides (e.g., “Hewlett Packard” → “HP”).
        - Category: prefer the most specific accurate category implied by both sides (e.g., “Projection Screens”, “USB Flash Drives”).
        - Model number: extract the primary P/N or SKU; uppercase; keep A–Z/0–9 and -._/; drop color-only suffixes (e.g., “/ Navy”).
        - Missing: empty fields → ""; price → float with two decimals if valid, else "unknown".
        - Prices: NEVER fabricate/copy/average across sides.

        

        OUTPUT RULES — STRICT
        - Return exactly one JSON object.
        - No code fences/markdown/comments/logs.
        - Keys must be exactly: left.title, left.category, left.brand, left.modelno, left.price, right.title, right.category, right.brand, right.modelno, right.price.
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
        You are a product-normalization expert. Lightly CLEAN two Walmart/Amazon product records for DeepMatcher WITHOUT aligning them. Preserve discriminative tokens and cues.

        Return a SINGLE valid JSON object with exactly two top-level keys: "left" and "right".
        Each side must follow this schema:
          • "title" (string)
          • "category" (string)
          • "brand" (string)
          • "modelno" (string)
          • "price" (float or "unknown")

        LIGHT NORMALIZATION FOR NON-MATCHED PAIRS (label = 0)
        - Titles: keep platform/spec/marketing descriptors; only fix casing/punctuation and duplicate spaces.
        - Do NOT copy tokens (capacity, series, edition) across sides.
        - Brand: shorten obvious suffixes (Inc., Ltd., Corp.) when unambiguous; do NOT force two brands to match.
        - Category: minor cleanup (Title Case); keep as-is unless a more specific, unambiguous category is obvious.
        - Model number: per side independently; uppercase; keep A–Z/0–9 and -._/; strip color-only suffixes (e.g., “/ Navy”) if clearly color.
        - Missing: empty fields → ""; price → float with two decimals if valid, else "unknown".
        - Prices: NEVER fabricate.

        

        OUTPUT RULES — STRICT
        - Return exactly one JSON object.
        - No code fences/markdown/comments/logs.
        - Keys must be exactly: left.title, left.category, left.brand, left.modelno, left.price, right.title, right.category, right.brand, right.modelno, right.price.
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
        
        print("passed",content)
        try:
            return self._extract_json(content)
        except json.JSONDecodeError as jde:
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
          - label == 1 → alignment-oriented normalization (match)
          - label == 0 or None → conservative cleanup (non-match)
        All domain logic is inside the prompt; Python only coerces types.
        """
        if label == 1:
            prompt = self._build_prompt_match(left_record, right_record)
        else:
            prompt = self._build_prompt_nonmatch(left_record, right_record)

        try:
            parsed = self._chat_json(prompt)
            left_out = self.normalize_llm_output(parsed.get("left", {}))
            right_out = self.normalize_llm_output(parsed.get("right", {}))
            return left_out, right_out
        except Exception as e:
            print(f"❌ Extraction error: {e}")
            # Fallback: minimally cleaned originals (no domain normalization)
            return self.normalize_llm_output(left_record), self.normalize_llm_output(right_record)

    # -------------------- Dataset utilities --------------------
    def split_record(self, row: Dict[str, Any], side: str) -> Dict[str, Any]:
        """
        Map CSV columns:
          left_title,right_title,left_category,right_category,left_brand,right_brand,left_modelno,right_modelno,left_price,right_price
        into per-side dicts with keys: title, category, brand, modelno, price
        """
        return {
            "title": row.get(f"{side}_title"),
            "category": row.get(f"{side}_category"),
            "brand": row.get(f"{side}_brand"),
            "modelno": row.get(f"{side}_modelno"),
            "price": row.get(f"{side}_price"),
        }

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
    extractor = OllamaFeatureExtractor(model_name="llama3.1")

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
