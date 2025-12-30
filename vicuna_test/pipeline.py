import os, json, re, argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "lmsys/vicuna-7b-v1.5"

def build_prompt(col, dtype, raw_min, raw_max, samples):
    # Keep it short + strict. Use tags to reliably extract.
    # Also: tell it to output ONLY the JSON between tags.
    return f"""
You are analyzing dataset column metadata.

Column name: {col}
Pandas dtype: {dtype}
Observed min (if numeric): {raw_min}
Observed max (if numeric): {raw_max}
Sample values: {samples}

Classify semantic_type as one of:
- numeric_measure
- identifier
- categorical
- text
- datetime
Then decide if min/max is semantically meaningful for this column.

Output ONLY one JSON object wrapped exactly like:
<JSON>{{"semantic_type":"...","min_max_applicable":true,"short_reason":"..."}}</JSON>

No extra text. No markdown.
""".strip()

def extract_json(text: str):
    # Prefer the <JSON>...</JSON> block
    m = re.search(r"<JSON>\s*(\{.*?\})\s*</JSON>", text, flags=re.DOTALL)
    if m:
        block = m.group(1).strip()
        try:
            return json.loads(block), None
        except Exception as e:
            return None, f"JSON parse failed in <JSON> block: {e}"

    # Fallback: first {...}
    m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if m:
        block = m.group(1).strip()
        try:
            return json.loads(block), None
        except Exception as e:
            return None, f"JSON parse failed in fallback block: {e}"

    return None, "No JSON found"

def to_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no"):
            return False
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--max_rows", type=int, default=None, help="optional sampling for speed")
    ap.add_argument("--n_samples", type=int, default=8, help="how many sample values to show the model per column")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.max_rows:
        df = df.head(args.max_rows)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    rows = []
    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)

        is_numeric = pd.api.types.is_numeric_dtype(series)
        raw_min = float(series.min()) if is_numeric else None
        raw_max = float(series.max()) if is_numeric else None

        # Sample values (as strings), include NA if present
        sample_vals = series.dropna().astype(str).head(args.n_samples).tolist()
        if len(sample_vals) == 0:
            sample_vals = ["<ALL_NULL>"]

        prompt = build_prompt(col, dtype, raw_min, raw_max, sample_vals)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=220,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )

        # IMPORTANT: decode only the generated continuation, not the prompt+continuation
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        completion = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        j, err = extract_json(completion)
        if j is None:
            j = {
                "semantic_type": "unknown",
                "min_max_applicable": False,
                "short_reason": (err or "Could not parse JSON") + " | " + completion[:160].replace("\n", " ")
            }

        min_app = to_bool(j.get("min_max_applicable"))
        if min_app and is_numeric:
            min_val, max_val = raw_min, raw_max
            reason = j.get("short_reason", "")
        else:
            min_val, max_val = "NA", "NA"
            reason = j.get("short_reason", "")

        rows.append({
            "column_name": col,
            "dtype": dtype,
            "semantic_type": j.get("semantic_type", "unknown"),
            "min_value": min_val,
            "max_value": max_val,
            "reason": reason
        })

    out_df = pd.DataFrame(rows)
    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Saved: {args.output_csv}")

if __name__ == "__main__":
    main()