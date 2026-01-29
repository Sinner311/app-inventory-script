import os
import yaml
import pandas as pd
from dotenv import load_dotenv
import re


# ---------- Load ENV ----------
load_dotenv()
MASTER_FILE = os.getenv("MASTER_FILE")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

# ---------- Utils ----------
def load_template(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def normalize_col(col: str) -> str:
    col = str(col)
    col = col.replace("\n", " ").replace("/", " ")
    col = col.replace("\u00a0", " ")
    col = re.sub(r"[^a-zA-Z0-9 ]", "", col)
    col = re.sub(r"\s+", " ", col)

    return col.strip().lower()

def load_master(path):
    df = pd.read_excel(path , header=9)

    df = df.ffill(axis=1)
    df = df.ffill(axis=0)

    print("\n=== RAW COLUMNS ===")
    for c in df.columns:
        print(repr(c))

    df.columns = [normalize_col(c) for c in df.columns]

    return df

def normalize_value(val):
    if pd.isna(val):
        return ""
    return normalize_col(val)


def apply_filter(df, filters):
    if not filters:
        return df

    for raw_col, condition in filters.items():
        col = normalize_col(raw_col)

        if col not in df.columns:
            raise ValueError(
                f"Filter column not found: {raw_col}\nAvailable: {list(df.columns)}"
            )

        series = df[col].apply(normalize_value)

        if isinstance(condition, dict):
            if "in" in condition:
                values = [normalize_value(v) for v in condition["in"]]
                df = df[series.isin(values)]
            elif "neq" in condition:
                df = df[series != normalize_value(condition["neq"])]
        else:
            df = df[series == normalize_value(condition)]

    return df




# ---------- Main Process ----------
def run(template_path):
    print(f"▶ Running template: {template_path}")

    template = load_template(template_path)

    # Load master ONCE
    df_master = load_master(MASTER_FILE)

    # Output folder (custom)
    template_name = os.path.splitext(os.path.basename(template_path))[0]
    folder_name = template.get("output_folder", template_name)
    output_base = os.path.join(OUTPUT_DIR, folder_name)
    ensure_folder(output_base)

    outputs = template.get("outputs", [])

    if not outputs:
        raise ValueError("No outputs defined in template")

    for item in outputs:
        print(f"  ➜ Creating file: {item['file_name']}")

        df = df_master.copy()

        # Apply filter
        df = apply_filter(df, item.get("filter"))

        # Normalize columns
        template_columns = [normalize_col(c) for c in item["columns"]]

        # Validate columns
        for col in template_columns:
            if col not in df.columns:
                raise ValueError(f"Column not found in master: {col}")

        # Select columns
        df = df[template_columns]

        # Write output
        output_path = os.path.join(output_base, item["file_name"])
        df.to_excel(output_path, index=False)

        print(f"    ✔ Rows: {len(df)} → {output_path}")


    # ---------- Entry ----------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py <template.yaml | all>")
        sys.exit(1)

    arg = sys.argv[1]

    if arg == "all":
        for file in os.listdir("templates"):
            if file.endswith(".yaml"):
                run(os.path.join("templates", file))
    else:
        run(arg)
