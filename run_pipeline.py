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

def validate_template(template: dict):
    if "outputs" not in template:
        raise ValueError("Template missing 'outputs'")

    if not isinstance(template["outputs"], list):
        raise ValueError("'outputs' must be a list")

    for i, output in enumerate(template["outputs"]):
        prefix = f"outputs[{i}]"

        required_keys = ["file_name", "columns"]

        for k in required_keys:
            if k not in output:
                raise ValueError(f"{prefix} missing '{k}'")

        if not isinstance(output["columns"], list):
            raise ValueError(f"{prefix}.columns must be a list")

        if "filter" in output and not isinstance(output["filter"], dict):
            raise ValueError(f"{prefix}.filter must be a dict")
        
        



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

            # ---------- IN ----------
            if "in" in condition:
                values = [normalize_value(v) for v in condition["in"]]
                df = df[series.isin(values)]

            # ---------- NOT EQUAL ----------
            elif "neq" in condition:
                df = df[series != normalize_value(condition["neq"])]

            # ---------- CONTAINS (AND) ----------
            elif "contains" in condition or "contains_all" in condition:
                values = condition.get("contains") or condition.get("contains_all")
                values = values if isinstance(values, list) else [values]

                for v in values:
                    keyword = normalize_value(v)
                    df = df[series.str.contains(keyword, na=False)]

            # ---------- CONTAINS (OR) ----------
            elif "contains_any" in condition:
                values = condition["contains_any"]
                values = values if isinstance(values, list) else [values]

                mask = False
                for v in values:
                    keyword = normalize_value(v)
                    mask = mask | series.str.contains(keyword, na=False)

                df = df[mask]

            else:
                raise ValueError(
                    f"Unsupported filter operator for column '{raw_col}': {condition}"
                )

        else:
            df = df[series == normalize_value(condition)]

    return df



def analyze_filters(df, filters):
    if not filters:
        return

    working_df = df.copy()
    start_rows = len(working_df)

    print(f"\n🔍 Filter analysis (start rows = {start_rows})")

    for raw_col, condition in filters.items():
        col = normalize_col(raw_col)

        if col not in working_df.columns:
            raise ValueError(
                f"Filter column not found: '{raw_col}'\n"
                f"Available columns: {working_df.columns.tolist()}"
            )

        series = working_df[col].apply(normalize_value)
        before = len(working_df)

        if isinstance(condition, dict):

            # ---------- IN ----------
            if "in" in condition:
                values = [normalize_value(v) for v in condition["in"]]
                missing = set(values) - set(series.unique())

                if missing:
                    raise ValueError(
                        f"Invalid filter values for '{raw_col}': {missing}"
                    )

                working_df = working_df[series.isin(values)]
                desc = f"in {condition['in']}"

            # ---------- NOT EQUAL ----------
            elif "neq" in condition:
                value = normalize_value(condition["neq"])

                if value not in set(series.unique()):
                    raise ValueError(
                        f"Invalid filter value '{condition['neq']}' "
                        f"for column '{raw_col}'"
                    )

                working_df = working_df[series != value]
                desc = f"!= {condition['neq']}"

            # ---------- CONTAINS (AND) ----------
            elif "contains" in condition or "contains_all" in condition:
                values = condition.get("contains") or condition.get("contains_all")
                values = values if isinstance(values, list) else [values]

                for v in values:
                    keyword = normalize_value(v)

                    if not series.str.contains(keyword, na=False).any():
                        raise ValueError(
                            f"No rows contain '{v}' in column '{raw_col}'"
                        )

                    working_df = working_df[
                        working_df[col].apply(normalize_value)
                        .str.contains(keyword, na=False)
                    ]

                desc = f"contains ALL {values}"

            # ---------- CONTAINS (OR) ----------
            elif "contains_any" in condition:
                values = condition["contains_any"]
                values = values if isinstance(values, list) else [values]

                if not any(
                    series.str.contains(normalize_value(v), na=False).any()
                    for v in values
                ):
                    raise ValueError(
                        f"No rows contain any of {values} in column '{raw_col}'"
                    )

                mask = False
                for v in values:
                    mask = mask | series.str.contains(normalize_value(v), na=False)

                working_df = working_df[mask]
                desc = f"contains ANY {values}"

            else:
                raise ValueError(
                    f"Unsupported filter operator in column '{raw_col}': {condition}"
                )

        else:
            value = normalize_value(condition)

            if value not in set(series.unique()):
                raise ValueError(
                    f"Invalid filter value '{condition}' for column '{raw_col}'"
                )

            working_df = working_df[series == value]
            desc = f"== {condition}"

        after = len(working_df)
        print(f"  • {raw_col} {desc}: {before} → {after}")

        if after == 0:
            raise ValueError(
                f"Filter '{raw_col} {desc}' caused result to be empty"
            )

    print("All filters validated successfully")




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
        analyze_filters(df, item.get("filter"))
        df = apply_filter(df, item.get("filter"))
        if df.empty:
            raise ValueError(
                f"No data after filter for file '{item['file_name']}'.\n"
                f"Filter used: {item.get('filter')}"
            )


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
