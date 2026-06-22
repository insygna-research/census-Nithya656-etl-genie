"""
ETL Genie — Tool functions
Each function is an independently testable Python function exposed as a
LangChain tool. The agent's LangGraph pipeline (see agent.py) calls these
in a fixed sequence, with conditional abort routing on failure.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel
from langchain_core.tools import tool

# ── paths ─────────────────────────────────────────────────────────────────
DATA_DIR       = Path("data")
OUTPUT_DIR     = Path("output")
QUARANTINE_DIR = Path("quarantine")

OUTPUT_DIR.mkdir(exist_ok=True)
QUARANTINE_DIR.mkdir(exist_ok=True)


# ── expected schema ──────────────────────────────────────────────────────
class SalesSchema(BaseModel):
    order_id:      int
    customer_name: str
    product:       str
    quantity:      int
    unit_price:    float
    order_date:    str
    region:        str
    status:        str


EXPECTED_COLUMNS = list(SalesSchema.model_fields.keys())

# ── shared run stats (read/written by every tool + the run report) ──────
run_stats = {
    "rows_extracted":   0,
    "rows_quarantined": 0,
    "rows_loaded":      0,
    "issues":           [],
    "start_time":       None,
    "aborted":          False,
    "abort_reason":     "",
}


# ── helper ────────────────────────────────────────────────────────────────
def load_working():
    """Load the working CSV. Returns (df, error_message)."""
    path = DATA_DIR / "_working.csv"
    if not path.exists():
        return None, "No data loaded yet. Call read_source_data first."
    df = pd.read_csv(path)
    return df, ""


# ══════════════════════════════════════════════════════════════════════
# TOOL 1 — read_source_data
# ══════════════════════════════════════════════════════════════════════
@tool
def read_source_data(filename: str) -> str:
    """Read a CSV file from the data folder and return a confirmation with row and column count."""
    path = DATA_DIR / filename
    if not path.exists():
        msg = f"ABORT: File '{filename}' not found in data folder."
        run_stats["aborted"] = True
        run_stats["abort_reason"] = msg
        return msg
    try:
        df = pd.read_csv(path)
    except Exception as e:
        msg = f"ABORT: Could not read file '{filename}'. Error: {e}"
        run_stats["aborted"] = True
        run_stats["abort_reason"] = msg
        return msg

    if len(df) == 0:
        msg = f"ABORT: File '{filename}' has no data rows — only headers or completely empty."
        run_stats["aborted"] = True
        run_stats["abort_reason"] = msg
        return msg

    run_stats["rows_extracted"] = len(df)
    run_stats["start_time"]     = datetime.now().isoformat()
    run_stats["aborted"]        = False
    run_stats["abort_reason"]   = ""
    df.to_csv(DATA_DIR / "_working.csv", index=False)
    return f"SUCCESS: Loaded {len(df)} rows and {len(df.columns)} columns. Columns: {list(df.columns)}"


# ══════════════════════════════════════════════════════════════════════
# TOOL 2 — profile_data
# ══════════════════════════════════════════════════════════════════════
@tool
def profile_data(dummy: str = "") -> str:
    """Profile the currently loaded dataset. Returns row count, column names, data types, and null counts per column."""
    df, err = load_working()
    if err:
        return f"ERROR: {err}"
    profile = {
        "row_count":   len(df),
        "columns":     list(df.columns),
        "dtypes":      df.dtypes.astype(str).to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "sample_row":  df.iloc[0].to_dict() if len(df) > 0 else {},
    }
    return json.dumps(profile, default=str)


# ══════════════════════════════════════════════════════════════════════
# TOOL 3 — validate_schema
# ══════════════════════════════════════════════════════════════════════
@tool
def validate_schema(dummy: str = "") -> str:
    """Validate that the loaded data has the expected columns and flag any missing or extra columns."""
    df, err = load_working()
    if err:
        return f"ERROR: {err}"
    actual   = set(df.columns)
    expected = set(EXPECTED_COLUMNS)
    missing  = expected - actual
    extra    = actual - expected
    if not missing and not extra:
        return "PASS: Schema is valid. All expected columns present."

    issues = []
    if missing:
        issues.append(f"Missing columns: {list(missing)}")
    if extra:
        issues.append(f"Extra columns: {list(extra)}")
    run_stats["issues"].extend(issues)

    # abort if more than half the expected columns are missing — too
    # different from the target schema to safely continue
    if len(missing) > len(expected) / 2:
        msg = f"ABORT: Schema too different to process. {issues}"
        run_stats["aborted"]      = True
        run_stats["abort_reason"] = msg
        return msg

    return "FAIL: " + " | ".join(issues)


# ══════════════════════════════════════════════════════════════════════
# TOOL 4 — check_data_quality
# ══════════════════════════════════════════════════════════════════════
@tool
def check_data_quality(dummy: str = "") -> str:
    """Check the loaded data for quality issues: nulls, duplicates, negative prices, and invalid quantities."""
    df, err = load_working()
    if err:
        return f"ERROR: {err}"

    issues = []
    null_counts = df.isnull().sum()
    for col, cnt in null_counts.items():
        if cnt > 0:
            issues.append(f"Column '{col}' has {cnt} null values")

    dup_count = df.duplicated().sum()
    if dup_count > 0:
        issues.append(f"{dup_count} duplicate rows found")

    if "unit_price" in df.columns:
        try:
            neg = (pd.to_numeric(df["unit_price"], errors="coerce") < 0).sum()
            if neg > 0:
                issues.append(f"{neg} rows have negative unit_price")
        except Exception:
            issues.append("Could not validate unit_price column")

    if "quantity" in df.columns:
        try:
            zero_qty = (pd.to_numeric(df["quantity"], errors="coerce") <= 0).sum()
            if zero_qty > 0:
                issues.append(f"{zero_qty} rows have zero or negative quantity")
        except Exception:
            issues.append("Could not validate quantity column")

    run_stats["issues"].extend(issues)
    if not issues:
        return "PASS: No data quality issues found."
    return "ISSUES FOUND:\n" + "\n".join(f"  - {i}" for i in issues)


# ══════════════════════════════════════════════════════════════════════
# TOOL 5 — quarantine_records
# ══════════════════════════════════════════════════════════════════════
@tool
def quarantine_records(dummy: str = "") -> str:
    """Move bad records to the quarantine folder and keep only clean records for processing."""
    df, err = load_working()
    if err:
        return f"ERROR: {err}"

    bad_mask = pd.Series([False] * len(df), index=df.index)

    if "customer_name" in df.columns:
        bad_mask |= df["customer_name"].isnull()
    if "region" in df.columns:
        bad_mask |= df["region"].isnull()
    if "unit_price" in df.columns:
        bad_mask |= pd.to_numeric(df["unit_price"], errors="coerce") < 0
    bad_mask |= df.duplicated()

    bad_df  = df[bad_mask]
    good_df = df[~bad_mask]

    if len(good_df) == 0:
        msg = "ABORT: All records failed quality checks — no clean data to process."
        run_stats["aborted"]          = True
        run_stats["abort_reason"]     = msg
        run_stats["rows_quarantined"] = len(bad_df)
        return msg

    if len(bad_df) > 0:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bad_df.to_csv(QUARANTINE_DIR / f"quarantine_{ts}.csv", index=False)
        run_stats["rows_quarantined"] = len(bad_df)

    good_df.to_csv(DATA_DIR / "_working.csv", index=False)
    return f"Quarantined {len(bad_df)} bad records. {len(good_df)} clean records remain for processing."


# ══════════════════════════════════════════════════════════════════════
# TOOL 6 — clean_data
# ══════════════════════════════════════════════════════════════════════
@tool
def clean_data(dummy: str = "") -> str:
    """Clean the data: strip whitespace, fix data types, standardise date format."""
    df, err = load_working()
    if err:
        return f"ERROR: {err}"
    try:
        str_cols = df.select_dtypes(include="object").columns
        df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())
        df["order_id"]   = pd.to_numeric(df["order_id"],   errors="coerce").fillna(0).astype(int)
        df["quantity"]   = pd.to_numeric(df["quantity"],   errors="coerce").fillna(0).astype(int)
        df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").fillna(0.0)
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df.to_csv(DATA_DIR / "_working.csv", index=False)
        return f"SUCCESS: Data cleaned. {len(df)} rows ready for transformation."
    except Exception as e:
        return f"ERROR during cleaning: {e}"


# ══════════════════════════════════════════════════════════════════════
# TOOL 7 — apply_business_rules
# ══════════════════════════════════════════════════════════════════════
@tool
def apply_business_rules(dummy: str = "") -> str:
    """Apply business rules: calculate total_value, standardise region and status."""
    df, err = load_working()
    if err:
        return f"ERROR: {err}"
    try:
        df["total_value"] = df["quantity"] * df["unit_price"]
        if "region" in df.columns:
            df["region"] = df["region"].str.title()
        if "status" in df.columns:
            df["status"] = df["status"].str.lower()
        df.to_csv(DATA_DIR / "_working.csv", index=False)
        return f"SUCCESS: Business rules applied. Added total_value column. {len(df)} rows processed."
    except Exception as e:
        return f"ERROR applying business rules: {e}"


# ══════════════════════════════════════════════════════════════════════
# TOOL 8 — deduplicate
# ══════════════════════════════════════════════════════════════════════
@tool
def deduplicate(dummy: str = "") -> str:
    """Remove duplicate rows based on order_id. Keep the first occurrence."""
    df, err = load_working()
    if err:
        return f"ERROR: {err}"
    before = len(df)
    if "order_id" in df.columns:
        df = df.drop_duplicates(subset=["order_id"], keep="first")
    removed = before - len(df)
    df.to_csv(DATA_DIR / "_working.csv", index=False)
    return f"SUCCESS: Removed {removed} duplicate order_ids. {len(df)} rows remain."


# ══════════════════════════════════════════════════════════════════════
# TOOL 9 — validate_output
# ══════════════════════════════════════════════════════════════════════
@tool
def validate_output(dummy: str = "") -> str:
    """Validate the processed data before writing."""
    df, err = load_working()
    if err:
        return f"ERROR: {err}"
    issues = []
    if len(df) == 0:
        issues.append("Output has 0 rows — something went wrong")
    loss_pct = (run_stats["rows_extracted"] - len(df)) / max(run_stats["rows_extracted"], 1) * 100
    if loss_pct > 80:
        issues.append(f"Row loss is {loss_pct:.1f}% — exceeds 80% threshold")
    required = ["order_id", "customer_name", "product"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        issues.append(f"Missing required output columns: {missing}")
    if issues:
        return "VALIDATION FAILED:\n" + "\n".join(f"  - {i}" for i in issues)
    return f"PASS: Output looks good. {len(df)} rows, row loss is {loss_pct:.1f}%."


# ══════════════════════════════════════════════════════════════════════
# TOOL 10 — write_output
# ══════════════════════════════════════════════════════════════════════
@tool
def write_output(dummy: str = "") -> str:
    """Write the final cleaned data to the output folder as a Parquet file."""
    df, err = load_working()
    if err:
        return f"ERROR: {err}"
    try:
        ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"sales_clean_{ts}.parquet"
        df.to_parquet(output_path, index=False)
        run_stats["rows_loaded"] = len(df)
        return f"SUCCESS: Written {len(df)} rows to {output_path}"
    except Exception as e:
        return f"ERROR writing output: {e}"


# ══════════════════════════════════════════════════════════════════════
# TOOL 11 — generate_run_report
# ══════════════════════════════════════════════════════════════════════
@tool
def generate_run_report(dummy: str = "") -> str:
    """Generate a summary report of the entire ETL run."""
    end_time = datetime.now()
    start    = datetime.fromisoformat(run_stats["start_time"]) if run_stats["start_time"] else end_time
    duration = round((end_time - start).total_seconds(), 1)

    status = "ABORTED" if run_stats["aborted"] else "COMPLETED"

    report_lines = [
        "=" * 50,
        f"ETL AGENT RUN REPORT — {status}",
        "=" * 50,
        f"Run completed   : {end_time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Duration        : {duration} seconds",
        f"Rows extracted  : {run_stats['rows_extracted']}",
        f"Rows quarantined: {run_stats['rows_quarantined']}",
        f"Rows loaded     : {run_stats['rows_loaded']}",
        f"Issues found    : {len(run_stats['issues'])}",
    ]
    if run_stats["aborted"]:
        report_lines.append(f"\nAbort reason: {run_stats['abort_reason']}")
    if run_stats["issues"]:
        report_lines.append("\nIssues detail:")
        for issue in run_stats["issues"]:
            report_lines.append(f"  - {issue}")
    report_lines.append("=" * 50)

    report = "\n".join(report_lines)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = OUTPUT_DIR / f"run_report_{ts}.txt"
    report_path.write_text(report)
    return report
