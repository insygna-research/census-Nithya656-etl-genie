"""
ETL Genie — Entry point

Run the agent end-to-end against the sample dataset and preview the
output, quarantine, and run report. This is the quickest way to see the
full pipeline working after setup.

Usage:
    python main.py
"""

from pathlib import Path

import pandas as pd

from agent import run_etl_agent

# ── run the agent ────────────────────────────────────────────────────────
run_etl_agent("sales_raw.csv")

# ── preview the output ───────────────────────────────────────────────────
print("\n── Output file preview ──")
output_files = sorted(Path("output").glob("sales_clean_*.parquet"))
if output_files:
    df = pd.read_parquet(output_files[-1])
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(df.to_string(index=False))
else:
    print("No output file found.")

# ── preview the quarantine ───────────────────────────────────────────────
print("\n── Quarantine file preview ──")
q_files = sorted(Path("quarantine").glob("quarantine_*.csv"))
if q_files:
    qdf = pd.read_csv(q_files[-1])
    print(f"Shape: {qdf.shape[0]} rows quarantined")
    print(qdf.to_string(index=False))
else:
    print("No quarantine file found.")

# ── print the run report ─────────────────────────────────────────────────
print("\n── Run report ──")
report_files = sorted(Path("output").glob("run_report_*.txt"))
if report_files:
    print(report_files[-1].read_text())
else:
    print("No report file found.")
