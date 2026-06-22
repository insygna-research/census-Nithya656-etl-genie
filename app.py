"""
ETL Genie — Streamlit dashboard

A browser-based demo UI for the agent: upload or select a CSV, run the
pipeline, and inspect results across four tabs (run log, output data,
quarantine, run report).

Usage:
    streamlit run app.py
"""

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="ETL Genie", page_icon="🧞", layout="wide")

st.title("🧞 ETL Genie")
st.caption("Local agentic ETL pipeline — extract, validate, transform, load")
st.divider()

# ── sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    uploaded_file = st.file_uploader(
        "Upload a CSV file", type=["csv"],
        help="Upload the source CSV file to process",
    )

    data_dir   = Path("data")
    csv_files  = [f.name for f in data_dir.glob("*.csv") if not f.name.startswith("_")]
    chosen_file = st.selectbox("Or choose an existing file", options=["-- select --"] + csv_files)

    st.divider()
    run_button = st.button("▶ Run ETL Agent", type="primary", use_container_width=True)
    st.divider()
    st.caption("Phase 1 — Local sandbox")
    st.caption("LLM: Llama 3.1 via Ollama")
    st.caption("Framework: LangGraph")

filename = None
if uploaded_file:
    save_path = data_dir / uploaded_file.name
    save_path.write_bytes(uploaded_file.getvalue())
    filename = uploaded_file.name
    st.sidebar.success(f"Uploaded: {uploaded_file.name}")
elif chosen_file != "-- select --":
    filename = chosen_file

# ── tabs ──────────────────────────────────────────────────────────────────
tab_log, tab_output, tab_quarantine, tab_report = st.tabs(
    ["📋 Run log", "✅ Output data", "⚠️ Quarantine", "📄 Run report"]
)

# ── run agent ────────────────────────────────────────────────────────────
if run_button:
    if not filename:
        st.sidebar.error("Please upload or select a file first.")
    else:
        with tab_log:
            st.subheader(f"Processing: {filename}")
            progress = st.progress(0)
            status   = st.empty()
            log_area = st.empty()

            from agent import run_etl_agent
            from tools import run_stats

            class StreamCapture:
                def __init__(self, original):
                    self.original = original
                    self.captured = []

                def write(self, text):
                    self.original.write(text)
                    if text.strip():
                        self.captured.append(text.strip())
                        log_area.code("\n".join(self.captured), language=None)

                def flush(self):
                    self.original.flush()

            capture = StreamCapture(sys.stdout)
            sys.stdout = capture

            start = time.time()
            try:
                run_etl_agent(filename)
                elapsed = round(time.time() - start, 1)
                progress.progress(1.0)
                status.success(f"Completed in {elapsed}s")
            except Exception as e:
                status.error(f"Agent error: {e}")
            finally:
                sys.stdout = capture.original

            st.session_state["run_stats"] = dict(run_stats)
            st.session_state["ran"]       = True
            st.session_state["filename"]  = filename

# ── output tab ───────────────────────────────────────────────────────────
with tab_output:
    output_files = sorted(Path("output").glob("sales_clean_*.parquet"))
    if output_files:
        df = pd.read_parquet(output_files[-1])
        st.subheader("Clean output data")

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows loaded", len(df))
        col2.metric("Columns", len(df.columns))
        col3.metric("Output file", output_files[-1].name)

        st.dataframe(df, use_container_width=True)
        st.download_button(
            "⬇ Download as CSV",
            data=df.to_csv(index=False),
            file_name=output_files[-1].stem + ".csv",
            mime="text/csv",
        )
    else:
        st.info("No output file yet. Run the agent first.")

# ── quarantine tab ───────────────────────────────────────────────────────
with tab_quarantine:
    q_files = sorted(Path("quarantine").glob("quarantine_*.csv"))
    if q_files:
        qdf = pd.read_csv(q_files[-1])
        st.subheader("Quarantined records")
        st.metric("Records quarantined", len(qdf))

        def highlight_issues(row):
            colors = []
            for col in row.index:
                if pd.isnull(row[col]):
                    colors.append("background-color: #fff3cd")
                elif col == "unit_price" and pd.to_numeric(row[col], errors="coerce") < 0:
                    colors.append("background-color: #f8d7da")
                else:
                    colors.append("")
            return colors

        st.dataframe(qdf.style.apply(highlight_issues, axis=1), use_container_width=True)
        st.caption("🟡 Yellow = null value   🔴 Red = invalid value")
    else:
        st.info("No quarantine file yet. Run the agent first.")

# ── report tab ───────────────────────────────────────────────────────────
with tab_report:
    report_files = sorted(Path("output").glob("run_report_*.txt"))
    if report_files:
        report_text = report_files[-1].read_text()
        st.subheader("Run report")

        lines = report_text.split("\n")
        metrics = {}
        for line in lines:
            if ":" in line:
                k, v = line.split(":", 1)
                metrics[k.strip()] = v.strip()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Extracted",   metrics.get("Rows extracted", "-"))
        col2.metric("Quarantined", metrics.get("Rows quarantined", "-"))
        col3.metric("Loaded",      metrics.get("Rows loaded", "-"))
        col4.metric("Issues",      metrics.get("Issues found", "-"))

        st.divider()
        st.code(report_text, language=None)
        st.download_button(
            "⬇ Download report",
            data=report_text,
            file_name=report_files[-1].name,
            mime="text/plain",
        )
    else:
        st.info("No report yet. Run the agent first.")
