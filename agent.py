"""
ETL Genie — Agent pipeline

Built on a LangGraph StateGraph rather than a ReAct loop. Each ETL step is
a graph node; the graph enforces execution order deterministically. The
local LLM (Llama 3.1 via Ollama) is wired in for reasoning but does not
control flow — this is intentional. Small local models are unreliable at
autonomous multi-step tool orchestration, and production data pipelines
need deterministic, debuggable execution order regardless of model size.
"""

import operator
from pathlib import Path
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

from tools import (
    read_source_data, profile_data, validate_schema,
    check_data_quality, quarantine_records, clean_data,
    apply_business_rules, deduplicate, validate_output,
    write_output, generate_run_report, run_stats,
)


def reset_stats():
    """Reset shared run statistics before a fresh agent run."""
    run_stats.update({
        "rows_extracted":   0,
        "rows_quarantined": 0,
        "rows_loaded":      0,
        "issues":           [],
        "start_time":       None,
        "aborted":          False,
        "abort_reason":     "",
    })


class ETLState(TypedDict):
    filename:     str
    step_results: Annotated[list[str], operator.add]
    current_step: str
    error:        str


# Local LLM — available for future reasoning steps (e.g. "should this be
# quarantined or auto-fixed?"). Not used for flow control today.
llm = ChatOllama(model="llama3.1", temperature=0)


def _node(state, step_num, step_name, tool_fn, tool_args):
    """Generic node executor: call one tool, log it, record the result."""
    if run_stats["aborted"]:
        return {
            "step_results": [],
            "current_step": state["current_step"],
            "error":        run_stats["abort_reason"],
        }
    print(f"\n[Step {step_num}/11] {step_name}...")
    result = tool_fn.invoke(tool_args)
    result_str = str(result)
    print(f"  → {result_str[:120]}{'...' if len(result_str) > 120 else ''}")
    return {
        "step_results": [f"STEP {step_num} - {step_name}: {result}"],
        "current_step": step_name,
        "error":        "",
    }


def should_continue(state: ETLState) -> str:
    """Conditional router: abort early and jump to the report node, or continue."""
    return "report" if run_stats["aborted"] else "next"


def run_etl_agent(filename: str):
    """Build and run the ETL pipeline graph for a given source filename."""
    reset_stats()

    working = Path("data/_working.csv")
    if working.exists():
        working.unlink()

    graph = StateGraph(ETLState)

    nodes = {
        "reading_source_data":     lambda s: _node(s, 1,  "Reading source data",     read_source_data,     {"filename": filename}),
        "profiling_data":          lambda s: _node(s, 2,  "Profiling data",          profile_data,         {"dummy": ""}),
        "validating_schema":       lambda s: _node(s, 3,  "Validating schema",       validate_schema,      {"dummy": ""}),
        "checking_data_quality":   lambda s: _node(s, 4,  "Checking data quality",   check_data_quality,   {"dummy": ""}),
        "quarantining_records":    lambda s: _node(s, 5,  "Quarantining records",    quarantine_records,   {"dummy": ""}),
        "cleaning_data":           lambda s: _node(s, 6,  "Cleaning data",           clean_data,           {"dummy": ""}),
        "applying_business_rules": lambda s: _node(s, 7,  "Applying business rules", apply_business_rules, {"dummy": ""}),
        "deduplicating":           lambda s: _node(s, 8,  "Deduplicating",           deduplicate,          {"dummy": ""}),
        "validating_output":       lambda s: _node(s, 9,  "Validating output",       validate_output,      {"dummy": ""}),
        "writing_output":          lambda s: _node(s, 10, "Writing output",          write_output,         {"dummy": ""}),
        "generating_run_report":   lambda s: _node(s, 11, "Generating run report",   generate_run_report,  {"dummy": ""}),
    }

    node_names = list(nodes.keys())
    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.set_entry_point(node_names[0])
    for i in range(len(node_names) - 1):
        graph.add_conditional_edges(
            node_names[i],
            should_continue,
            {"next": node_names[i + 1], "report": node_names[-1]},
        )
    graph.add_edge(node_names[-1], END)
    pipeline = graph.compile()

    print("\n" + "=" * 55)
    print(f"  ETL AGENT STARTING — {filename}")
    print("=" * 55)

    pipeline.invoke({
        "filename":     filename,
        "step_results": [],
        "current_step": "",
        "error":        "",
    })

    print("\n" + "=" * 55)
    print("  ETL AGENT COMPLETE")
    print("=" * 55)
