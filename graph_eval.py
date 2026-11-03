"""LangGraph RCA evaluation across Toph Eval scenarios."""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import LLM_MODEL
from app.data.loader import load_scenarios, load_ground_truth
from app.workflow import run_incident
from evaluation.baseline_rca import JUDGE_PROMPT, call_llm, parse_judge_scores

OUT = os.path.join(os.path.dirname(__file__), "graph_results.csv")
BASELINE_OUT = os.path.join(os.path.dirname(__file__), "baseline_results.csv")
FIELDS = ["scenario_id", "system_identification", "error_classification", "fix_correctness", "grader_result", "retry_count", "model"]
DIMS = ["system_identification", "error_classification", "fix_correctness"]


def load_existing():
    if not os.path.isfile(OUT):
        return []
    with open(OUT, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        for r in rows:
            for d in DIMS + ["retry_count"]:
                if d in r:
                    r[d] = int(r[d])
        return rows


def save(results):
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(results)


def print_comparison(graph_results):
    if not os.path.isfile(BASELINE_OUT):
        return
    with open(BASELINE_OUT, "r", encoding="utf-8") as f:
        base_rows = {r["scenario_id"]: r for r in csv.DictReader(f)}
    graph_map = {r["scenario_id"]: r for r in graph_results}
    common = [sid for sid in graph_map if sid in base_rows]
    if not common:
        return
    print(f"\n--- Baseline vs LangGraph ({len(common)} common scenarios) ---")
    print(f"{'Dimension':<25} | {'Baseline':<10} | {'LangGraph':<10}")
    print("-" * 52)
    for d in DIMS:
        b_sum = sum(int(base_rows[s][d]) for s in common)
        g_sum = sum(int(graph_map[s][d]) for s in common)
        print(f"{d:<25} | {b_sum:<10} | {g_sum:<10}")


def run_eval(n):
    results = load_existing()
    done = {r["scenario_id"] for r in results}
    scenarios = load_scenarios()[:n]

    for i, s in enumerate(scenarios):
        sid = s["scenario_id"]
        if sid in done:
            print(f"[{i+1}/{n}] Skipping (already done): {sid}")
            continue
        print(f"[{i+1}/{n}] Running graph: {sid}")
        query = f"Case {i+1}: investigate the failed pipeline run"
        logs_input = {"failure_logs": s["failure_logs"], "success_logs": s["success_logs"]}
        state = run_incident(logs_input, query)

        gt = load_ground_truth(sid)
        dims = gt["evaluation_dimensions"]
        judge_prompt = JUDGE_PROMPT.format(
            prediction=state.get("rca_draft", ""),
            gt_system=dims["system_identification"]["correct_answer"],
            gt_error=dims["error_classification"]["correct_answer"],
            gt_fix=dims["fix_correctness"]["correct_answer"],
        )
        print("  Judging...")
        judge_text = call_llm(judge_prompt)
        scores = parse_judge_scores(judge_text)

        row = {
            "scenario_id": sid,
            "system_identification": scores["system_identification"],
            "error_classification": scores["error_classification"],
            "fix_correctness": scores["fix_correctness"],
            "grader_result": state.get("grader_result", ""),
            "retry_count": state.get("retry_count", 0),
            "model": LLM_MODEL,
        }
        results.append(row)
        done.add(sid)
        save(results)
        print(f"  Scores: sys={row['system_identification']}, err={row['error_classification']}, fix={row['fix_correctness']} (grader: {row['grader_result']}, retries: {row['retry_count']})")
        time.sleep(3)

    print(f"\nResults saved to {OUT}")
    print("Graph Totals:", {d: sum(int(r[d]) for r in results) for d in DIMS}, "of", len(results))
    print_comparison(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()
    run_eval(args.n)

