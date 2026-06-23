"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness.

    Calls the agent, extracts the SQL produced at each iteration from history,
    runs each against the DB, and compares to gold SQL execution result.

    Returns:
        {
            "question": str,
            "db_id": str,
            "gold_sql": str,
            "agent_error": str | None,   # HTTP / agent-level error
            "iterations": int,           # how many iterations the agent took
            "final_sql": str | None,
            "final_correct": bool,
            "iter_results": [            # one entry per iteration (0-indexed)
                {"iter": int, "sql": str, "correct": bool}
            ],
        }
    """
    db_id = question["db_id"]
    gold_sql = question["gold_sql"]

    # Run gold SQL once up front; if it errors we cannot score anything.
    gold_ok, gold_rows, gold_err = run_sql(db_id, gold_sql)

    # Call the agent.
    try:
        resp = httpx.post(
            agent_url,
            json={"question": question["question"], "db": db_id},
            timeout=120.0,
        )
        resp.raise_for_status()
        result = resp.json()
    except Exception as e:  # noqa: BLE001
        return {
            "question": question["question"],
            "db_id": db_id,
            "gold_sql": gold_sql,
            "agent_error": str(e),
            "iterations": 0,
            "final_sql": None,
            "final_correct": False,
            "iter_results": [],
        }

    history = result.get("history", [])
    final_sql = result.get("sql")
    iterations = result.get("iterations", 1)

    # Extract the SQL emitted at each iteration (generate_sql = iter 0, each
    # revise increments the counter).
    iter_sqls: list[str] = [
        entry["sql"]
        for entry in history
        if entry.get("node") in ("generate_sql", "revise") and entry.get("sql")
    ]

    iter_results: list[dict] = []
    for i, sql in enumerate(iter_sqls):
        _, pred_rows, _ = run_sql(db_id, sql)
        correct = matches(gold_rows, pred_rows) if gold_ok else False
        iter_results.append({"iter": i, "sql": sql, "correct": correct})

    final_correct = iter_results[-1]["correct"] if iter_results else False

    return {
        "question": question["question"],
        "db_id": db_id,
        "gold_sql": gold_sql,
        "agent_error": result.get("error"),
        "iterations": iterations,
        "final_sql": final_sql,
        "final_correct": final_correct,
        "iter_results": iter_results,
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.

    Returns:
        {
            "total": int,
            "overall_accuracy": float,   # final SQL correct / total
            "iter_pass_rates": {         # pass rate if we stopped at iter N
                "iter_0": float,
                "iter_1": float,
                ...
            },
        }
    """
    total = len(results)
    if total == 0:
        return {"total": 0, "overall_accuracy": 0.0, "iter_pass_rates": {}}

    overall_correct = sum(1 for r in results if r["final_correct"])

    # Find the maximum number of iterations across all questions.
    max_iter = max(
        (r["iter_results"][-1]["iter"] for r in results if r["iter_results"]),
        default=0,
    )

    iter_pass_rates: dict[str, float] = {}
    for k in range(max_iter + 1):
        correct_at_k = 0
        for r in results:
            iter_results = r["iter_results"]
            if not iter_results:
                continue
            # Carry forward: use the last available iteration <= k.
            candidates = [e for e in iter_results if e["iter"] <= k]
            best = candidates[-1] if candidates else iter_results[-1]
            if best["correct"]:
                correct_at_k += 1
        iter_pass_rates[f"iter_{k}"] = round(correct_at_k / total, 4)

    return {
        "total": total,
        "overall_accuracy": round(overall_correct / total, 4),
        "iter_pass_rates": iter_pass_rates,
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
