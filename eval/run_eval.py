"""
eval/run_eval.py
-----------------
Quality evaluation for the MoSPI RAG chatbot.

Stretch goal (Section 8):
  "Quality eval: a tiny Q&A set vs. your chatbot with basic accuracy metrics."

How it works:
  1. Load Q&A pairs from eval/qa_pairs.json
  2. Send each question to POST /ask
  3. Check if expected answer appears in chatbot response
  4. Report accuracy % and save full results to eval/results/eval_report.json

Run:
    python -m eval.run_eval
    make eval
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import requests

# ── Config ────────────────────────────────────────────────────────────────────

API_URL       = "http://localhost:8000"
QA_PATH       = Path("eval/qa_pairs.json")
RESULTS_DIR   = Path("eval/results")
REPORT_PATH   = RESULTS_DIR / "eval_report.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_qa_pairs() -> List[dict]:
    """Load question-answer pairs from qa_pairs.json."""
    if not QA_PATH.exists():
        print(f"ERROR: {QA_PATH} not found.")
        sys.exit(1)
    with open(QA_PATH, encoding="utf-8") as f:
        return json.load(f)


def check_answer(response: str, expected: str) -> bool:
    """
    Simple keyword check — does the expected answer appear
    anywhere in the chatbot's response (case-insensitive)?

    For multi-word expected answers, checks if ALL keywords appear.
    """
    response_lower = response.lower()
    keywords = expected.lower().split()
    return all(kw in response_lower for kw in keywords)


def ask_question(question: str, k: int = 5) -> dict:
    """
    Send one question to the /ask endpoint.
    Returns the full response dict or an error dict.
    """
    try:
        resp = requests.post(
            f"{API_URL}/ask",
            json={"question": question, "k": k, "temperature": 0.0},
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            return {
                "answer": "",
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
            }
    except requests.Timeout:
        return {"answer": "", "error": "Request timed out"}
    except requests.ConnectionError:
        return {"answer": "", "error": "Cannot connect to API"}


# ── Main eval loop ────────────────────────────────────────────────────────────

def run_evaluation() -> dict:
    """
    Run all Q&A pairs through the chatbot and compute accuracy.

    Returns a report dict with per-question results and summary stats.
    """
    qa_pairs = load_qa_pairs()
    results  = []
    correct  = 0

    print(f"\n── MoSPI Chatbot Quality Evaluation ───────────────")
    print(f"  Questions : {len(qa_pairs)}")
    print(f"  API       : {API_URL}")
    print(f"────────────────────────────────────────────────────\n")

    for i, pair in enumerate(qa_pairs, start=1):
        question = pair["question"]
        expected = pair["expected"]
        source   = pair.get("source", "")

        print(f"[{i:2d}/{len(qa_pairs)}] {question[:65]}")

        start    = time.monotonic()
        response = ask_question(question)
        elapsed  = time.monotonic() - start

        answer  = response.get("answer", "")
        error   = response.get("error", "")
        passed  = check_answer(answer, expected) if not error else False

        if passed:
            correct += 1
            status = "✅ PASS"
        elif error:
            status = f"❌ ERROR: {error}"
        else:
            status = "❌ FAIL"

        print(f"         {status}  ({elapsed:.1f}s)")
        if not passed and not error:
            print(f"         Expected : '{expected}'")
            print(f"         Got      : '{answer[:120]}...'")
        print()

        results.append({
            "question":       question,
            "expected":       expected,
            "source":         source,
            "answer":         answer,
            "passed":         passed,
            "error":          error,
            "response_time_s": round(elapsed, 2),
        })

    # ── Summary ───────────────────────────────────────────────────
    total    = len(qa_pairs)
    accuracy = correct / total * 100 if total > 0 else 0.0

    report = {
        "generated_at":    datetime.utcnow().isoformat(),
        "total_questions": total,
        "correct":         correct,
        "failed":          total - correct,
        "accuracy":        f"{accuracy:.1f}%",
        "api_url":         API_URL,
        "results":         results,
    }

    # ── Save report ───────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"── Evaluation Summary ───────────────────────────────")
    print(f"  Total     : {total}")
    print(f"  Correct   : {correct}")
    print(f"  Failed    : {total - correct}")
    print(f"  Accuracy  : {accuracy:.1f}%")
    print(f"  Report    : {REPORT_PATH}")
    print(f"────────────────────────────────────────────────────\n")

    return report


if __name__ == "__main__":
    run_evaluation()
