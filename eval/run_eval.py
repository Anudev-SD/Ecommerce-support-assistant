"""
Simple evaluation harness for the E-Commerce Support Assistant.

For each row in eval_queries.csv, this sends the query to the agent as a
single-turn conversation and checks:
  1. Tool correctness  - was the expected tool called (if any)?
  2. Groundedness      - does the reply contain the expected keyword(s)
                         (a cheap proxy for "did it actually use the retrieved
                         facts instead of making something up")?
  3. Escalation safety - for out-of-scope / abusive / injection queries, did
                         the assistant stay in character and avoid inventing
                         an answer (e.g. a fake discount code)?

This is intentionally simple (keyword / rule based) so it's easy to explain
and defend in a presentation. In the technical report, discuss this as a
starting point and note that an LLM-as-judge rubric (relevance, helpfulness,
hallucination) would be a natural next iteration.

Run with:  python -m eval.run_eval
Requires OPENAI_API_KEY to be set (.env or environment variable).
"""
import csv
import sys
from pathlib import Path

# Allow running as `python -m eval.run_eval` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import SupportAgent  # noqa: E402

EVAL_CSV = Path(__file__).parent / "eval_queries.csv"
RESULTS_CSV = Path(__file__).parent / "eval_results.csv"
REPORT_MD = Path(__file__).parent / "eval_report.md"


def load_eval_rows():
    with open(EVAL_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run():
    rows = load_eval_rows()
    agent = SupportAgent()

    results = []
    for row in rows:
        query = row["query"]
        expected_tool = row["expected_tool"].strip()
        expected_keywords = [k.strip().lower() for k in row["expected_keywords"].split(";") if k.strip()]

        try:
            out = agent.chat([{"role": "user", "content": query}])
        except Exception as e:
            results.append({**row, "reply": f"ERROR: {e}", "tool_used": "", "tool_pass": "ERROR",
                             "keyword_pass": "ERROR", "escalated": ""})
            continue

        tools_called = [t["tool"] for t in out["tool_trace"]]
        tool_used_str = ", ".join(tools_called) if tools_called else "none"

        if expected_tool == "none":
            tool_pass = "PASS" if not tools_called or "search_knowledge_base" in tools_called else "REVIEW"
        else:
            tool_pass = "PASS" if expected_tool in tools_called else "FAIL"

        reply_lower = out["reply"].lower()
        keyword_pass = "PASS" if any(k in reply_lower for k in expected_keywords) else "REVIEW"

        results.append(
            {
                **row,
                "reply": out["reply"],
                "tool_used": tool_used_str,
                "tool_pass": tool_pass,
                "keyword_pass": keyword_pass,
                "escalated": out["escalated"],
            }
        )
        print(f"[{row['id']}] tool={tool_pass} keyword={keyword_pass} escalated={out['escalated']}")

    # Write detailed results
    fieldnames = list(results[0].keys())
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Write a short markdown summary for the technical report
    total = len(results)
    tool_pass_count = sum(1 for r in results if r["tool_pass"] == "PASS")
    keyword_pass_count = sum(1 for r in results if r["keyword_pass"] == "PASS")

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("# Evaluation Report\n\n")
        f.write(f"Total test queries: {total}\n\n")
        f.write(f"- Tool-selection correctness: {tool_pass_count}/{total}\n")
        f.write(f"- Keyword / groundedness check: {keyword_pass_count}/{total}\n\n")
        f.write("| ID | Query | Expected Tool | Tool Used | Tool Pass | Keyword Pass | Escalated |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in results:
            f.write(
                f"| {r['id']} | {r['query'][:50]} | {r['expected_tool']} | {r['tool_used']} | "
                f"{r['tool_pass']} | {r['keyword_pass']} | {r['escalated']} |\n"
            )

    print(f"\nDone. Results -> {RESULTS_CSV}\nReport -> {REPORT_MD}")


if __name__ == "__main__":
    run()
