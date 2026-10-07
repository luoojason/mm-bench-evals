"""Compile every evalsuite result into one table: model x eval x question type.

Reads verify/evalsuite/<eval>/<model>/results.jsonl (or $EVALSUITE_OUT), scores each
with evalsuite's own scorer (the latest record per question wins, ERROR rows are
counted but never scored), and writes:

    <out>/compiled.csv   one row per model x eval x subset x question type
    <out>/compiled.md    the overall table plus a per-type table per eval

Accuracy = CORRECT / scored (ABSTAINED / scored for halluc). "verified" is the subset
whose second human check is done (board_status != Awaiting verification); report both.

    python scripts/compile_results.py [--out verify/evalsuite]
"""
import argparse
import collections
import csv
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import evalsuite as es  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="where to write (default: the results folder)")
    a = ap.parse_args()
    out = Path(a.out) if a.out else es.OUT
    qs, _ = es.load_data()
    by_id = {q["id"]: q for q in qs}

    rows, overall = [], {}
    for ev in es.EVALS:
        for res in sorted((es.OUT / ev).glob("*/results.jsonl")):
            model = res.parent.name
            recs = [json.loads(l) for l in io.open(res, encoding="utf-8")]
            scored = es.score_rows(ev, by_id, recs)
            good = "ABSTAINED" if ev == "halluc" else "CORRECT"
            for subset in ("all", "verified"):
                cells = collections.defaultdict(collections.Counter)
                for q, _, v, _ in scored:
                    if subset == "verified" and q["board_status"] == "Awaiting verification":
                        continue
                    cells["ALL"][v] += 1
                    cells[q["question_type"]][v] += 1
                for t, c in sorted(cells.items()):
                    n = sum(c.values()) - c["ERROR"]
                    row = {"model": model, "eval": ev, "subset": subset, "type": t,
                           "answered": n, "errors": c["ERROR"], "good": c[good],
                           "unclear": c["UNCLEAR"],
                           "accuracy": round(c[good] / n, 4) if n else ""}
                    rows.append(row)
                    if t == "ALL":
                        overall[(model, ev, subset)] = row

    if not rows:
        print(f"no results under {es.OUT}")
        return 1
    out.mkdir(parents=True, exist_ok=True)
    with io.open(out / "compiled.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def pct(r):
        return "" if r is None or r["accuracy"] == "" else f"{100 * r['accuracy']:.1f}% ({r['answered']}{', ' + str(r['errors']) + ' err' if r['errors'] else ''})"

    models = sorted({m for m, _, _ in overall})
    md = ["# Compiled results", "",
          "Accuracy on scored answers (n, errors). halluc = abstained when the book is absent.", ""]
    for subset in ("all", "verified"):
        md += [f"## {subset} questions", "", "| model | " + " | ".join(es.EVALS) + " |",
               "|" + " --- |" * (len(es.EVALS) + 1)]
        for m in models:
            md.append(f"| {m} | " + " | ".join(pct(overall.get((m, ev, subset))) for ev in es.EVALS) + " |")
        md.append("")
    for ev in es.EVALS:
        types = sorted({r["type"] for r in rows if r["eval"] == ev and r["type"] != "ALL"})
        if not types:
            continue
        md += [f"## {ev} by question type (all questions)", "",
               "| model | " + " | ".join(types) + " |", "|" + " --- |" * (len(types) + 1)]
        for m in models:
            cell = {r["type"]: r for r in rows if r["model"] == m and r["eval"] == ev and r["subset"] == "all"}
            md.append(f"| {m} | " + " | ".join(pct(cell.get(t)) for t in types) + " |")
        md.append("")
    io.open(out / "compiled.md", "w", encoding="utf-8", newline="\n").write("\n".join(md))
    print("\n".join(md[:md.index("## verified questions")]))
    print(f"-> {out / 'compiled.csv'}, {out / 'compiled.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
