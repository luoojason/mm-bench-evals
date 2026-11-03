"""Check a pilot run before the full one: answers came back, the cache is hit, and
what it cost. Reads only the results files `evalsuite.py run` wrote; sends nothing.

    python scripts/pilot_check.py --model <provider/model-id>

Cache: the first call on a book writes it, so expect about one low-cache call per
book context and most of the prompt cached on every other call. Many more low-cache
calls than contexts means the cache is not being hit; fix that before the full run.
Cost is OpenRouter's own figure per call (usage.cost), compaction calls included.
"""
import argparse
import collections
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import evalsuite as es  # noqa: E402

LOW = 0.5  # a call with less than half its prompt read from cache counts as a miss


def cached_share(u):
    p = u.get("prompt_tokens") or 0
    c = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return c / p if p else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    slug, cost, problems = es.slug(a.model), 0.0, 0
    book_calls = []  # (context, cached share) for calls that carry a book or summary
    for ev in es.EVALS:
        res = es.OUT / ev / slug / "results.jsonl"
        if not res.exists():
            continue
        rows = [json.loads(l) for l in io.open(res, encoding="utf-8")]
        st = collections.Counter(r["status"] for r in rows)
        ok = [r for r in rows if r["status"] == "ok"]
        empty = sum(1 for r in ok if not (r.get("response") or "").strip())
        length = sum(1 for r in ok if r.get("finish") == "length")
        cost += sum((r.get("usage") or {}).get("cost") or 0 for r in ok)
        print(f"{ev}: {dict(st)}; {empty} empty answers, {length} cut off at --max-tokens")
        problems += st["error"] + empty
        for r in rows[:3]:
            if r["status"] == "error":
                print(f"   error: {r['error'][:160]}")
        if ev != "tools":
            book_calls += [((ev == "compact", r["context"]), cached_share(r["usage"]))
                           for r in ok]
        if ev == "compact":
            for f in sorted((res.parent / "summaries").glob("*.json")):
                s = json.load(io.open(f, encoding="utf-8"))
                cost += (s.get("usage") or {}).get("cost") or 0
                if s.get("finish") == "length":
                    print(f"   summary {f.stem} cut off at --summary-tokens")
    known = [(k, v) for k, v in book_calls if v is not None]
    if known:
        contexts = len({k for k, _ in known})
        misses = sum(1 for _, v in known if v < LOW)
        print(f"cache: {len(known)} calls over {contexts} contexts; {misses} read under "
              f"{LOW:.0%} of the prompt from cache (expect about {contexts}, the first "
              f"call on each)")
        if misses > contexts + max(2, contexts // 5):
            print("   CACHE NOT HITTING: check before the full run")
            problems += 1
    else:
        print("cache: no usage reported, so it cannot be checked from here")
    print(f"cost reported by the endpoint: ${cost:.2f}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
