"""Write the compact eval's per-context summaries ahead of a running `evalsuite.py run`.

Each summary takes ~8-10 min on GPT-6.1 Sol, and `run` makes them one context at a
time. This fills the same summaries/ folder from the END of the context order, several
at once, so the main run finds them already on disk. Same model, prompt and budget as
`run` (it calls evalsuite.get_summary), so a prefetched summary is identical in kind.

    python scripts/prefetch_summaries.py --model chatgpt/gpt-6.1-sol \
        --base-url http://localhost:4000/v1 --key-env "" --reasoning-effort medium --workers 3
"""
import argparse
import collections
import concurrent.futures as cf
import sys
import threading
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import evalsuite as es  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--reasoning-effort")
    ap.add_argument("--summary-tokens", type=int, default=32000)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--start", choices=["end", "middle"], default="end",
                    help="end: walk back from the last context; middle: walk forward from the "
                         "midpoint, for a second prefetch alongside the first")
    a = ap.parse_args()
    a.max_tokens, a.extra, a.plain_content = 16000, None, False

    qs, stories = es.load_data()
    keys = sorted({es.ctx_key(q["story_ids"]) for q in qs}, reverse=True)
    if a.start == "middle":
        keys = keys[::-1]
        keys = keys[len(keys) // 2:] + keys[:len(keys) // 2]
    outdir = es.OUT / "compact" / es.slug(a.model) / "summaries"
    outdir.mkdir(parents=True, exist_ok=True)
    todo = [k for k in keys if not (outdir / f"{k}.json").exists()]
    print(f"{len(keys)} contexts, {len(todo)} without a summary; {a.workers} workers", flush=True)

    env = {"summaries": outdir, "summary_tokens": a.summary_tokens,
           "locks": collections.defaultdict(threading.Lock)}
    model = es.OpenAICompat(a)
    stats, lock = collections.Counter(), threading.Lock()

    def one(key):
        if (outdir / f"{key}.json").exists():  # the main run got there first
            status = "skip"
        else:
            t0 = time.time()
            try:
                es.get_summary(key, es.pack(key.split("+"), stories), model, env)
                status = "ok"
            except Exception as e:
                status = "error"
                print(f"  {key}: {type(e).__name__}: {str(e)[:200]}", flush=True)
        with lock:
            stats[status] += 1
            n = sum(stats.values())
            if n % 5 == 0 or status == "error":
                print(f"  [{n}/{len(todo)}] {dict(stats)}", flush=True)

    with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(one, todo))
    print(f"done: {dict(stats)}", flush=True)


if __name__ == "__main__":
    main()
