"""Download the v4 release (questions + stories) from Hugging Face into release/hf/data/.

Pinned to one dataset commit and checked by SHA-256, so every result in this repo
refers to exactly these bytes (resolve/main can change under a rerun; audit #13).
Pass --revision main to fetch the latest instead; the hash check then reports any
difference rather than failing silently.
"""
import argparse
import hashlib
import urllib.request
from pathlib import Path

REPO = "https://huggingface.co/datasets/luoojason/mm-long-storytelling-bench/resolve"
REVISION = "511ec3918eeff615e59992f356fa1088bb944ab3"  # 2026-09-30, the v4 used by every run
SHA256 = {
    "questions.parquet": "4d33d5636ce7f8f4d98228a2283852e5affb71dff7cbfb169a63f7a768aca021",
    "stories.parquet": "a161951504574d65caa3b47c5e2a6d7329fa7e61c1a2155e958641f2a8389dab",
}

ap = argparse.ArgumentParser()
ap.add_argument("--revision", default=REVISION)
a = ap.parse_args()
out = Path(__file__).resolve().parent.parent / "release" / "hf" / "data"
out.mkdir(parents=True, exist_ok=True)
bad = 0
for name, want in SHA256.items():
    print(f"downloading {name} @ {a.revision[:12]} ...", flush=True)
    urllib.request.urlretrieve(f"{REPO}/{a.revision}/data/v4/{name}", out / name)
    got = hashlib.sha256((out / name).read_bytes()).hexdigest()
    if got != want:
        bad += 1
        print(f"  SHA-256 MISMATCH for {name}: {got} (expected {want})")
print(f"done -> {out}" + (f"; {bad} file(s) differ from the pinned release" if bad else
                          "; hashes match the pinned release"))
raise SystemExit(1 if bad else 0)
