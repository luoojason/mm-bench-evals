"""Download the v4 release (questions + stories) from Hugging Face into release/hf/data/."""
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/luoojason/mm-long-storytelling-bench/resolve/main/data/v4/"
out = Path(__file__).resolve().parent.parent / "release" / "hf" / "data"
out.mkdir(parents=True, exist_ok=True)
for name in ("questions.parquet", "stories.parquet"):
    print(f"downloading {name} ...", flush=True)
    urllib.request.urlretrieve(BASE + name, out / name)
print(f"done -> {out}")
