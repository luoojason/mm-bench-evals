# MM Long Storytelling Bench: evaluation harness

Runs the benchmark's evaluations on the released dataset
([luoojason/mm-long-storytelling-bench](https://huggingface.co/datasets/luoojason/mm-long-storytelling-bench),
v4: 1,749 questions over 146 public-domain novels) and compiles the results.

## The evals

| | eval | what it measures |
|---|---|---|
| a | `zeroshot` | Full book(s) in context; answer directly. |
| b | `compact` | The model summarizes the book once, then answers every question from its own summary. Also reports **evidence survival**: how much of each gold quote is still in the summary, which separates "the summary lost it" from "the model misread it". |
| c | `halluc` | A question about book A asked with a different book B in context. Correct = abstain (UNKNOWN). 300-question sample. |
| d | `tools` | No book in the prompt; the book is a file and the model searches it with shell-style tools under a turn cap. |

Agreed settings (2026-10-03): evals a-d, **one generation**, **medium reasoning effort**.

## How the run is organised (caching)

- `--eval all` runs every eval for one book together. The first call on a book runs
  alone and writes the prompt cache; the rest of that book's calls (zeroshot, halluc
  questions whose distractor is that book) read it. Tools calls carry no book and run last.
- The book is sent as its own content block with a cache marker, the question after it,
  so the cached prefix is byte-identical across calls.
- Compaction summaries are made once per book and shared by its questions.
- Stopped runs resume: finished questions are skipped.
- A failed call is recorded as `ERROR` and never scored as wrong. `--retry-errors` reruns them.

## Setup

```
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Linux/macOS: .venv/bin/pip
.venv/Scripts/python scripts/download_data.py      # ~28 MB into release/hf/data/
.venv/Scripts/python scripts/evalsuite.py selftest # fake model, all four evals, no spend
```

## Running a model

Always pilot first (`--limit 5`), check the answers and the cache, then drop `--limit`.

**Any OpenAI-compatible API (default OpenRouter, key in `OPENROUTER_API_KEY`):**

```
python scripts/evalsuite.py run --eval all --model <provider/model-id> \
    --reasoning-effort medium --halluc-sample 300 --limit 5
# --base-url / --key-env for another endpoint; --dry-run prints jobs and token counts
```

**GPT through a ChatGPT / Codex subscription (no API bill):** start the LiteLLM proxy
(setup, one-time device sign-in and caveats are in the header of
`scripts/litellm_chatgpt.yaml`), then:

```
python scripts/evalsuite.py run --eval all --model chatgpt/gpt-6.1-sol \
    --base-url http://localhost:4000/v1 --key-env "" \
    --reasoning-effort medium --halluc-sample 300
```

The device-code sign-in prints its code to the proxy's own terminal about a minute after
start; run the proxy in a normal terminal window, not a captured/background shell you
cannot see.

**Gemini through a Google plan (no API key, no tools eval):**

```
cd tools-gemini-cli && npm i && cd ..
GEMINI_CLI_HOME="$PWD/tools-gemini-cli/home" tools-gemini-cli/node_modules/.bin/gemini
#   -> "Login with Google", finish in the browser, then /quit
python scripts/evalsuite.py run --eval all --backend gemini-cli \
    --model gemini-3.8-flash-medium --workers 2 --halluc-sample 300 --limit 5
```

The CLI's own tools are denied and a neutral system prompt replaces its coding-agent
prompt. Caching on this route is whatever the plan does. Do not use the Antigravity CLI:
it truncates every message to ~45k tokens (median book is ~103k).

**Speeding up compaction (optional).** A summary takes ~8-10 min on a frontier model and
`run` makes them one book at a time. In a second terminal, alongside the run:

```
python scripts/prefetch_summaries.py --model <same id> [--base-url ... --key-env ...] \
    --reasoning-effort medium --workers 3
```

It fills the same summaries folder from the other end of the book list. A second copy
with `--start middle` works forward from the midpoint, doubling the rate.

## Timing

Most calls take seconds. The slow part is (b): each book's compaction summary takes
~10-15 min on a frontier model at medium effort (~20k-token summaries; the subscription
route cannot cap their length), and multi-book contexts take longer. The per-call limit
(`--timeout`) defaults to 1 hour for this reason; at 15 min, two-book summaries failed.
`run` works on up to `--workers` books at once (each book's first call still goes alone,
to write the cache). On the subscription route the usage limit, not the script, is
usually what sets the pace.

## Checking a pilot

- Answers: `verify/evalsuite/<eval>/<model>/results.jsonl` (`EVALSUITE_OUT=<dir>` moves it).
- Cache: the 2nd+ question on the same book should show most of the book under
  `usage.prompt_tokens_details.cached_tokens`. If it is ~0, fix that before the full run.
- Thinking budget: if answers come back empty with `finish: length`, the model spent the
  output budget reasoning; raise `--max-tokens` (default 16,000) and `--retry-errors`.
- Compaction: `score --eval compact` warns when summaries were cut off at `--summary-tokens`.

## Scoring and compiling

```
python scripts/evalsuite.py score --eval zeroshot --model <id>   # one eval, one model
python scripts/compile_results.py                                # everything, one table
```

`compile_results.py` writes `compiled.md` and `compiled.csv`: accuracy per model x eval x
question type over all questions (every question is verified). Halluc accuracy = share
abstained. ERROR rows are counted but not scored; UNCLEAR verdicts are scored and listed.

## Files

| file | role |
|---|---|
| `scripts/evalsuite.py` | run / score / selftest for the four evals |
| `scripts/gcheck.py` | the answer adjudicator (`evalsuite` imports it) |
| `scripts/compile_results.py` | one table across all results |
| `scripts/prefetch_summaries.py` | parallel compaction summaries |
| `scripts/download_data.py` | fetch the v4 data from Hugging Face |
| `scripts/litellm_chatgpt.yaml` | proxy config for the subscription GPT route |
| `tools-gemini-cli/package.json` | pinned Gemini CLI for the Google-plan route |
