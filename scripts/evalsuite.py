"""Eval suite for the released dataset: the four evals in release/EVAL_PLAN.md.

    zeroshot  (a) the whole book(s) in context, answer the question
    compact   (b) the model compacts each context once, then answers from its own
                  summary; also reports whether each question's gold evidence
                  survived the summary
    halluc    (c) a question about novel A asked with only novel B in context;
                  the right answer is UNKNOWN
    tools     (d) no book in context; the book is a file the model can search
                  with grep / read tools, under a turn cap

Reads the release parquet (release/hf/data/{questions,stories}.parquet, built by
release/build_release.py), so it evaluates exactly what is published.

Any OpenAI-compatible chat endpoint works (OpenRouter, OpenAI, a local
llama-server or vLLM). Questions run grouped by context, first question of a
context alone and the rest after it, so every provider that caches prefixes
writes each book once. Claude and Gemini through OpenRouter need an explicit
cache breakpoint, which the context block carries.

Results go to verify/evalsuite/<eval>/<model>/results.jsonl, one line per
question, appended as they finish, so a stopped run resumes where it left off.
A call that FAILED is written with status "error" and is never scored: an
infrastructure failure written into a results field reads exactly like a wrong
answer (the hub's cheat code 17). `score` counts errors separately.

    python scripts/evalsuite.py selftest
    python scripts/evalsuite.py run --eval zeroshot --model openai/gpt-6-sol \
        --base-url https://openrouter.ai/api/v1 --key-env OPENROUTER_API_KEY --limit 5
    python scripts/evalsuite.py score --eval zeroshot --model openai/gpt-6-sol

`--eval all` (or a comma list such as `zeroshot,compact,halluc`) runs the evals
together, one novel at a time: every eval that puts a given book in context
(zeroshot, the compaction call, halluc questions that use it as the distractor)
runs back to back, so the book is written to the cache once and read by all of
them. Tools calls carry no book and run last. Each eval still writes its own
results file, so `score` is unchanged.

GPT through a ChatGPT/Codex subscription: start the LiteLLM proxy
(scripts/litellm_chatgpt.yaml, notes in its header) and point `run` at it:
    --base-url http://localhost:4000/v1 --key-env "" --model chatgpt/<model>

Gemini through a Google plan (Jason's student plan): Google's Gemini CLI, installed
in tools-gemini-cli/ and signed in once (release/GEMINI_ROUTE.md; class GeminiCLI
has the caveats; no tools eval on this route):
    --backend gemini-cli --model gemini-3.8-flash-medium --workers 2

Nothing here spends money unless `run` is pointed at a paid endpoint. `--dry-run`
prints what would be sent (calls, contexts, input tokens) and stops.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import functools
import io
import json
import os
import random
import re
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from gcheck import adjudicate, norm  # noqa: E402  the project's one scorer

DATA = ROOT / "release" / "hf" / "data"
OUT = Path(os.environ.get("EVALSUITE_OUT") or ROOT / "verify" / "evalsuite")  # pilots set it
SEP = "\n\n" + "=" * 70 + "\n\n"
EVALS = ("zeroshot", "compact", "halluc", "tools")

ANSWER_RULES = """Rules:
- Answer directly and briefly. For a count, give the number. For a name or
  phrase, give it as the novel gives it. For a list, give the items in order,
  separated by semicolons.
- If the text genuinely does not settle it, reply exactly: UNKNOWN

FORMAT: your FIRST line must be exactly `ANSWER: <your answer>`.
Put any reasoning on the lines after it."""

ZEROSHOT = """You are answering a reading-comprehension question about the novel above.
Answer ONLY from the novel text provided. It is the complete work.
Do NOT write or execute code. Read the text and answer.

""" + ANSWER_RULES + """

QUESTION:
{question}"""

COMPACT_REQUEST = """Compact the text above into notes that someone could later use
INSTEAD of the text to answer detailed questions about it.

Keep everything: every character and what they are called, every event in the
order it happens, every place, object, number, date and quoted phrase that
matters, who said or did what to whom. Drop only redundancy and style. If there
is more than one book, keep them apart under their BOOK headings.

Write the notes only. Do not comment on the text or answer any question."""

FROM_SUMMARY = """The notes above are a compacted version of a novel (you wrote them
earlier from the full text, which is no longer available). Answer the question
from the notes alone.

""" + ANSWER_RULES + """

QUESTION:
{question}"""

TOOLS_SYSTEM = """You are answering a reading-comprehension question about a novel.
The novel is NOT in this conversation. It is stored as a text file you can
search with the tools provided: {files}.
Use the tools to find what you need, then answer. You have at most {turns}
tool-using turns.

""" + ANSWER_RULES

TOOLS_SPEC = [
    {"type": "function", "function": {
        "name": "grep",
        "description": "Search a book file for a regular expression (Python "
                       "syntax). Returns matching line numbers and lines, and "
                       "the total number of matching lines.",
        "parameters": {"type": "object", "properties": {
            "file": {"type": "string"},
            "pattern": {"type": "string"},
            "ignore_case": {"type": "boolean", "default": True},
            "max_results": {"type": "integer", "default": 50}},
            "required": ["file", "pattern"]}}},
    {"type": "function", "function": {
        "name": "read",
        "description": "Read lines from a book file, 1-based, at most 200 "
                       "lines per call.",
        "parameters": {"type": "object", "properties": {
            "file": {"type": "string"},
            "start_line": {"type": "integer"},
            "num_lines": {"type": "integer", "default": 100}},
            "required": ["file", "start_line"]}}},
    {"type": "function", "function": {
        "name": "wc",
        "description": "Line, word and character count of a book file.",
        "parameters": {"type": "object", "properties": {
            "file": {"type": "string"}}, "required": ["file"]}}},
]


# ---- data -------------------------------------------------------------------

def load_data():
    qs = pq.read_table(DATA / "questions.parquet").to_pylist()
    stories = {s["story_id"]: s for s in pq.read_table(DATA / "stories.parquet").to_pylist()}
    return qs, stories


def pack(sids, stories):
    """The same packing as build_release.py and gcheck.py."""
    books = [stories[s]["text"] for s in sids]
    if len(books) == 1:
        return books[0]
    return SEP.join(f"### BOOK {n} of {len(books)}\n\n{b}" for n, b in enumerate(books, 1))


def ctx_key(sids):
    return "+".join(sids)


def select(qs, a):
    if a.ids:
        want = {i.strip() for i in a.ids.split(",")}
        qs = [q for q in qs if q["id"] in want]
    if a.types:
        want = {t.strip() for t in a.types.split(",")}
        qs = [q for q in qs if q["question_type"] in want]
    if a.reviewed_only:
        qs = [q for q in qs if q["board_status"] != "Awaiting verification"]
    if a.max_input_tokens:
        qs = [q for q in qs if q["input_tokens"] <= a.max_input_tokens]
    if a.sample:
        rnd = random.Random(a.seed)
        qs = rnd.sample(qs, min(a.sample, len(qs)))
    qs = sorted(qs, key=lambda q: (ctx_key(q["story_ids"]), q["id"]))
    if a.limit:
        qs = qs[:a.limit]
    return qs


# ---- model backends ---------------------------------------------------------

class CallError(Exception):
    pass


class OpenAICompat:
    """POST /chat/completions on any OpenAI-compatible endpoint."""

    def __init__(self, a):
        import requests
        self.s = requests.Session()
        self.url = a.base_url.rstrip("/") + "/chat/completions"
        key = os.environ.get(a.key_env, "") if a.key_env else ""
        self.headers = {"Content-Type": "application/json"}
        if key:
            self.headers["Authorization"] = f"Bearer {key}"
        self.model, self.max_tokens, self.timeout = a.model, a.max_tokens, a.timeout
        self.extra = json.loads(a.extra) if a.extra else {}
        effort = getattr(a, "reasoning_effort", None)
        if effort:  # OpenRouter takes a reasoning object; OpenAI-style APIs a flat field
            if "openrouter.ai" in self.url:
                self.extra.setdefault("reasoning", {"effort": effort})
            else:
                self.extra.setdefault("reasoning_effort", effort)
        self.plain = a.plain_content

    def _flatten(self, messages):
        if not self.plain:
            return messages
        out = []
        for m in messages:
            c = m.get("content")
            if isinstance(c, list):
                m = dict(m, content="\n\n".join(p["text"] for p in c))
            out.append(m)
        return out

    def chat(self, messages, tools=None, max_tokens=None):
        body = {"model": self.model, "messages": self._flatten(messages),
                "max_tokens": max_tokens or self.max_tokens, **self.extra}
        if tools:
            body["tools"] = tools
        last = None
        for attempt in range(4):
            try:
                r = self.s.post(self.url, headers=self.headers, json=body,
                                timeout=self.timeout)
            except Exception as e:  # network
                last = f"{type(e).__name__}: {e}"
            else:
                if r.status_code == 200:
                    j = r.json()
                    if j.get("error"):
                        last = f"provider error: {str(j['error'])[:300]}"
                    elif not j.get("choices"):
                        last = f"no choices: {str(j)[:300]}"
                    else:
                        msg = j["choices"][0]["message"]
                        return {"content": msg.get("content") or "",
                                "tool_calls": msg.get("tool_calls") or [],
                                "finish": j["choices"][0].get("finish_reason"),
                                "usage": j.get("usage") or {}}
                else:
                    last = f"HTTP {r.status_code}: {r.text[:300]}"
                    if r.status_code in (400, 401, 403, 404):
                        break  # not transient
            time.sleep(2 ** attempt * 3)
        raise CallError(last)


class GeminiCLI:
    """Gemini through Google's Gemini CLI signed in with a Google account, so
    calls draw on that account's plan (Jason's student plan) instead of an API key.

    Everything the CLI keeps lives under tools-gemini-cli/home (GEMINI_CLI_HOME):
    the sign-in, the settings below, and session files, which are deleted after
    each call (a book is up to 3 MB and ~3,500 calls would fill the disk). The
    settings deny every tool (an allowlist naming no real tool), and a neutral
    system prompt replaces the coding-agent one, so the model answers from the
    prompt alone. Thinking level is set by a model alias the settings define:
    --model gemini-3.8-flash-medium. Stdin carries the prompt (8 MB cap, longest
    book ~3.3 MB). No function calling, so the tools eval cannot run here, and
    no output cap. Sign in once: see release/GEMINI_ROUTE.md."""

    DIR = ROOT / "tools-gemini-cli"
    SYSTEM = ("You are a careful reader answering questions about a text. Answer "
              "from the text given in the message.\n")

    def __init__(self, a):
        self.model, self.timeout = a.model, a.timeout
        self.exe = a.gemini or str(self.DIR / "node_modules" / ".bin" / (
            "gemini.cmd" if os.name == "nt" else "gemini"))
        home = self.DIR / "home"
        (home / ".gemini").mkdir(parents=True, exist_ok=True)
        self.work = self.DIR / "work"
        self.work.mkdir(exist_ok=True)
        sysmd = self.DIR / "system.md"
        io.open(sysmd, "w", encoding="utf-8", newline="\n").write(self.SYSTEM)
        cfg = home / ".gemini" / "settings.json"
        s = json.load(io.open(cfg, encoding="utf-8")) if cfg.exists() else {}
        s.setdefault("security", {}).setdefault("auth", {})["selectedType"] = "oauth-personal"
        s["tools"] = {"core": ["no_tools_for_this_benchmark"]}
        aliases = {}
        for level in ("low", "medium", "high"):
            aliases[f"gemini-3.8-flash-{level}"] = {
                "extends": "chat-base-3", "modelConfig": {
                    "model": "gemini-3.8-flash", "generateContentConfig": {
                        "thinkingConfig": {"thinkingLevel": level.upper()}}}}
        s["modelConfigs"] = {"customAliases": aliases}
        io.open(cfg, "w", encoding="utf-8", newline="\n").write(json.dumps(s, indent=1))
        self.home = home
        self.env = dict(os.environ, GEMINI_CLI_HOME=str(home), GEMINI_SYSTEM_MD=str(sysmd),
                        GEMINI_CLI_TRUST_WORKSPACE="true", NO_COLOR="1")
        self.env.pop("GEMINI_API_KEY", None)  # never fall back to a billed key

    def _purge_sessions(self):
        for d in (self.home / ".gemini" / "tmp").glob("*/chats"):
            shutil.rmtree(d, ignore_errors=True)

    def chat(self, messages, tools=None, max_tokens=None):
        import subprocess
        if tools:
            raise CallError("gemini-cli route has no function calling (tools eval)")
        parts = []
        for m in messages:
            c = m.get("content")
            parts.append("\n\n".join(p["text"] for p in c) if isinstance(c, list) else c)
        cmd = [self.exe, "--model", self.model, "--output-format", "json",
               "--skip-trust", "-p", "Answer the question above."]
        last = None
        for attempt in range(3):
            try:
                p = subprocess.run(cmd, input="\n\n".join(parts).encode("utf-8"),
                                   cwd=self.work, env=self.env, capture_output=True,
                                   timeout=self.timeout, shell=False)
            except subprocess.TimeoutExpired:
                last = f"gemini-cli timed out after {self.timeout}s"
                continue
            finally:
                self._purge_sessions()
            out = p.stdout.decode("utf-8", "replace")
            j = None
            try:
                j = json.loads(out[out.index("{"):])
            except ValueError:
                pass
            if j and j.get("response") is not None and not j.get("error"):
                tok = collections.Counter()
                for mstats in ((j.get("stats") or {}).get("models") or {}).values():
                    tok.update({k: v for k, v in (mstats.get("tokens") or {}).items()
                                if isinstance(v, (int, float))})
                calls = ((j.get("stats") or {}).get("tools") or {}).get("totalCalls", 0)
                return {"content": j["response"], "tool_calls": [], "finish": "stop",
                        "usage": {"prompt_tokens": tok.get("prompt", tok.get("input", 0)),
                                  "completion_tokens": tok.get("candidates", 0),
                                  "thinking_tokens": tok.get("thoughts", 0),
                                  "cached_tokens": tok.get("cached", 0),
                                  "cli_tool_calls": calls}}
            err = (j or {}).get("error") or out[-300:] or p.stderr.decode("utf-8", "replace")[-300:]
            last = f"gemini-cli (exit {p.returncode}): {str(err)[:400]}"
            if "auth" in last.lower() or "login" in last.lower():
                break  # not transient: sign in first
            time.sleep(2 ** attempt * 10)
        raise CallError(last)


class Fake:
    """Scripted model for `selftest`. `fn(messages, tools)` returns a reply dict."""

    def __init__(self, fn):
        self.fn, self.calls = fn, []
        self.lock = threading.Lock()

    def chat(self, messages, tools=None, max_tokens=None):
        with self.lock:
            self.calls.append(messages)
        return self.fn(messages, tools)


# ---- the four evals ---------------------------------------------------------

def cached_user(context, tail):
    """Context as its own block with a cache breakpoint, then the variable part."""
    return {"role": "user", "content": [
        {"type": "text", "text": context, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": SEP + tail}]}


def add_usage(tot, u):
    for k, v in (u or {}).items():
        if isinstance(v, (int, float)):
            tot[k] = tot.get(k, 0) + v
        elif isinstance(v, dict):  # prompt_tokens_details etc.
            add_usage(tot.setdefault(k, {}), v)
    return tot


def run_zeroshot(q, ctx, model, env):
    r = model.chat([cached_user(ctx, ZEROSHOT.format(question=q["question"]))])
    return {"response": r["content"], "usage": r["usage"], "finish": r["finish"]}


def evidence_survival(quotes, summary):
    """Per gold quote, the share of its content words (len > 3) found in the
    summary. A summary paraphrases, so this is a recall proxy, not a match: it
    separates "the summary dropped the fact" from "the model misread it"."""
    sw = set(norm(summary).split())
    out = []
    for qt in quotes or []:
        words = [w for w in norm(qt).split() if len(w) > 3]
        if words:
            out.append(round(sum(w in sw for w in words) / len(words), 3))
    return out


def get_summary(key, ctx, model, env):
    """One compaction per context, cached on disk and shared by its questions."""
    path = env["summaries"] / f"{key}.json"
    with env["locks"][key]:
        if path.exists():
            return json.load(io.open(path, encoding="utf-8"))
        r = model.chat([cached_user(ctx, COMPACT_REQUEST)],
                       max_tokens=env["summary_tokens"])
        if not r["content"].strip():
            raise CallError(f"empty summary (finish={r['finish']})")
        rec = {"summary": r["content"], "usage": r["usage"], "finish": r["finish"]}
        io.open(path, "w", encoding="utf-8", newline="\n").write(
            json.dumps(rec, ensure_ascii=False))
        return rec


def run_compact(q, ctx, model, env):
    s = get_summary(ctx_key(q["story_ids"]), ctx, model, env)
    r = model.chat([cached_user(s["summary"], FROM_SUMMARY.format(question=q["question"]))])
    return {"response": r["content"], "usage": r["usage"], "finish": r["finish"],
            "summary_tokens_est": len(s["summary"]) // 4,
            "summary_finish": s["finish"],
            "evidence_survival": evidence_survival(q["supporting_quotes"], s["summary"])}


def pick_distractors(qs, stories, seed):
    """For each single-novel gold story A, one other novel B: different author
    (so no shared series or characters), closest in length, spread so that no
    B serves too many As. Deterministic."""
    golds = sorted({q["story_ids"][0] for q in qs if len(q["story_ids"]) == 1})
    pool = sorted(stories.values(), key=lambda s: s["tokens"])
    used = collections.Counter()
    rnd = random.Random(seed)
    out = {}
    for a in golds:
        A = stories[a]
        cands = [s for s in pool if s["author"] != A["author"]]
        cands.sort(key=lambda s: (used[s["story_id"]], abs(s["tokens"] - A["tokens"]),
                                  rnd.random()))
        out[a] = cands[0]["story_id"]
        used[out[a]] += 1
    return out


def halluc_eligible(q, b_norm):
    """Skip a question whose gold answer happens to occur in B: then B might
    genuinely answer it and UNKNOWN is not the only right reply. `b_norm` is
    norm(B's text), computed once per B."""
    if len(q["story_ids"]) != 1:
        return False
    g = norm(q["expected_output"])
    return not (len(g) > 3 and re.search(r"\b" + re.escape(g) + r"\b", b_norm))


def run_halluc(q, ctx, model, env):
    r = model.chat([cached_user(ctx, ZEROSHOT.format(question=q["question"]))])
    return {"response": r["content"], "usage": r["usage"], "finish": r["finish"],
            "distractor": env["distractors"][q["story_ids"][0]]}


def tool_exec(call, files):
    """Run one tool call against the book files. Pure Python, read-only, no shell."""
    name = call["function"]["name"]
    try:
        args = json.loads(call["function"].get("arguments") or "{}")
    except json.JSONDecodeError as e:
        return f"error: arguments are not valid JSON ({e})"
    f = args.get("file")
    if f not in files:
        return f"error: unknown file {f!r}; files are {sorted(files)}"
    lines = files[f]
    if name == "wc":
        text = "\n".join(lines)
        return f"{len(lines)} lines, {len(text.split())} words, {len(text)} chars"
    if name == "grep":
        try:
            rx = re.compile(args["pattern"], re.I if args.get("ignore_case", True) else 0)
        except (re.error, KeyError) as e:
            return f"error: bad pattern ({e})"
        hits = [(i, ln) for i, ln in enumerate(lines, 1) if rx.search(ln)]
        cap = max(1, min(int(args.get("max_results") or 50), 200))
        body = "\n".join(f"{i}: {ln[:400]}" for i, ln in hits[:cap])
        more = f"\n... {len(hits) - cap} more" if len(hits) > cap else ""
        return f"{len(hits)} matching lines\n{body}{more}"[:30000]
    if name == "read":
        s = max(1, int(args.get("start_line") or 1))
        n = max(1, min(int(args.get("num_lines") or 100), 200))
        return "\n".join(f"{i}: {ln}" for i, ln in
                         enumerate(lines[s - 1:s - 1 + n], s))[:30000] or "(past end of file)"
    return f"error: unknown tool {name!r}"


def run_tools(q, ctx, model, env):
    stories = env["stories"]
    names = ([f"book_{n}.txt" for n in range(1, len(q["story_ids"]) + 1)]
             if len(q["story_ids"]) > 1 else ["book.txt"])
    files = {nm: stories[s]["text"].splitlines() for nm, s in zip(names, q["story_ids"])}
    msgs = [{"role": "system", "content": TOOLS_SYSTEM.format(
                files=", ".join(names), turns=env["max_turns"])},
            {"role": "user", "content": "QUESTION:\n" + q["question"]}]
    usage, trace = {}, []
    for turn in range(env["max_turns"] + 1):
        tools = TOOLS_SPEC if turn < env["max_turns"] else None  # last turn: must answer
        r = model.chat(msgs, tools=tools)
        add_usage(usage, r["usage"])
        if not r["tool_calls"]:
            return {"response": r["content"], "usage": usage, "finish": r["finish"],
                    "turns": turn, "tool_calls": trace}
        msgs.append({"role": "assistant", "content": r["content"] or None,
                     "tool_calls": r["tool_calls"]})
        for c in r["tool_calls"]:
            out = tool_exec(c, files)
            trace.append({"tool": c["function"]["name"],
                          "args": c["function"].get("arguments"), "out_chars": len(out)})
            msgs.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": out})
        if turn == env["max_turns"] - 1:
            msgs.append({"role": "user", "content": "Tool budget used up. Answer now."})
    return {"response": "", "usage": usage, "finish": "turn_cap",
            "turns": env["max_turns"], "tool_calls": trace}


RUNNERS = {"zeroshot": run_zeroshot, "compact": run_compact,
           "halluc": run_halluc, "tools": run_tools}


# ---- run --------------------------------------------------------------------

def slug(model):
    return re.sub(r"[^\w.-]+", "_", model)


def prepare_jobs(ev, qs, stories, a):
    """(question, context key, input tokens) per job, in cache-friendly order,
    plus env. Context text is built when its context runs (`env["context"]`),
    never up front: packed per question, the full set is ~1 GB of strings."""
    env = {"stories": stories, "max_turns": a.max_turns,
           "summary_tokens": a.summary_tokens,
           "locks": collections.defaultdict(threading.Lock)}
    jobs, skipped = [], 0
    if ev == "halluc":
        env["distractors"] = pick_distractors(qs, stories, a.seed)
        normed = {}
        for q in qs:
            b = env["distractors"].get(q["story_ids"][0]) if len(q["story_ids"]) == 1 else None
            if b and b not in normed:
                normed[b] = norm(stories[b]["text"])
            if b and halluc_eligible(q, normed[b]):
                jobs.append((q, b, stories[b]["tokens"]))
            else:
                skipped += 1
        jobs.sort(key=lambda j: (j[1], j[0]["id"]))
        n = getattr(a, "halluc_sample", None)
        if n and n < len(jobs):  # same draw as release/budget_v2.py at seed 0
            jobs = sorted(random.Random(a.seed).sample(jobs, n),
                          key=lambda j: (j[1], j[0]["id"]))
        # A single book packs to its bare text (pack()), so this context is
        # byte-identical to that book's zeroshot context and shares its cache.
        env["context"] = lambda key: stories[key]["text"]
    else:
        for q in qs:
            jobs.append((q, ctx_key(q["story_ids"]), q["input_tokens"]))
        env["context"] = functools.lru_cache(maxsize=16)(
            lambda key: pack(key.split("+"), stories))
    return jobs, env, skipped


def parse_evals(s):
    evs = list(EVALS) if s == "all" else [e.strip() for e in s.split(",") if e.strip()]
    bad = [e for e in evs if e not in EVALS]
    if bad or not evs:
        raise SystemExit(f"--eval: unknown {bad}; choose from {', '.join(EVALS)} or all")
    return evs


# Within one book's group, the eval that runs first writes the cache. zeroshot
# and halluc send the book itself; compact's first call is the compaction of it.
FIRST = {"zeroshot": 0, "halluc": 1, "compact": 2}


def cmd_run(a, model=None):
    qs, stories = load_data()
    qs = select(qs, a)
    evs = parse_evals(a.eval)
    if getattr(a, "backend", "openai") == "gemini-cli" and "tools" in evs:
        print("gemini-cli route: skipping tools (needs function calling; run it on the API)")
        evs = [e for e in evs if e != "tools"]
    st, units = {}, []
    for ev in evs:
        jobs, env, skipped = prepare_jobs(ev, qs, stories, a)
        outdir = OUT / ev / slug(a.model)
        outdir.mkdir(parents=True, exist_ok=True)
        env["summaries"] = outdir / "summaries"
        env["summaries"].mkdir(exist_ok=True)
        res = outdir / "results.jsonl"
        done = set()
        if res.exists():
            for line in io.open(res, encoding="utf-8"):
                r = json.loads(line)
                if r["status"] == "ok" or not a.retry_errors:
                    done.add(r["id"])
        todo = [j for j in jobs if j[0]["id"] not in done]
        st[ev] = {"env": env, "res": res}
        units += [(ev, j) for j in todo]
        in_tok = sum(j[2] for j in todo) if ev != "tools" else 0
        print(f"{ev} / {a.model}: {len(jobs)} jobs ({skipped} ineligible), "
              f"{len(done)} already done, {len(todo)} to run over "
              f"{len({j[1] for j in todo})} contexts; "
              f"~{in_tok / 1e6:.1f}M input tokens before caching", flush=True)
    # One group per cached context, across evals; tools (no book in the prompt) last.
    groups = collections.OrderedDict()
    for ev, j in sorted(units, key=lambda u: (u[0] == "tools", u[1][1],
                                              FIRST.get(u[0], 9), u[1][0]["id"])):
        groups.setdefault(("tools", j[1]) if ev == "tools" else j[1], []).append((ev, j))
    if len(evs) > 1:
        print(f"together: {len(units)} calls in {len(groups)} context groups", flush=True)
    if a.dry_run or not units:
        return 0
    model = model or (GeminiCLI(a) if getattr(a, "backend", "openai") == "gemini-cli"
                      else OpenAICompat(a))
    wlock = threading.Lock()
    stats = collections.Counter()

    def one(unit):
        ev, (q, key, _) = unit
        env, res = st[ev]["env"], st[ev]["res"]
        ctx = env["context"](key)
        t0 = time.time()
        try:
            out = RUNNERS[ev](q, ctx, model, env)
            if not (out["response"] or "").strip() and out.get("finish") == "length":
                # A thinking model that spends the whole budget reasoning returns
                # nothing. That is a budget setting, not a wrong answer (seen on
                # the first real run, DeepSeek V4.1 Flash at 4,000 tokens).
                raise CallError("no answer within --max-tokens (reasoning used "
                                "the budget); raise --max-tokens and --retry-errors")
            rec = {"id": q["id"], "status": "ok", **out}
        except Exception as e:  # recorded as infrastructure, never as an answer
            rec = {"id": q["id"], "status": "error",
                   "error": f"{type(e).__name__}: {str(e)[:500]}"}
        rec.update(eval=ev, model=a.model, context=key,
                   seconds=round(time.time() - t0, 1))
        with wlock:
            with io.open(res, "a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats[rec["status"]] += 1
            n = sum(stats.values())
            if n % 10 == 0 or n == len(units) or rec["status"] != "ok":
                tag = rec.get("error", "")[:120]
                print(f"  [{n}/{len(units)}] {ev} {q['id']} {rec['status']} {tag}",
                      flush=True)

    with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for key, us in groups.items():
            if isinstance(key, tuple):  # tools: nothing cached, all at once
                list(pool.map(one, us))
                continue
            one(us[0])  # first call on a context alone: it writes the cache
            list(pool.map(one, us[1:]))
    print(f"done: {dict(stats)} -> " + ", ".join(str(st[ev]["res"]) for ev in evs))
    return 0 if not stats["error"] else 2


# ---- score ------------------------------------------------------------------

def first_answer_line(resp):
    m = re.search(r"^\s*\**\s*answer\s*\**\s*[:\-]\s*(.*)$", resp or "", re.I | re.M)
    return m.group(1).strip() if m else (resp or "").strip()


def verdict(ev, q, rec):
    if ev == "halluc":
        v, _ = adjudicate("__never__", first_answer_line(rec["response"]))
        return ("ABSTAINED" if v == "UNKNOWN" else "ANSWERED"), ""
    if rec.get("finish") == "turn_cap":
        return "WRONG", "turn cap reached without an answer"
    return adjudicate(q["expected_output"], first_answer_line(rec["response"]))


def score_rows(ev, qs_by_id, results):
    last = {}
    for r in results:
        last[r["id"]] = r  # a retried question keeps its latest record
    rows = []
    for qid, rec in sorted(last.items()):
        q = qs_by_id[qid]
        if rec["status"] != "ok":
            rows.append((q, rec, "ERROR", rec.get("error", "")))
            continue
        v, why = verdict(ev, q, rec)
        rows.append((q, rec, v, why))
    return rows


def cmd_score(a):
    qs, _ = load_data()
    by_id = {q["id"]: q for q in qs}
    res = OUT / a.eval / slug(a.model) / "results.jsonl"
    rows = score_rows(a.eval, by_id, [json.loads(l) for l in io.open(res, encoding="utf-8")])
    counts = collections.Counter(v for _, _, v, _ in rows)
    scored = [r for r in rows if r[2] != "ERROR"]
    print(f"{a.eval} / {a.model}: {len(rows)} results, {counts['ERROR']} errors (not scored)")
    for v, n in sorted(counts.items()):
        pct = "" if v == "ERROR" else f"  ({100 * n / max(1, len(scored)):.1f}% of scored)"
        print(f"  {v:10} {n:5}{pct}")
    bytype = collections.defaultdict(collections.Counter)
    for q, _, v, _ in scored:
        bytype[q["question_type"]][v] += 1
    good = "ABSTAINED" if a.eval == "halluc" else "CORRECT"
    for t, c in sorted(bytype.items()):
        n = sum(c.values())
        print(f"    {t:26} {c[good]}/{n} {good.lower()}  unclear {c['UNCLEAR']}")
    if a.eval == "compact":
        surv = [s for _, rec, _, _ in scored for s in rec.get("evidence_survival", [])]
        if surv:
            print(f"  evidence survival: mean {sum(surv) / len(surv):.2f} of gold-quote "
                  f"content words present in the summary ({len(surv)} quotes)")
        cut = sum(rec.get("summary_finish") == "length" for _, rec, _, _ in scored)
        if cut:
            # A summary cut off by the output budget is a shorter summary, not the
            # model's own choice of what to keep. Seen on the first real run:
            # DeepSeek V4.1 Flash spent 25k of 32k tokens reasoning, even at
            # reasoning effort "low", and its notes stopped mid-book.
            print(f"  WARNING: {cut} answers came from summaries cut off at "
                  f"--summary-tokens; raise it or cap reasoning for this model")
    use = {}
    for _, rec, _, _ in rows:
        add_usage(use, rec.get("usage"))
    if use:
        print("  usage: " + json.dumps(use))
    out = res.with_name("scored.jsonl")
    with io.open(out, "w", encoding="utf-8", newline="\n") as fh:
        for q, rec, v, why in rows:
            fh.write(json.dumps({"id": q["id"], "question_type": q["question_type"],
                                 "verdict": v, "why": why,
                                 "gold": q["expected_output"],
                                 "answer": first_answer_line(rec.get("response", ""))[:300]},
                                ensure_ascii=False) + "\n")
    print(f"  -> {out}")
    return 0


# ---- selftest ---------------------------------------------------------------

def cmd_selftest(a):
    """Every eval end to end against scripted fake models: no network, no spend.
    Includes the broken cases: a failing endpoint must produce ERROR rows, not
    WRONG ones, and a model that always answers must score 0% on halluc."""
    global OUT
    qs, stories = load_data()
    by_id = {q["id"]: q for q in qs}
    tmp = Path(tempfile.mkdtemp(prefix="evalsuite_selftest_"))
    OUT = tmp
    fails = []

    def check(cond, msg):
        print(("  ok   " if cond else "  FAIL ") + msg)
        if not cond:
            fails.append(msg)

    def q_of(messages):
        text = "".join(p["text"] if isinstance(p, dict) else p
                       for m in messages for p in (m["content"] if isinstance(m["content"], list)
                                                   else [m["content"] or ""]))
        return text.rsplit("QUESTION:\n", 1)[-1].strip()

    by_q = {q["question"].strip(): q for q in qs}

    def oracle(messages, tools):
        flat = "".join(p["text"] for m in messages if isinstance(m["content"], list)
                       for p in m["content"])
        if COMPACT_REQUEST in flat:
            return {"content": "SUMMARY " * 50, "tool_calls": [], "finish": "stop",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        q = by_q[q_of(messages)]
        return {"content": f"ANSWER: {q['expected_output']}\nbecause.", "tool_calls": [],
                "finish": "stop", "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    def args(ev, **kw):
        base = dict(eval=ev, model="fake", ids=None, types=None, reviewed_only=False,
                    max_input_tokens=None, sample=None, seed=0, limit=None,
                    max_turns=4, summary_tokens=1000, retry_errors=False,
                    dry_run=False, workers=4)
        base.update(kw)
        return argparse.Namespace(**base)

    def results(ev):
        p = OUT / ev / "fake" / "results.jsonl"
        return [json.loads(l) for l in io.open(p, encoding="utf-8")]

    # pick a spread of types, including a multi-novel item
    pick = []
    for t in sorted({q["question_type"] for q in qs}):
        pick += [q["id"] for q in qs if q["question_type"] == t][:2]
    ids = ",".join(pick)

    print("zeroshot, oracle model (gold answers) -> near 100% CORRECT")
    m = Fake(oracle)
    cmd_run(args("zeroshot", ids=ids), model=m)
    rows = score_rows("zeroshot", by_id, results("zeroshot"))
    corr = sum(v == "CORRECT" for _, _, v, _ in rows)
    check(len(rows) == len(pick), f"{len(rows)} results for {len(pick)} questions")
    check(corr >= len(rows) - 2, f"oracle scored {corr}/{len(rows)} CORRECT")
    first = m.calls[0][0]["content"]
    check(isinstance(first, list) and "cache_control" in first[0],
          "context sent as a separate block with a cache breakpoint")
    multi = next(q for q in qs if q["id"] in pick and q["n_stories"] > 1)
    sent = [c for c in m.calls if q_of(c) == multi["question"].strip()][0]
    check("### BOOK 2 of" in sent[0]["content"][0]["text"],
          f"multi-novel {multi['id']} gets every book, labelled")

    print("resume: a second run sends nothing")
    m2 = Fake(oracle)
    cmd_run(args("zeroshot", ids=ids), model=m2)
    check(len(m2.calls) == 0, f"second run made {len(m2.calls)} calls")

    print("broken endpoint -> ERROR rows, never WRONG")
    def boom(messages, tools):
        raise CallError("HTTP 500: simulated outage")
    OUT_z = OUT / "zeroshot" / "fake" / "results.jsonl"
    OUT_z.unlink()
    cmd_run(args("zeroshot", ids=ids), model=Fake(boom))
    rows = score_rows("zeroshot", by_id, results("zeroshot"))
    check(all(v == "ERROR" for _, _, v, _ in rows), "every failed call scored ERROR")
    cmd_run(args("zeroshot", ids=ids, retry_errors=True), model=Fake(oracle))
    rows = score_rows("zeroshot", by_id, results("zeroshot"))
    check(not any(v == "ERROR" for _, _, v, _ in rows), "--retry-errors replaces them")

    print("compact: one summary per context, shared; survival computed")
    m = Fake(oracle)
    cmd_run(args("compact", ids=ids), model=m)
    rs = results("compact")
    n_ctx = len({r["context"] for r in rs})
    n_sum = sum(1 for c in m.calls if COMPACT_REQUEST in c[0]["content"][1]["text"])
    check(n_sum == n_ctx, f"{n_sum} compaction calls for {n_ctx} contexts")
    check(rs and all(r["status"] == "ok" and "evidence_survival" in r for r in rs),
          f"{len(rs)} compact runs ok, evidence_survival recorded")
    check(evidence_survival(["The Colonel rode north"], "colonel rode away") == [0.667],
          "survival = share of content words kept")

    print("halluc: distractor is another author; always-answer model scores 0%")
    d = pick_distractors(qs, stories, 0)
    check(all(stories[a]["author"] != stories[b]["author"] for a, b in d.items()),
          f"{len(d)} distractors, none by the gold author")
    m = Fake(oracle)  # answers with the gold every time = hallucinating
    cmd_run(args("halluc", ids=ids), model=m)
    rows = score_rows("halluc", by_id, results("halluc"))
    check(rows and all(v == "ANSWERED" for _, _, v, _ in rows),
          f"always-answering model: {len(rows)} ANSWERED, 0 ABSTAINED")
    check(all(r[0]["n_stories"] == 1 for r in rows), "only single-novel questions")
    texts = {s["text"]: sid for sid, s in stories.items()}
    ctx_ids = [(by_q[q_of(c)]["story_ids"][0], texts.get(c[0]["content"][0]["text"]))
               for c in m.calls]
    check(all(b and a != b for a, b in ctx_ids),
          "the context is exactly one other novel, never the gold one")
    abst = Fake(lambda msgs, t: {"content": "ANSWER: UNKNOWN", "tool_calls": [],
                                 "finish": "stop", "usage": {}})
    (OUT / "halluc" / "fake" / "results.jsonl").unlink()
    cmd_run(args("halluc", ids=ids), model=abst)
    rows = score_rows("halluc", by_id, results("halluc"))
    check(all(v == "ABSTAINED" for _, _, v, _ in rows), "UNKNOWN scores ABSTAINED")

    print("tools: grep/read work on the book; turn cap enforced")
    q0 = next(q for q in qs if q["id"] in pick and q["n_stories"] == 1)
    files = {"book.txt": stories[q0["story_ids"][0]]["text"].splitlines()}
    g = tool_exec({"function": {"name": "grep", "arguments": json.dumps(
        {"file": "book.txt", "pattern": "the", "max_results": 3})}}, files)
    check(g.split()[0].isdigit() and int(g.split()[0]) > 100, "grep counts matches")
    rd = tool_exec({"function": {"name": "read", "arguments": json.dumps(
        {"file": "book.txt", "start_line": 1, "num_lines": 5})}}, files)
    check(rd.startswith("1: "), "read returns numbered lines")
    check(tool_exec({"function": {"name": "grep", "arguments": '{"file":"x.txt","pattern":"a"}'}},
                    files).startswith("error"), "unknown file is an error, not a crash")
    check(tool_exec({"function": {"name": "grep", "arguments": '{"file":"book.txt","pattern":"("}'}},
                    files).startswith("error"), "bad regex is an error, not a crash")

    def searcher(messages, tools):
        n_tool = sum(1 for m in messages if m["role"] == "tool")
        if n_tool == 0 and tools:
            return {"content": "", "finish": "tool_calls", "usage": {"prompt_tokens": 5},
                    "tool_calls": [{"id": "c1", "type": "function", "function": {
                        "name": "grep", "arguments": json.dumps(
                            {"file": "book_1.txt" if "book_1.txt" in messages[0]["content"]
                             else "book.txt", "pattern": "said"})}}]}
        q = by_q[messages[1]["content"].split("QUESTION:\n", 1)[1].strip()]
        return {"content": f"ANSWER: {q['expected_output']}", "tool_calls": [],
                "finish": "stop", "usage": {"prompt_tokens": 5}}
    cmd_run(args("tools", ids=ids), model=Fake(searcher))
    rs = results("tools")
    check(all(r["status"] == "ok" and r["turns"] == 1 for r in rs),
          f"{len(rs)} tool runs: one grep, then an answer")
    rows = score_rows("tools", by_id, rs)
    check(sum(v == "CORRECT" for _, _, v, _ in rows) >= len(rows) - 2, "tools oracle scores")

    def looper(messages, tools):
        if tools:
            return {"content": "", "finish": "tool_calls", "usage": {},
                    "tool_calls": [{"id": "c", "type": "function", "function": {
                        "name": "wc", "arguments": '{"file":"book.txt"}'}}]}
        return {"content": "", "tool_calls": [], "finish": "length", "usage": {}}
    (OUT / "tools" / "fake" / "results.jsonl").unlink()
    one_id = next(q["id"] for q in qs if q["id"] in pick and q["n_stories"] == 1)
    m = Fake(looper)
    cmd_run(args("tools", ids=one_id, max_turns=3), model=m)
    check(len(m.calls) == 4, f"turn cap 3 -> {len(m.calls)} calls (3 with tools + 1 forced answer)")
    check(m.calls[-1][-1]["content"].startswith("Tool budget used up"),
          "the last call tells the model to answer")

    print("--eval all: every eval in one pass, each book's calls back to back")
    OUT = tmp / "together"
    # Add questions ON the books the picked questions use as distractors, so some
    # book is both a zeroshot context and a halluc distractor in this run.
    sel = [by_id[i] for i in pick]
    extra = []
    for b in set(pick_distractors(sel, stories, 0).values()):
        extra += [q["id"] for q in qs if q["story_ids"] == [b]][:1]
    sel_all = sel + [by_id[i] for i in extra]
    golds = {q["story_ids"][0] for q in sel_all if len(q["story_ids"]) == 1}
    both = golds & set(pick_distractors(sel_all, stories, 0).values())
    ids = ",".join(pick + extra)
    m = Fake(oracle)
    cmd_run(args("all", ids=ids), model=m)
    counts = {ev: len(results(ev)) for ev in EVALS}
    check(counts["zeroshot"] == counts["compact"] == counts["tools"] == len(pick) + len(extra)
          and counts["halluc"] > 0, f"one results file per eval: {counts}")
    check(not any(r["status"] != "ok" for ev in EVALS for r in results(ev)),
          "all ok")
    seq = []  # (call index, book text) for calls that carry a book in context
    for i, c in enumerate(m.calls):
        cont = c[0]["content"]
        if isinstance(cont, list) and not cont[0]["text"].startswith("SUMMARY"):
            seq.append((i, cont[0]["text"]))
    runs = [t for k, (_, t) in enumerate(seq) if k == 0 or seq[k - 1][1] != t]
    check(len(runs) == len(set(runs)),
          f"{len(set(runs))} book contexts, each in one unbroken run of calls")
    halluc_books = {r["context"] for r in results("halluc")}
    shared = [b for b in both if b in halluc_books and stories[b]["text"] in runs]
    check(len(shared) > 0, f"{len(shared)} books are both a zeroshot context and a "
          "halluc distractor, and each sits in a single run (cache written once)")
    last_book = max(i for i, _ in seq)
    tool_calls = [i for i, c in enumerate(m.calls) if c[0]["role"] == "system"]
    check(tool_calls and min(tool_calls) > last_book, "tools calls run after every book")
    n_sum = sum(1 for c in m.calls if isinstance(c[0]["content"], list)
                and COMPACT_REQUEST in c[0]["content"][1]["text"])
    check(n_sum == len({r["context"] for r in results("compact")}),
          f"{n_sum} compactions, one per context")
    m2 = Fake(oracle)
    cmd_run(args("all", ids=ids), model=m2)
    check(len(m2.calls) == 0, f"resume across evals: second run made {len(m2.calls)} calls")

    print("--halluc-sample and --reasoning-effort")
    OUT = tmp / "sample"
    cmd_run(args("halluc", ids=ids, halluc_sample=2), model=Fake(oracle))
    check(len(results("halluc")) == 2, f"halluc sample of 2 -> {len(results('halluc'))}")
    ns = lambda url: argparse.Namespace(base_url=url, key_env="", model="m", max_tokens=1,
                                        timeout=1, extra=None, plain_content=False,
                                        reasoning_effort="medium")
    check(OpenAICompat(ns("https://openrouter.ai/api/v1")).extra ==
          {"reasoning": {"effort": "medium"}}, "OpenRouter gets reasoning.effort")
    check(OpenAICompat(ns("http://localhost:4000/v1")).extra ==
          {"reasoning_effort": "medium"}, "proxy gets reasoning_effort")

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILED'}")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--eval", required=True,
                   help=f"one of {', '.join(EVALS)}; a comma list; or all (run per novel)")
    p.add_argument("--model", required=True, help="model id as the endpoint names it")
    p.add_argument("--backend", choices=("openai", "gemini-cli"), default="openai",
                   help="openai: any OpenAI-compatible endpoint; gemini-cli: Gemini through "
                        "the signed-in Gemini CLI (Google plan, no API key)")
    p.add_argument("--gemini", help="path to the gemini executable "
                                    "(default: tools-gemini-cli/node_modules/.bin)")
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--key-env", default="OPENROUTER_API_KEY",
                   help="environment variable holding the API key")
    p.add_argument("--max-tokens", type=int, default=16000,
                   help="output budget per answer, thinking included (Sonnet 5 averaged ~12.7k)")
    p.add_argument("--summary-tokens", type=int, default=32000,
                   help="compaction output budget (compact eval)")
    p.add_argument("--max-turns", type=int, default=20, help="tool turns (tools eval)")
    p.add_argument("--extra", help='JSON merged into every request, e.g. '
                                   '\'{"reasoning": {"effort": "low"}}\'')
    p.add_argument("--reasoning-effort", choices=("minimal", "low", "medium", "high"),
                   help="sent as reasoning.effort (OpenRouter) or reasoning_effort (others)")
    p.add_argument("--halluc-sample", type=int,
                   help="run only N halluc questions (the budget assumes 300)")
    p.add_argument("--plain-content", action="store_true",
                   help="send content as one string (for endpoints without content parts)")
    p.add_argument("--timeout", type=int, default=3600,
                   help="seconds per call; a compaction summary of a multi-book context on a "
                        "frontier model can take 20+ min (900 failed on GPT-6.1 Sol)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--ids"), p.add_argument("--types")
    p.add_argument("--reviewed-only", action="store_true",
                   help="skip questions still Awaiting verification")
    p.add_argument("--max-input-tokens", type=int,
                   help="skip questions whose context is longer (o200k), e.g. for a 128k model")
    p.add_argument("--sample", type=int), p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int)
    p.add_argument("--retry-errors", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("score")
    p.add_argument("--eval", choices=EVALS, required=True)
    p.add_argument("--model", required=True)
    p.set_defaults(func=cmd_score)
    sub.add_parser("selftest").set_defaults(func=cmd_selftest)
    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
