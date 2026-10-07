"""G-check: can a frontier model answer the question WITH the whole novel in front of it?

The measurement the project has never made. E asks whether an item is answerable
from memory and P from a synopsis -- both are floors, proving an item is not
trivially recoverable. G is the ceiling: given the text, how often does a model
actually get it right? Without G there is no accuracy number, so "long-context
reading is hard" stays a design argument.

Asked for by Shivank on 2026-08-09: paste the entire story plus the question to
a frontier model, code generation OFF, and try Claude, GPT and Gemini. Code
generation is disabled on purpose -- a model that writes a script to count
occurrences is measuring its tool use, not its reading, and counting is exactly
what the computation items ask for.

Split at the model boundary, like echeck.py and pcheck.py, for the same reason:
everything except the model pass is local, deterministic and testable.

    prepare   write one open-book prompt per question + a manifest
    run       OPTIONAL convenience runner for Claude via the `claude` CLI
    score     adjudicate the responses against gold
    selftest  run the scorer against known cases (no model, no network)

Prompts go under data/ (regenerable, gitignored). Responses go under verify/
because model inference costs real money and cannot be regenerated.

    python scripts/gcheck.py prepare --sample 24
    python scripts/gcheck.py run --model claude
    python scripts/gcheck.py score
"""
import argparse
import io
import json
import re
import subprocess
import threading
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
PROMPTS = ROOT / "data" / "v3" / "gcheck" / "prompts"
RESPONSES = ROOT / "verify" / "gcheck" / "responses"
MANIFEST = ROOT / "verify" / "gcheck" / "manifest.json"

INSTRUCTION = """You are answering a reading-comprehension question about the novel above.

Rules:
- Answer ONLY from the novel text provided. It is the complete work.
- Do NOT write or execute code, and do not describe a program you would write.
  Read the text and answer.
- Answer directly and briefly. For a count, give the number. For a name or
  phrase, give it as the novel gives it.
- If the text genuinely does not settle it, reply exactly: UNKNOWN

FORMAT -- this matters:
Your FIRST line must be exactly `ANSWER: <your answer>` and nothing else.
Put any reasoning on the lines after it. A count question must give the count
as a bare number on that first line.

(This is not a style preference. Replies that lead with reasoning cannot be
scored mechanically: one earlier reply opened by mentioning a distance of
"four miles" and another numbered its working "(1)...(4)", and both defeat any
attempt to read the answer out of the prose.)

QUESTION:
{question}

ANSWER:"""


def norm(s):
    s = re.sub(r"[^\w\s]", " ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()


NUMWORD = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
           "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
           "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30,
           "forty": 40, "fifty": 50, "sixty": 60, "hundred": 100,
           "thousand": 1000}


def numbers_in(s):
    out = {int(m.replace(",", "")) for m in re.findall(r"\d[\d,]*", str(s))}
    out |= {NUMWORD[w] for w in norm(s).split() if w in NUMWORD}
    return out


def adjudicate(gold, response):
    """CORRECT / WRONG / UNKNOWN / UNCLEAR for one open-book answer.

    Deliberately conservative: it returns UNCLEAR rather than guessing, because
    a wrong automatic verdict here corrupts the one accuracy number the project
    has. UNCLEAR is a queue for a judge or a human, not a failure.
    """
    g, r = norm(gold), norm(response)
    if not r:
        return "UNCLEAR", "empty response"
    if re.match(r"^\W*unknown\W*$", r):
        return "UNKNOWN", "model declined"

    gn = numbers_in(gold)
    if gn:
        # The model's STATED answer, not any number anywhere in its reply.
        #
        # Scoring against every number in the response is badly wrong for
        # exactly the items that matter. comp-1003 answered "**Four.**" against
        # a gold of 3 and then enumerated its reasoning "(1) ... (2) ... (3)
        # ... (4)"; the list marker 3 matched the gold and the item scored
        # CORRECT. A count question invites the model to number its working, so
        # the failure is systematic on the type it most affects, and it inflates
        # the one accuracy figure this project has.
        #
        # The stated answer is the number in the opening sentence -- models put
        # the answer first when asked to answer directly, which the prompt does.
        # Only three shapes count as a STATED answer, and anything else is
        # queued rather than judged. Taking "the first sentence" was the
        # obvious next guess and it is also wrong: comp-1004's reply opens
        # "Bardon complains Annabella never managed to come four miles to see
        # him", where the 4 is incidental reasoning, and scoring that WRONG
        # against a gold of 10 is a false negative as bad as the false positive
        # it replaced. When the answer cannot be located confidently, say so.
        stated = None
        # A SHORT reply has no room for incidental numbers, so every number in
        # it is part of the answer: "The answer is three." and "about 1,200
        # pounds" are unambiguous even though neither leads with a digit. The
        # ambiguity this whole branch guards against only appears once a model
        # starts reasoning at length.
        body = response.strip()
        # Exactly ONE number, not merely a short reply. A 118-character answer
        # can still enumerate its working ("(1) ... (2) ... (3) ... (4)"), and
        # then "short" is no protection at all -- caught by the scorer's own
        # test case, not in the wild. One number means one candidate answer.
        if len(body) <= 120:
            cand = numbers_in(body)
            if len(cand) == 1:
                stated = cand
        m = None if stated else re.match(
            r"\W*(?:answer\s*[:\-]\s*)?\**\s*"
            r"(\d[\d,]*|[a-z]+)\b", body, re.I)
        if m:
            cand = numbers_in(m.group(1))
            if cand:
                stated = cand
        if stated is None:
            m = re.search(r"^\s*(?:final\s+)?answer\s*[:\-]\s*(.+)$",
                          response, re.I | re.M)
            if m:
                cand = numbers_in(m.group(1)[:80])
                if cand:
                    stated = cand
        if stated is None:
            return "UNCLEAR", ("numeric gold; the reply does not lead with a "
                               "number, so its stated answer cannot be read "
                               "mechanically")
        if gn & stated:
            return "CORRECT", f"stated {sorted(stated)} matches gold {sorted(gn)}"
        return "WRONG", f"stated {sorted(stated)}, gold {sorted(gn)}"

    if g and g in r:
        return "CORRECT", "gold answer appears in the response"
    gwords = [w for w in g.split() if len(w) > 3]
    if gwords:
        hits = sum(1 for w in gwords if re.search(r"\b" + re.escape(w) + r"\b", r))
        frac = hits / len(gwords)
        if frac >= 0.8:
            return "CORRECT", f"{hits}/{len(gwords)} gold content words present"
        # 0.25, not 0.4. A prose gold like "a metal shaving-stick box" can be
        # answered correctly as "a small metal container for shaving soap",
        # which shares one content word in three. Auto-scoring that WRONG would
        # understate accuracy on exactly the items where wording varies most,
        # and a wrong automatic verdict corrupts the one accuracy number this
        # project has. Partial overlap queues for a judge; only zero overlap is
        # called wrong outright.
        if frac >= 0.25:
            return "UNCLEAR", f"{hits}/{len(gwords)} gold content words present"
    return "WRONG", "no overlap with gold"


def load_live():
    rows = [r for r in json.load(
        io.open(ROOT / "authoring/notion_pull.json", encoding="utf-8"))
        if r.get("question") and r.get("answer")]
    assigns = {r["id"]: r for r in json.load(
        io.open(ROOT / "authoring/assignments.json", encoding="utf-8"))}
    return rows, assigns


def cmd_prepare(a):
    rows, assigns = load_live()
    by_type = defaultdict(list)
    for r in rows:
        if r["id"] in assigns:
            by_type[r.get("type")].append(r)

    if a.ids:
        want = {i.strip() for i in a.ids.split(",")}
        picked = [r for r in rows if r["id"] in want]
    else:
        # Even across types, so the accuracy number is not dominated by
        # whichever type happens to be most numerous.
        per = max(1, a.sample // max(1, len(by_type)))
        picked = []
        for t in sorted(by_type):
            picked += sorted(by_type[t], key=lambda r: r["id"])[:per]

    PROMPTS.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest, total = [], 0
    for r in picked:
        asg = assigns[r["id"]]
        # EVERY gold, not just the primary. A cross-novel item's halves live in
        # different books, so supplying one is supplying half the question --
        # and the model then answers "UNKNOWN, only <book> was supplied", which
        # scores as WRONG. Measured 2026-08-24: all four cross-novel items in
        # the first run were given one book, and two of the four "model errors"
        # were this harness, not the model.
        #
        # That is the anti-pattern this project already paid for once: an
        # infrastructure failure written into a results field is
        # indistinguishable from a negative result. Third instance of the same
        # structural gap in one day, after gate_drafts.py and the repair pass --
        # anything that reads an item's gold must read `co_source_ids` too.
        sids = [asg["source_id"]] + list(asg.get("co_source_ids") or [])
        books = []
        for s in sids:
            books.append(io.open(ROOT / f"data/v3/by_story/{s}/story.md",
                                 encoding="utf-8").read())
        sep = "\n\n" + "=" * 70 + "\n\n"
        if len(books) > 1:
            # Labelled, because a solver told only "here are two books" cannot
            # say which half it is answering from.
            body = sep.join(f"### BOOK {n} of {len(books)}\n\n{b}"
                            for n, b in enumerate(books, 1))
        else:
            body = books[0]
        prompt = body + sep + INSTRUCTION.format(question=r["question"])
        p = PROMPTS / f"{r['id']}.txt"
        io.open(p, "w", encoding="utf-8", newline="\n").write(prompt)
        total += len(prompt)
        manifest.append({"id": r["id"], "type": r.get("type"),
                         "subtype": r.get("subtype"), "source_id": sids[0],
                         "source_ids": sids,
                         "gold": str(r["answer"]), "chars": len(prompt)})
    io.open(MANIFEST, "w", encoding="utf-8", newline="\n").write(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {len(manifest)} prompts to {PROMPTS}")
    print(f"total {total/1e6:.1f}M chars (~{total/4/1e3:.0f}k tokens) per model run")
    print(f"manifest: {MANIFEST}")
    print(f"\nDrop replies in {RESPONSES}/<model>/<qid>.txt, then: "
          f"python scripts/gcheck.py score")
    return 0


def cmd_run(a):
    """Optional runner. Only Claude, and only through the local CLI.

    The runner is deliberately thin and opt-in: the project's other model checks
    keep it out of scope entirely so they work with subagents, an API script or
    a human. This exists because the Claude arm can run today on the local
    subscription while GPT and Gemini have no keys configured.
    """
    manifest = json.load(io.open(MANIFEST, encoding="utf-8"))
    out = RESPONSES / a.model
    out.mkdir(parents=True, exist_ok=True)
    todo = [m for m in manifest if not (out / f"{m['id']}.txt").exists()]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(todo)} of {len(manifest)} still to run against {a.model}")
    for i, m in enumerate(todo, 1):
        prompt = io.open(PROMPTS / f"{m['id']}.txt", encoding="utf-8").read()
        print(f"  [{i}/{len(todo)}] {m['id']} ({m['chars']/1e3:.0f}k chars) ...",
              flush=True)
        try:
            # Prompt on STDIN, never argv. These prompts average ~350,000
            # characters and Windows caps a command line near 32,000, so the
            # argv form does not merely truncate -- it fails outright, and a
            # runner that failed silently would fill the response files with
            # error text that later scores as WRONG.
            p = subprocess.run(["claude", "-p"], input=prompt,
                               capture_output=True, text=True,
                               encoding="utf-8", timeout=a.timeout)
            reply = (p.stdout or "").strip() or \
                f"(no stdout; rc={p.returncode}; err={(p.stderr or '')[:200]})"
        except subprocess.TimeoutExpired:
            reply = "(timeout)"
        io.open(out / f"{m['id']}.txt", "w", encoding="utf-8",
                newline="\n").write(reply)
        print(f"        -> {reply[:90]}")
    return 0


def cmd_score(a):
    manifest = {m["id"]: m for m in json.load(io.open(MANIFEST, encoding="utf-8"))}
    root = (ROOT / a.dir) if getattr(a, "dir", None) else RESPONSES
    models = sorted(d.name for d in root.glob("*") if d.is_dir()) \
        if root.exists() else []
    if not models:
        print(f"no responses under {root}")
        return 1
    for model in models:
        rows, counts, overturned = [], Counter(), Counter()
        for qid, m in sorted(manifest.items()):
            p = root / model / f"{qid}.txt"
            if not p.exists():
                continue
            reply = io.open(p, encoding="utf-8").read().strip()
            verdict, why = adjudicate(m["gold"], reply)
            if getattr(a, "judged", False):
                jf = judged_dir(root, model) / f"{qid}.json"
                if jf.exists():
                    j = json.load(io.open(jf, encoding="utf-8"))
                    if j["verdict"] != verdict:
                        overturned[f"{verdict}->{j['verdict']}"] += 1
                    verdict, why = j["verdict"], "judge: " + j["why"]
            counts[verdict] += 1
            rows.append((qid, m["type"], verdict, why, reply[:70]))
        n = sum(counts.values())
        tag = "" if root == RESPONSES else f" [{root.name}]"
        print(f"\n=== {model}: {n} answered{tag} ===")
        for v in ("CORRECT", "WRONG", "UNKNOWN", "UNCLEAR"):
            if counts[v]:
                print(f"  {v:8} {counts[v]:3}  ({100.0*counts[v]/n:.0f}%)")
        bytype = defaultdict(Counter)
        for qid, t, v, _w, _r in rows:
            bytype[t][v] += 1
        for t in sorted(bytype):
            c = bytype[t]
            tot = sum(c.values())
            print(f"    {t:12} {c['CORRECT']}/{tot} correct")
        if overturned:
            print("  judge overturned: " + ", ".join(
                f"{k} x{v}" for k, v in sorted(overturned.items())))
        if a.verbose:
            for qid, t, v, why, reply in rows:
                if v != "CORRECT":
                    print(f"    {v:8} {qid:11} {why} | {reply}")
    return 0


def cmd_selftest(a):
    cases = [
        ("3", "3", "CORRECT"), ("3", "The answer is three.", "CORRECT"),
        ("3", "4", "WRONG"), ("30500", "$30,500", "CORRECT"),
        # the real comp-1003 shape: stated answer four, working numbered 1..4,
        # so the gold 3 appears only as a list marker
        ("3", "**Four.** She lodges at (1) the vicarage; (2) the cottage; "
              "(3) the vicarage again; and (4) the fisherman's hut.", "WRONG"),
        ("4", "**Four.** Because (1) a, (2) b, (3) c and (4) d.", "CORRECT"),
        ("3", "Let me work through the evidence carefully.", "UNCLEAR"),
        ("a metal shaving-stick box", "a metal shaving-stick box", "CORRECT"),
        # zero content-word overlap -> WRONG outright
        ("a metal shaving-stick box", "a tin box", "WRONG"),
        # partial overlap on a legitimate paraphrase -> queued, never auto-wrong
        ("a metal shaving-stick box",
         "a small metal container he used for shaving soap", "UNCLEAR"),
        ("Chicago", "Chicago", "CORRECT"), ("Chicago", "London", "WRONG"),
        ("Chicago", "UNKNOWN", "UNKNOWN"), ("Chicago", "", "UNCLEAR"),
        ("1200", "about 1,200 pounds", "CORRECT"),
        ("four hundred feet or more", "four hundred feet or more", "CORRECT"),
    ]
    ok = fail = 0
    for gold, resp, want in cases:
        got, why = adjudicate(gold, resp)
        if got == want:
            ok += 1
            print(f"  ok   {gold!r:32} <- {resp!r:28} = {got}")
        else:
            fail += 1
            print(f"  FAIL {gold!r:32} <- {resp!r:28} = {got} (want {want}) {why}")
    print(f"\n{ok}/{ok+fail} scorer cases pass")
    return 1 if fail else 0


JUDGED = ROOT / "verify" / "gcheck" / "judged"

JUDGE_PROMPT = """You are adjudicating one answer against a reference answer.

You are NOT judging whether the reference is right. It is authoritative and was
written by a human who read the book. You judge ONE thing: does the candidate
state the SAME answer as the reference?

QUESTION
{question}

REFERENCE ANSWER (authoritative)
{gold}

CANDIDATE ANSWER
{reply}

Rules:
- CORRECT: the candidate states the same fact as the reference, even in
  completely different words, and even if it adds detail around it.
- WRONG: the candidate states a different fact -- a different number, person,
  place, cause or object -- or never answers the question.
- UNCLEAR: the candidate floats several answers without committing to one, or
  is too vague to compare against the reference.

A number must match exactly to be CORRECT. Extra correct detail is not a defect.
Missing detail is only a defect if the missing part IS the answer.

MULTI-PART REFERENCES. If the reference is labelled (a), (b), (c) ... then
EVERY labelled part is part of the answer. The candidate is CORRECT only if it
answers every one correctly. If it gets some parts right and any part wrong, or
omits a part, the verdict is WRONG -- not CORRECT and not UNCLEAR. Do not
average across parts and do not award the item for a majority.

Reply with exactly one line and nothing else:
VERDICT: <CORRECT|WRONG|UNCLEAR> - <at most 20 words of reason>
"""


def judged_dir(respdir, model):
    return JUDGED / f"{respdir.name}__{model}"


def cmd_judge(a):
    """Adjudicate the verdicts the mechanical scorer refused to call.

    `adjudicate()` is deliberately conservative and returns UNCLEAR rather than
    guessing -- that is correct, and its thresholds must NOT be loosened to make
    the bucket smaller. Loosening them was the obvious first move here and it is
    wrong: the comments in adjudicate() record two measured cases where a looser
    rule scored a wrong answer CORRECT. UNCLEAR is a QUEUE, and this drains it.

    The judge never sees the novel. It compares three strings -- question, gold,
    reply -- and decides whether two answers are the same answer. That keeps it
    cheap (a few hundred tokens against ~350k for an open-book run) and, more
    importantly, keeps it from re-deciding the question on its own evidence: a
    judge holding the novel would start grading the GOLD, and the gold is the
    fixed point everything else is measured against.

    WRONG is judged too, not only UNCLEAR. "No overlap with gold" is a statement
    about vocabulary, not about meaning, and a correct paraphrase that happens
    to share no content words lands there.

    Verdicts are cached per response directory, so re-scoring costs nothing.
    """
    manifest = {m["id"]: m for m in json.load(io.open(MANIFEST, encoding="utf-8"))}
    live = {r["id"]: r for r in json.load(
        io.open(ROOT / "authoring/notion_pull.json", encoding="utf-8"))}
    respdir = ROOT / a.dir if not Path(a.dir).is_absolute() else Path(a.dir)
    src = respdir / a.model
    if not src.is_dir():
        print(f"no responses under {src}")
        return 1
    out = judged_dir(respdir, a.model)
    out.mkdir(parents=True, exist_ok=True)

    todo = []
    for qid, m in sorted(manifest.items()):
        f = src / f"{qid}.txt"
        if not f.exists():
            continue
        reply = io.open(f, encoding="utf-8").read().strip()
        mech, why = adjudicate(m["gold"], reply)
        if mech in ("CORRECT", "UNKNOWN"):
            continue
        if (out / f"{qid}.json").exists() and not a.redo:
            continue
        todo.append((qid, m, reply, mech, why))

    print(f"{len(todo)} verdict(s) to adjudicate in {respdir.name}/{a.model}")
    if not todo:
        return 0

    lock = threading.Lock()
    done = []

    def work(item):
        qid, m, reply, mech, why = item
        q = (live.get(qid) or {}).get("question", "(question text unavailable)")
        prompt = JUDGE_PROMPT.format(
            question=q, gold=m["gold"], reply=reply[:3000])
        try:
            pr = subprocess.run(["claude", "-p"], input=prompt,
                                capture_output=True, text=True,
                                encoding="utf-8", timeout=a.timeout)
            txt = (pr.stdout or "").strip()
        except subprocess.TimeoutExpired:
            txt = ""
        g = re.search(r"VERDICT:\s*(CORRECT|WRONG|UNCLEAR)\s*[-\u2013\u2014:]?\s*(.*)",
                      txt, re.I)
        if g:
            verdict, reason = g.group(1).upper(), g.group(2).strip()[:120]
        else:
            # An unparseable judge reply must not silently become a verdict.
            verdict, reason = "UNCLEAR", f"judge reply unparseable: {txt[:60]!r}"
        io.open(out / f"{qid}.json", "w", encoding="utf-8", newline="\n").write(
            json.dumps({"id": qid, "mech": mech, "mech_why": why,
                        "verdict": verdict, "why": reason},
                       ensure_ascii=False, indent=1) + "\n")
        with lock:
            done.append(qid)
            print(f"  [{len(done)}/{len(todo)}] {qid:11} {mech:8} -> "
                  f"{verdict:8} {reason[:60]}", flush=True)

    sem = threading.Semaphore(a.jobs)

    def guarded(item):
        with sem:
            work(item)

    threads = [threading.Thread(target=guarded, args=(i,)) for i in todo]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"\n-> {out}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--sample", type=int, default=24)
    p.add_argument("--ids")
    p.set_defaults(func=cmd_prepare)
    p = sub.add_parser("run")
    p.add_argument("--model", default="claude")
    p.add_argument("--limit", type=int)
    p.add_argument("--timeout", type=int, default=900)
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("score")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--dir", help="response root (default verify/gcheck/responses)")
    p.add_argument("--judged", action="store_true",
                   help="fold in cached judge verdicts for UNCLEAR/WRONG")
    p.set_defaults(func=cmd_score)
    p = sub.add_parser("judge")
    p.add_argument("--dir", default="verify/gcheck/responses")
    p.add_argument("--model", default="claude")
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--redo", action="store_true")
    p.set_defaults(func=cmd_judge)
    sub.add_parser("selftest").set_defaults(func=cmd_selftest)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
