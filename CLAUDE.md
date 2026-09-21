# Hiver SDE Intern — Take-Home Assignment

## What this repo is

An AI customer-support agent for **one brand** picked from the Customer Support
on Twitter dataset. The agent must, per incoming customer message:

1. **Classify intent** — into a small set of intents defined from the data (not
   an off-the-shelf taxonomy, unless justified).
2. **Draft a reply** — grounded in how this brand has historically resolved
   similar issues (retrieval over past resolved threads, not free-form
   generation).
3. **Decide auto-handle vs. escalate** — with a stated reason attached to
   every decision.

The grading bar is not "does it run" — it's **"can you prove it's good enough
to trust, and can you explain/modify the code live."** Treat evaluation and
the report as first-class deliverables, not an afterthought bolted on at the
end. Budget time accordingly: a mediocre agent with a rigorous eval beats a
fancier agent with a hand-wavy one.

## Non-negotiable constraints

- **README must let a grader reproduce headline results in under 15
  minutes.** Assume they run it once, on a subsample, on a cold machine.
  Pin dependency versions. Don't require the full 3M-row dataset — a
  subsample is expected and encouraged by the assignment itself.
- **Never assume the full dataset is loaded.** Grading explicitly uses a
  subsample. Code paths must degrade gracefully on small slices — no logic
  that only produces sane output above some row-count threshold.
- **Golden eval set: 150–250 hand-labelled examples**, built by the author
  (not scraped from an existing labelled set), with a documented sampling +
  labelling method. Keep the labelling notes with the data, not just in the
  report.
- **LLM-as-judge must be validated against human judgment** — report
  agreement (e.g. Cohen's kappa, or agreement %) between the judge and a
  human on a subset. A judge with unreported/unknown agreement is a red
  flag for this assignment, not a nice-to-have.
- **Two baselines minimum**: one trivial (e.g. majority-class intent,
  canned/most-common reply, always-escalate) and one simple (e.g.
  keyword/TF-IDF classifier, plain retrieval-only reply with no LLM
  reasoning). The agent must be compared against both, not just reported in
  isolation.
- **Report ≤ 6 pages** (or an equivalent README section) covering: problem
  framing (what "good" means for this brand + what was deliberately not
  built), results vs. baselines, top-5 failure modes with real examples and
  hypotheses, a mandatory "what's misleading about my headline number"
  section, next steps with one more week, and a decision log of 10–15
  non-obvious decisions with reasons (bullets are fine).
- **Cite everything borrowed** — code, prompts, snippets from AI assistants
  or elsewhere. Not knowing what was borrowed is treated as a failure
  condition by the graders, not a style nitpick.

## Suggested structure (adapt as the repo grows)

```
data/            # raw/sample data + golden eval set (keep golden set separate from training/dev data)
src/             # pipeline code: ingestion, intent classifier, retrieval, reply drafting, escalation policy
eval/            # eval harness: automated metrics + LLM-judge, judge-vs-human agreement analysis
notebooks/       # exploration only — nothing load-bearing for reproducing headline results should live only in a notebook
report/          # report.md (or PDF), decision log
README.md        # setup + reproduce-in-15-minutes instructions
```

Keep the "one command reproduces headline numbers" path scriptable
end-to-end (e.g. a single `make eval` / `run.sh` / `python run_pipeline.py`).
Don't let the reproducible path depend on manual notebook steps.

## Working conventions

- Prefer a single brand chosen early and stated explicitly in the README —
  don't build brand-agnostic abstraction the assignment doesn't ask for.
- Every escalation decision needs a human-readable reason string attached to
  it (this is a stated deliverable requirement, not internal debug output).
- When adding a metric, ask "what would make this number misleading?" up
  front — the report's mandatory section should not be written in a rush at
  the end.
- The author must be able to explain and modify any code live — avoid
  opaque, over-engineered abstractions even if an assistant suggests them.
  Simple, legible code beats clever code for this assignment.

## Submission

- Submit via the form (repo link — public or access-granted-private — plus
  the report). Do not email submissions.
- Do not run against the full dataset when producing headline numbers —
  graders will not do so either.
