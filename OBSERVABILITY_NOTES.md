# Observability Notes — SFR

What instrumenting this chain actually revealed, and what to look for next.

The file is split deliberately. Part 1 is measured and reproducible on any machine
with no credentials and no model. Part 2 needs a real provider and a LangSmith
key, and is left as questions rather than answers — writing plausible findings
there would defeat the point of having observability at all.

A note on what tracing can and cannot do for this project. SFR generates from the
ticket in front of it and retrieves nothing, so there is exactly one place an
answer can go wrong: generation. That makes its traces less diagnostic than the
RAG counterpart's ([past-ticket-knowledge-rag](https://github.com/LogicHarborgini/past-ticket-knowledge-rag)),
where a bad answer is either bad retrieval or bad generation and the trace is what
tells the two apart. Here the trace answers narrower questions: how long did it
take, how many tokens did it cost, and what exactly was the model asked.

---

## Part 1 — Measured offline

Reproduce with no API key and no model:

```bash
LLM_PROVIDER=fake LANGSMITH_TRACING=false python run_sfr_traces.py
LLM_PROVIDER=fake python -m evals.simple_eval
```

Configuration: `LLM_PROVIDER=fake` (canned responses, real code path), 5 sample
tickets across P1–P3, `MAX_CONTENT_CHARS=4000`.

### Finding 1 — `preprocess-ticket` is not the bottleneck, and the span is how we know

Whitespace collapsing across all 5 sample tickets:

| Ticket | Priority | Before | After | Saved |
|---|---|---|---|---|
| TICK-2001 | P1 | 170 | 170 | 0 |
| TICK-2002 | P2 | 171 | 171 | 0 |
| TICK-2003 | P3 | 122 | 122 | 0 |
| TICK-2004 | P1 | 138 | 109 | 29 |
| TICK-2005 | P2 | 174 | 174 | 0 |
| **Total** | | **775** | **746** | **29 (3.7%)** |

That is roughly 7 tokens saved across five tickets — negligible. Only TICK-2004
saves anything, because it is the one sample written as a hard-wrapped multi-line
string; the other four are already single-line.

The useful conclusion is not "preprocessing is worthless." It is that the span
earns its place by *proving* preprocessing is not where the time or the tokens go,
which is only worth knowing because it is cheap to establish and stops the
optimisation from being attempted. The step pays for itself on real pasted ticket
bodies carrying quoted email chains, and the sample set does not contain one.

### Finding 2 — the sample set never exercises the truncation path

The longest sample ticket is 174 characters against a 4000-character cap, so
`preprocess_ticket` truncates nothing in any of the five traces. The truncation
branch is covered by [tests/test_preprocessing.py](tests/test_preprocessing.py)
and by no trace.

This matters because the token-budget question in Part 2 cannot be answered from
these traces. A ticket that arrives at 6000 characters is where the interaction
between the content cap, the system prompt, and the 512-token response ceiling
becomes real, and nothing here reaches it. Adding one deliberately oversized
sample ticket is the cheapest way to close that gap.

### Finding 3 — a 100% eval score under `fake` measures the harness, not the model

`simple_eval` reports:

```
Overall: 100% across 3 test case(s)
```

with `signals_urgency` marked `n/a` on the P2 and P3 cases rather than passed —
urgency language is required on P1 and not expected on P3, and a criterion that
does not apply is skipped rather than counted.

The score is still worth nothing as a quality claim. `LLM_PROVIDER=fake` returns
canned responses that were written to satisfy these criteria, so a 100% here
confirms the criteria run and the skip logic works. The harness prints the caveat
itself on every run — `responses are canned. Tracing and evals are real; response
quality is not` — and that line is the reason the number is safe to publish.

Re-run against `bedrock` before quoting this figure anywhere. Until then the
honest statement is "the regression gate is wired and green on a known input,"
not "responses score 100%."

### Finding 4 — the trace generator died on its own progress output

Before the current fix, `run_sfr_traces.py` printed a `U+2192` arrow and
`format_sfr_output` built its header from `U+2500` box-drawing characters. A
Windows console defaults to cp1252, which encodes neither, and `print()` raises
`UnicodeEncodeError` rather than substituting — so the script crashed partway
through the second ticket, having generated one trace instead of five.

Two things worth taking from it. First, the failure was in the observability
tooling rather than the application, which is the category of bug that quietly
produces less data than you think you have. Second, it only reproduces on a
Windows console; the same code is fine on macOS and in CI. The fix is ASCII in
anything routed to stdout, non-ASCII confined to comments and docstrings, and
[a regression test asserting the header survives `cp1252.encode()`](tests/test_preprocessing.py).

**Caveat applying to all four findings:** these come from the `fake` provider. The
code path, the spans, and the character counts are real; latency and token figures
are not, because no model ran.

---

## Part 2 — To fill in from LangSmith

Not yet run. Requires a real provider and `LANGSMITH_API_KEY`:

```bash
LLM_PROVIDER=bedrock LANGSMITH_TRACING=true python run_sfr_traces.py
```

Then open each trace at https://smith.langchain.com and replace each question
with what you actually saw. A specific finding from a real trace is the difference
between describing observability and having used it.

### Latency split

```
SFR-TICK-2001                    [total: ____ms]
├── preprocess-ticket            [____ms]
└── RunnableSequence             [____ms]
    └── ChatBedrock              [____ms]
```

- What fraction of total latency is the model call? Part 1 predicts preprocessing
  is ~0% — confirm it, because if it is not, something is wrong with the input.
- Is first-token latency meaningfully lower than total? That number is the case
  for or against adding a streaming endpoint.

### Token budget

Open the `ChatBedrock` node and read the input token count.

- How many tokens does the system prompt consume before any ticket content?
- What is the ratio of system prompt to ticket content on the shortest sample
  (TICK-2003, 122 chars)? If the prompt dominates, prompt length is the cost
  driver, not ticket length.
- Does any response hit the 512-token ceiling? A truncated response mid-sentence
  is a visible failure the eval criteria do not currently check for.

### Does priority actually change the output

Filter by `priority:P1`, then `priority:P3`.

- Is P1 output measurably more urgent, or does the prompt only claim it is? The
  chain does not branch on priority — it is passed to the model as text, so any
  difference is the model's interpretation and could be nothing.
- Is latency or response length systematically different by priority? There is no
  mechanism that should make it so.

### The instruction the prompt gives that is hardest to keep

The system prompt asks the model not to diagnose the issue in a first response.

- Did any of the five responses attempt a diagnosis anyway?
- If so, that is the single highest-value prompt fix available, and `sfr_eval`
  already has a judge criterion for it — compare the judge's verdict against your
  own reading of the trace, because a judge that disagrees with you on a case you
  are certain about is a judge you cannot trust on the cases you are not.

---

## What to write down once Part 2 is filled in

One paragraph per finding, in this shape:

> I opened the trace for [ticket] and saw [specific observation with a number].
> That told me [what is responsible]. I changed [the specific lever] and
> [metric] moved from [before] to [after].

Two or three of those are worth more in an interview than a description of what
LangSmith does — anyone can install it; the finding proves you read the output.

## Related

- [past-ticket-knowledge-rag](https://github.com/LogicHarborgini/past-ticket-knowledge-rag)
  — the RAG counterpart, where tracing separates retrieval failure from generation
  failure and has correspondingly more to say
- `python -m evals.simple_eval` — the deterministic regression gate
- `python -m evals.sfr_eval` — LLM-as-judge on the criteria deterministic checks
  cannot express
