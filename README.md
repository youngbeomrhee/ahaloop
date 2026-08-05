# ahaloop

**Turn any video into a learning loop.**

> Status: pre-alpha — Session 0 (scaffold only). Nothing works yet.

## What it is

ahaloop takes a video you want to learn from — a conference talk, a lecture, a
long technical explainer — and turns it into a loop you can actually close:
a cited learning pack, retrieval quizzes drawn from it, and regeneration
targeted at whatever you turned out not to understand.

Every claim it produces points back at the timestamp it came from, so you can
check the source instead of trusting the model.

## Why

Watching a video feels like learning and mostly isn't. What actually moves
understanding is the moment something clicks — and the useful question is not
whether you get those moments, but **how often**. Frequency of insight, not
hours logged, is the thing worth optimizing.

The loop that produces them is **recursive gap filling**: run something you
don't fully understand, notice the exact point where your model breaks, descend
into that gap, explain it back in your own words, and repeat one level down.
The bottleneck is never material — it's noticing the gap and closing it before
it silently becomes a permanent hole.

ahaloop is that loop as a tool. It watches where your understanding fails
(wrong quiz answers, failed explanations) and regenerates material aimed at
precisely those gaps, instead of handing you another summary of what you
already knew.

**Bring your own LLM.** Inference runs on a model *you* own — your API key,
your subscription, or a local runtime like Ollama. ahaloop provides the loop,
never the inference bill. Every model call goes through a single provider-neutral
adapter, so no provider is hardcoded anywhere in the codebase.

## Roadmap

| Milestone | What ships |
| --- | --- |
| **M1** | CLI: YouTube URL → transcript → learning pack. First dogfooding run. |
| **M2** | Cited RAG over the source, plus a 20–30 item evaluation set so retrieval quality is measured rather than assumed. |
| **M3** | Gap-driven regeneration — quizzes generate material only where you failed — exposed as an MCP server for use from any agent. |
| **M4** | Minimal web UI: ingest, streaming progress, citations, feedback. |

### Beyond v1

- **Recommendations** — surface the next thing worth learning from your own gap log and interest graph, reusing the M2 embedding index.
- **Social learning graph** — follow other learners, share what you actually understood. The reward signal is *verified understanding* (gates passed, reviews survived), never watch time. Optimizing for watch time is how a learning tool decays into a consumption app.
- **BYO-LLM stays** — the principle holds at every scale. Users grow, inference costs don't move to us.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

Requires Python 3.11+.

## License

MIT
