# Agentic Retrieval Matrix (ARM)

**Factorial benchmark for agentic search** — isolate how **retriever**, **result delivery**, and **agent harness** interact, inspired by [*Is Grep All You Need? How Agent Harnesses Reshape Agentic Search*](https://arxiv.org/abs/2605.15184) (arXiv:2605.15184).

The paper’s core claim is not “grep always wins.” It’s that **evaluation must cover the full agent loop**. ARM makes that loop explicit and reproducible.

## What ARM measures

| Axis | Variants | Maps to paper |
|------|----------|----------------|
| **Retriever** | `grep` (ripgrep / lexical), `vector` (embeddings) | Experiment 1 & 2 |
| **Delivery** | `inline` (hits in context), `file` (write JSON → read) | Inline vs programmatic |
| **Harness** | `react` (minimal tool loop) | Chronos / CLI harnesses (extensible) |

Run the full matrix:

```bash
pip install -e .
arm matrix -b examples/fixture/benchmark.json
```

Example output:

```
┏━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━┓
┃ Retriever ┃ Delivery ┃ Harness ┃ Accuracy ┃ N ┃
┡━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━┩
│ grep      │ inline   │ react   │ 100.0%   │ 3 │
│ grep      │ file     │ react   │ 66.7%    │ 3 │
└───────────┴──────────┴─────────┴──────────┴───┘
```

On the bundled fixture, **inline grep ≥ file grep** — matching the paper’s direction that delivery mechanics matter.

## Quick start

```bash
cd agentic-retrieval-matrix
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Full matrix (grep only, no vector extras)
arm matrix -b examples/fixture/benchmark.json

# Single cell (fast iteration)
arm single -b examples/fixture/benchmark.json -r grep -d inline

# With vector retriever
pip install -e ".[vector]"
arm matrix -b examples/fixture/benchmark.json --retrievers grep,vector
```

Results are written to `results/matrix_<timestamp>.json` with per-question traces (evidence turns, predicted vs gold).

## Design principles

1. **Memory as files** — conversation turns materialize to `turn_XXXX.txt` so grep is first-class (same mechanical advantage as agent CLIs).
2. **Delivery is not retrieval** — identical hits, different surfacing → different accuracy (paper’s inline vs file split).
3. **No API key required for v0** — the default harness uses an extractive baseline so CI and local runs work offline. Plug in your LLM harness when ready.
4. **Small core, clear extension points** — register retrievers, delivery channels, and harnesses without forking the runner.

## Project layout

```
src/agentic_retrieval_matrix/
  retrievers/     # grep (ripgrep), vector (MiniLM + FAISS)
  delivery/       # inline vs file-based tool results
  harness/        # ReAct-style minimal agent loop
  eval/           # loader, grader, matrix runner
examples/fixture/ # 3-question demo benchmark
scripts/          # LongMemEval conversion stub
```

## Connecting LongMemEval

The paper evaluates a **116-question** subset of [LongMemEval](https://github.com/xiaowu0162/longmemeval). ARM expects a normalized directory:

```
longmemeval_subset/
  questions.jsonl       # {"id","question","answer","category","session_id"}
  sessions/
    <session_id>.json   # {"session_id","turns":[...]}
```

1. Clone LongMemEval and download data per upstream instructions.
2. Implement `scripts/convert_longmemeval.py` mapping (stub provided).
3. Run:

```bash
arm longmem --data-dir examples/longmemeval_subset --retrievers grep,vector --deliveries inline,file
```

## Extending ARM

### Add a harness (e.g. LangGraph, provider CLI wrapper)

Implement `run(question, corpus, config, memory_root) -> AgentAnswer` and register in `harness/__init__.py`.

### Add LLM grading

Replace `eval/grader.py` or add `--llm-grader` using the `[llm]` extra (OpenAI-compatible).

### Noise sweep (Experiment 2)

Materialize distractor sessions into `session_noise.json` corpora and sweep `benchmark.json` — a `noise-mem-stress` mode is the planned v0.2 command (`arm noise --ratio 0.3`).

## Related work

- [Is Grep All You Need?](https://arxiv.org/abs/2605.15184) — harness × retrieval × delivery
- [LongMemEval](https://arxiv.org/abs/2410.10813) — benchmark dataset
- Hybrid agent memory systems (Mastra observational memory, virtual FS agents) — complementary, not duplicated here

## Citation

If you use ARM in research, cite the paper above and this repo:

```bibtex
@software{agentic_retrieval_matrix2026,
  title  = {Agentic Retrieval Matrix},
  year   = {2026},
  url    = {https://github.com/cobusgreyling/agentic-retrieval-matrix}
}
```

## License

MIT