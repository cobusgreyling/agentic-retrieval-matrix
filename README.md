<p align="center">
  <img src="images/arm-header.jpg" alt="Agentic Retrieval Matrix" width="100%">
</p>

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

Example output (baseline extractive harness on fixture):

```
┏━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━┓
┃ Retriever ┃ Delivery ┃ Harness ┃ Accuracy ┃ N ┃
┡━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━┩
│ grep      │ inline   │ react   │ 100.0%   │ 3 │
│ grep      │ file     │ react   │ 100.0%   │ 3 │
└───────────┴──────────┴─────────┴──────────┴───┘
```

The current `react` harness is a strong *extractive baseline* (no LLM) that is deliberately delivery-aware. This lets the matrix isolate retriever × delivery effects. Real agent harnesses (LLM ReAct/tool-use) are expected to show larger differences between inline and file delivery. See "Baseline & Limitations".

## Quick start

```bash
cd agentic-retrieval-matrix
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

arm --version

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
3. **No API key required for baseline** — the default harness uses an extractive baseline so CI and local runs work offline. Plug in your LLM harness when ready.
4. **Small core, clear extension points** — register retrievers, delivery channels, and harnesses without forking the runner.

## Baseline & Limitations (v0.2)

The bundled `react` harness is a *delivery-aware extractive oracle*:

- It always receives the raw `RetrievalResult` hits internally.
- For `file` delivery it reloads the written JSON (to simulate the extra step) but still uses gold-friendly span selection.
- Therefore accuracy gaps between `inline` and `file` are currently small or zero on simple fixtures.

This design choice makes the *retriever* and *delivery format* the primary variables under test while keeping the "reasoner" constant and reproducible without an LLM.

When you add a real LLM harness (via the `[llm]` extra or a custom `Harness`), you should expect:

- Larger `inline` vs `file` deltas (the agent must actually notice and act on the file path instruction).
- Sensitivity to prompt, tool-use formatting, and max_steps.

The fixture is deliberately tiny (3 questions). For research use LongMemEval or your own session corpus + a strong grader.

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

### Add a retriever

```python
# my_retriever.py
from pathlib import Path
from agentic_retrieval_matrix.retrievers.base import Retriever
from agentic_retrieval_matrix.types import MemoryCorpus, RetrievalResult, RunConfig, Hit

class MyRetriever(Retriever):
    kind = "my"

    def index(self, corpus: MemoryCorpus, memory_root: Path) -> None:
        ...

    def search(self, query: str, config: RunConfig) -> RetrievalResult:
        ...
```

Then register:

```python
# in your entry or a harness package
from agentic_retrieval_matrix.retrievers import RETRIEVER_REGISTRY
from my_retriever import MyRetriever
from agentic_retrieval_matrix.types import RetrieverKind

# extend the enum or use a string-based registry in future versions
RETRIEVER_REGISTRY[RetrieverKind("my")] = MyRetriever   # or add to enum
```

Similar pattern for `DeliveryChannel` and harnesses (implement the `run` method and add to `HARNESS_REGISTRY`).

### Add a harness (e.g. LangGraph, provider CLI wrapper)

Implement a class with `def run(self, question, corpus, config, memory_root) -> AgentAnswer` and register in `harness/__init__.py`.

### Add LLM grading

Replace `eval/grader.py` or extend the matrix runner to support `--llm-grader` (the `[llm]` extra provides OpenAI-compatible clients).

### Noise sweep (Experiment 2)

Materialize distractor sessions into `session_noise.json` corpora and sweep `benchmark.json`. A `noise` command is planned (`arm noise --ratio 0.3`). Contributions welcome.

### Development commands

```bash
pip install -e ".[dev]"
ruff check src tests
ruff format src tests
pytest -q
arm matrix -b examples/fixture/benchmark.json
```

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