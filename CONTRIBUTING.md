# Contributing

ARM is intentionally small. Preferred contributions:

1. **LongMemEval converter** — flesh out `scripts/convert_longmemeval.py`
2. **Harness adapters** — LangGraph, OpenAI Agents SDK, provider CLIs
3. **LLM grader** — optional parity with paper’s GPT-4o grader
4. **Noise sweep CLI** — Experiment 2 distractor injection

Before opening a PR:

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
```

Keep changes scoped to one axis (retriever, delivery, or harness) per PR when possible.