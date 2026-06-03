# Getting Started

This project depends on **Temporal-Light**, which is included as a **git submodule**
(`./Temporal-Light`) and installed as a local editable package. A plain `git clone`
leaves that directory **empty**, which is the most common reason a fresh setup fails.

The main flow below uses **uv** (recommended). If you'd rather use pip, skip to
[Using pip instead](#using-pip-instead).

## Prerequisites

- Git
- Python 3.10+
- One of: [uv](https://docs.astral.sh/uv/) **or** pip
- Docker - only needed to run the full pipeline, **not** to run the tests

## Setup (uv)

```bash
# 1. Clone with the submodule (this is the step people miss)
git clone --recurse-submodules https://github.com/BertilBraun/Durable-Agentic-Coding-Runtime.git
cd Durable-Agentic-Coding-Runtime

# 2. Install everything: project, temporal-light (from the submodule), dev + eval extras
uv sync --extra dev --extra eval

# 3. Create your env file
cp .env.example .env        # Windows: Copy-Item .env.example .env

# 4. Run the tests
uv run pytest
```

Already cloned without `--recurse-submodules`? `Temporal-Light/` will be empty - pull it in
with `git submodule update --init --recursive`, then re-run `uv sync`.

Expected test result: roughly **152 passed, 1 skipped**. The skipped test is a host-workspace
integration test gated behind `RUN_HOST_TESTS=1`. Tests marked `integration` (Docker /
live Temporal-Light) are deselected by default and are not required for a green run.

Don't need the eval extras? Drop `--extra eval` (or use `uv sync --extra dev`).

## Running the smoke workflow

This runs the real end-to-end pipeline against a live Temporal-Light stack, so it needs
`LLM_API_KEY` set in `.env`.

```bash
(cd Temporal-Light && docker compose up -d) && uv run python -m src.eval.smoke_workflow
```

---

## Using pip instead

A complete alternative to the uv flow above. pip does not read `[tool.uv.sources]`, so the
submodule must be installed **first** — that satisfies the `temporal-light` dependency
locally and stops pip from searching PyPI for it.

```bash
# 1. Clone with the submodule
git clone --recurse-submodules https://github.com/BertilBraun/Durable-Agentic-Coding-Runtime.git
cd Durable-Agentic-Coding-Runtime

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1

# 3. Install the submodule first, then the project + dev extras
pip install -e ./Temporal-Light      # must come before the next line
pip install -e ".[dev]"              # or ".[dev,eval]" to include the eval extras

# 4. Create your env file and run the tests
cp .env.example .env                 # Windows: Copy-Item .env.example .env
pytest
```

To run the smoke workflow (with `.venv` activated and `LLM_API_KEY` set in `.env`):

```bash
(cd Temporal-Light && docker compose up -d) && python -m src.eval.smoke_workflow
```
