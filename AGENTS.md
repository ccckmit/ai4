# AGENTS.md

## Tri-lingual polyglot repo

Every module (`world`, `nn`, `ml`, `llm`) has implementations in **Python**, **TypeScript**, and **Rust** (parallel files like `core.py` / `core.ts` / `core.rs`).

## Package structure

| Dir | What | Python import | TS import (from `dist/`) |
|-----|------|---------------|--------------------------|
| `world/` | RL envs (Gym-style) | `import world` | `import { ... } from 'ai4/world'` |
| `nn/` | Neural nets + autodiff | `import nn` | `import { ... } from 'ai4/nn'` |
| `ml/` | sklearn-style ML toolkit | `import ml` | `import { ... } from 'ai4/ml'` |
| `llm/` | LLM agent (Ollama-based) | `import llm` | `import { ... } from 'ai4/llm'` |

- `world` and `nn` re-exported from `ai4`: `from ai4 import world` / `from ai4 import nn`
- `ml` and `llm` NOT re-exported — use `PYTHONPATH=. import ml` / `import llm`
- TS entry points are `*/index.ts` (not `__init__.ts`). `llm/index.ts` exports `{}` — nothing is importable from `ai4/llm` in TS.

## Python

**`uv run python` / `uv run pytest` preferred. `PYTHONPATH=. python` / `PYTHONPATH=. pytest` also works.**
Scripts (`pytest.sh`, `pyrun.sh`) auto-set `PYTHONPATH=.`.

```bash
./pytest.sh                              # all Python tests (all 4 modules)
uv run pytest world/tests                # single module
uv run pytest nn/tests/test_tensor.py    # single file
```

Pytest test paths: `world/tests`, `nn/tests`, `ml/tests` (from `pyproject.toml`).
`llm/tests` is NOT in `pyproject.toml` testpaths but `pytest.sh` runs it explicitly.

Important — `nn.chargpt_demo` uses relative imports, must run as module:
```bash
python -m nn.chargpt_demo    # correct
python nn/chargpt_demo.py    # FAILS
```

## TypeScript — two test runners

| Runner | Pattern | Example |
|--------|---------|---------|
| `npx tsx` | bare `.ts` (no `.test.ts`) | `npx tsx world/tests/test_world.ts` |
| `npx jest` | `**/tests/**/*.test.ts` | `npx jest nn/tests/test_tensor --no-coverage` |

`npx tsx` files use plain `import` + inline assert-style checks.
`npx jest` files use `describe`/`test`/`expect`.

```bash
./jstest.sh                    # runs ALL (see jstest.sh for order)
npx tsx nn/tests/test_nn.ts    # single bare test
npx jest nn/tests/test_tensor  # single jest test (--no-coverage to skip coverage)
```

Build before importing from `dist/`:
```bash
npx tsc                  # outputs to dist/ (ESM, moduleResolution: bundler)
```

`tsconfig.json` excludes `test*.ts`, `*_test.ts`, and `examples/*.ts` from compilation.

## Rust

```bash
cargo test --lib         # all Rust tests (alias: ./rstest.sh)
cargo run --bin <name>   # Cargo.toml defines all bins
```

Rust **test files** live in each module's `tests.rs` or standalone files, integrated via `mod.rs` `#[cfg(test)]` blocks — they're NOT standalone crates.

Rust **binary targets** are at arbitrary paths (not `src/bin/`). See `Cargo.toml` `[[bin]]` entries: `world_frozen_lake_example`, `world_frozenlake_qtable`, `world_cartpole_closed_form`, `nn_example`, `ml_example`, `chargpt_demo`, `mnist_train`.

`lib.rs` uses `#![allow(dead_code, unused, non_snake_case, private_interfaces)]`.

## Key scripts

| Script | Action |
|--------|--------|
| `./pytest.sh` | All Python tests (all 4 modules) |
| `./jstest.sh` | All TS tests (both runners) |
| `./rstest.sh` | `cargo test --lib` |
| `./pyrun.sh` | Run Python examples |
| `./jsrun.sh` | Run TS examples |
| `./rsrun.sh` | Run all Rust binaries |
| `./cnntest.sh` | `uv run pytest nn/tests/test_cnn_gemini.py` |
| `./mnist_run.sh` | Python MNIST training (hardcoded venv path, requires torch) |
| `./pypub.sh` | PyPI publish (dry-run/test/prod) |

## Conventions & quirks

- **`StepResult`** dataclass unpacks as 5-tuple: `obs, reward, terminated, truncated, info`
- **`nn.datasets`** (DataLoader, MNIST transforms) guarded by `try/except ImportError` — only available when torch is installed. TS/Rust versions exist but TS not exported from `nn/index.ts`.
- **BipedalWalker-v3** is only registered in Python (not in TS or Rust).
- **Python code comments**: English. Theory docs: Traditional Chinese (`_wiki/`, `*.md`).
- **`llm/tests/test_agent`**: requires Ollama running locally. Test silently returns on failure (`2>/dev/null || true` in `jstest.sh`).
- **PyPI publish**: `python -m build` + `twine`. See `pypub.sh`.
- **Crates.io publish**: `cargo login && cargo publish` (see `rspub.sh`).
- `checkpoints_ai4/` stores saved model weights (`.npz` files from SAC training).
