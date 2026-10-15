# ==============================================================================
# AGENTS.MD — VLLM SPECULATIVE SERVING MESH (SYS_03)
# Production Architecture, Rules & Operational Directives
# Author: Bernard Birendra Das (https://github.com/bernardbdas)
# ==============================================================================

> **MANDATORY SPECIFICATION FOR AUTONOMOUS AGENTS AND DEVELOPERS**
> This repository strictly adheres to the engineering standards, architecture, and design styles specified herein. No simplifications, removals, or degradations are permitted.

---

## 1. System Topology & Identity

- **System ID**: `SYS_03`
- **Repository Name**: `vllm-speculative-serving-mesh`
- **Primary Domain**: High-Throughput Speculative Decoding LLM Inference Mesh with Triton Fused Kernels & Semantic Response Cache
- **Backend Port**: `8003` (FastAPI with OpenAI-compatible completions API)
- **Frontend Framework**: Vite + React 19 + TypeScript (Scaffolded with `bun`)
- **Frontend Port (Dev)**: `5173`
- **UI Aesthetic Style**: **Neon Cyberpunk HUD / High-Performance Telemetry Terminal** (Inspired by Bernard's portfolio `bernardbdas.github.io` & UXPlanet HUD Brutalism)
  - Color Palette: Deep obsidian `#050508`, neon cyan `#00f0ff`, laser magenta `#f43f5e`, volt yellow `#eab308`, hairline borders `#181825`
  - Typography: Monospace (`JetBrains Mono`, `IBM Plex Mono`)
  - Layout: High-density real-time LLM inference cockpit with live streaming token output, speculative acceptance rate gauge (alpha = 0.82), TTFT and ITL histograms, and semantic cache hit rate monitor
- **Domain CLI**: `mesh-cli` (`src/cli.py`)

---

## 2. Decoupled Architecture & Execution Directives

1. **Decoupled Independence**:
   - Backend and frontend are completely decoupled and must run independently.
   - Run backend independently: `just backend` (FastAPI on port `8003`).
   - Run frontend independently: `just frontend` (Vite dev server on port `5173`).
   - Run both concurrently: `just dev`.
2. **Production Serving**:
   - Compiling frontend via `just build` writes to `frontend/dist`.
   - In production, FastAPI statically mounts `frontend/dist` at `/`.
   - OpenAI streaming endpoints (`/v1/completions`, `/telemetry`, `/metrics`, `/cache/stats`) always take routing priority.

---

## 3. Project Management with `uv`

1. **UV Enforcement**: All Python environment management, locking, and execution MUST use `uv`.
2. **Build System**: Managed via `hatchling` in `pyproject.toml`:
   ```toml
   [build-system]
   requires = ["hatchling"]
   build-backend = "hatchling.build"

   [tool.hatch.build.targets.wheel]
   packages = ["src"]
   ```
3. **Cross-Platform Compatibility**: Linux-specific packages (e.g. `vllm`, `sglang`, `triton`, `flash-attn`) must declare platform markers:
   ```toml
   dependencies = [
       "vllm>=0.6.0; sys_platform == 'linux'",
       "sglang>=0.3.0; sys_platform == 'linux'",
       "triton>=3.0.0; sys_platform == 'linux'",
   ]
   ```
   Ensuring pure macOS local development and testing alongside high-performance GPU cluster deployments.

---

## 4. Frontend & UI Design Rules (`bun` & Neon Cyberpunk HUD)

1. Managed exclusively using `bun` inside `frontend/`.
2. Neon Cyberpunk HUD styling:
   - High-density monospace ribbons, ASCII banners, and live SSE token generation stream.
   - Real-time speculative acceptance rate visualizer (draft proposals vs. verifications).
   - Redis semantic response cache performance telemetry (<1.2ms lookup, >38% GPU compute saved).

---

## 5. Command Runner (`just`) Specification

Standardized `Justfile` tasks:
- `just backend`: Run FastAPI server (`http://localhost:8003`) with reload.
- `just frontend`: Run Bun frontend dev server.
- `just dev`: Launch backend and frontend concurrently.
- `just install`: Install backend with `uv sync --extra dev` and frontend with `bun install`.
- `just test`: Run `uv run pytest` and frontend TypeScript check.
- `just build`: Compile frontend to `dist/` and build Python wheel.
- `just check`: Run `uv run ruff check src` and format checks.
- `just bench`: Run speculative decoding benchmark and latency measurement.
- `just cache-stats`: Inspect Redis semantic response cache performance.
- `just docker-build`: Build multi-stage production Docker image.
- `just docker-up`: Start serving mesh and Redis cache via Docker Compose.

---

## 6. Containerization & Docker

- Multi-stage `Dockerfile`:
  - Stage 1: `oven/bun:1-alpine` compiles `frontend/dist`.
  - Stage 2: `python:3.11-slim` installs dependencies via `uv sync --frozen --no-dev` and hosts FastAPI.
- `docker-compose.yml`: Spins up `vllm-speculative-serving-mesh` and `redis`.

---

## 7. Developer Tooling & `.vscode` Setup

- `.vscode/settings.json`: Configured with Ruff default formatter, format on save, and `.venv` python interpreter path.
- `.vscode/launch.json`: Debugging configurations for FastAPI backend, CLI commands, and Chrome frontend.
- `.vscode/extensions.json`: Recommended VS Code extensions.
- `.vscode/tasks.json`: Standard VS Code build and test tasks wired to `just`.

---

## 8. Environment Management Rules

- `example.env`: Tracked template with descriptions for target model, draft model, speculative K, and Redis URI.
- `local.env`: Tracked developer defaults for immediate out-of-the-box local execution.
- `.env` / `.env.local`: Ignored in `.gitignore` for private inference credentials.

---

## 9. Code Quality, Linters & Pre-Commit Hooks

- Linter & Formatter: `ruff` with `select = ["E", "F", "I"]`, `line-length = 100`.
- Pre-commit hooks (`.pre-commit-config.yaml`):
  - `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-toml`, `check-added-large-files` (`--maxkb=5000`), `ruff`, and `ruff-format`.
- All git commits must be verified against pre-commit hooks.

---

## 10. Continuous Integration (CI)

- `.github/workflows/ci.yml` runs on every push and PR to `main`:
  - Python lint (`ruff check src`) and test (`pytest`).
  - Bun frontend typecheck (`tsc -b`) and build (`bun run build`).

---

## 11. Git & Remote Repository Rules

- Branch: `main`.
- Remote: `https://github.com/bernardbdas/vllm-speculative-serving-mesh.git`.
