# ═══════════════════════════════════════════════════════════════════════
# SYS_03 // VLLM SPECULATIVE SERVING MESH JUSTFILE
# Frontend: Vite + React 19 + TypeScript (Bun)
# Backend:  FastAPI + OpenAI Triton Kernels + Redis (Python uv)
# ═══════════════════════════════════════════════════════════════════════

default:
    @just --list

install:
    @echo "[+] Syncing Python backend dependencies with uv..."
    uv sync --extra dev
    @echo "[+] Installing React frontend dependencies with bun..."
    cd frontend && bun install

# Run backend independently on port 8003
backend host="0.0.0.0" port="8003":
    @echo "[*] Launching vLLM Speculative Serving Mesh on http://{{host}}:{{port}}..."
    uv run uvicorn src.server:app --host {{host}} --port {{port}} --reload

# Run frontend independently on port 5173 (with proxy to :8003)
frontend:
    @echo "[*] Launching Vite React frontend on http://localhost:5173..."
    cd frontend && bun run dev

# Run both backend and frontend concurrently
dev:
    @echo "[*] Launching Speculative Serving Mesh (Backend: :8003 | Frontend: :5173)..."
    (trap 'kill 0' SIGINT; just backend & just frontend)

# Build production bundle with Vite
build:
    @echo "[*] Building Vite frontend production bundle..."
    cd frontend && bun run build

# Run test suite
test:
    @echo "[*] Running backend pytest suite..."
    uv run pytest
    @echo "[*] Typechecking frontend..."
    cd frontend && bun run build

# Benchmark speculative decoding acceleration via CLI
benchmark tokens="64" draft="5":
    uv run mesh-cli benchmark -n {{tokens}} -k {{draft}}

# Inspect Redis semantic cache efficiency via CLI
cache-stats:
    uv run mesh-cli cache-stats

docker-build:
    docker build -t vllm-speculative-serving-mesh:latest .

docker-up:
    docker compose up -d

docker-down:
    docker compose down
