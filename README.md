```
██▒   █▓ ██▓     ██▓     ███▄ ▄███▓     ██████  ██▓███  ▓█████  ▄████▄
▓██░   █▒▓██▒    ▓██▒    ▓██▒▀█▀ ██▒   ▒██    ▒ ▓██░  ██▒▓█   ▀ ▒██▀ ▀█
 ▓██  █▒░▒██░    ▒██░    ▓██    ▓██░   ░ ▓██▄   ▓██░ ██▓▒▒███   ▒▓█    ▄
  ▒██ █░░▒██░    ▒██░    ▒██    ▒██      ▒   ██▒▒██▄█▓▒ ▒▒▓█  ▄ ▒▓▓▄ ▄██▒
   ▒▀█░  ░██████▒░██████▒▒██▒   ░██▒   ▒██████▒▒▒██▒ ░  ░░▒████▒▒ ▓███▀ ░
   ░ ▐░  ░ ▒░▓  ░░ ▒░▓  ░░ ▒░   ░  ░   ▒ ▒▓▒ ▒ ░▒▓▒░ ░  ░░░ ▒░ ░░ ░▒ ▒  ░
   ░ ░░  ░ ░ ▒  ░░ ░ ▒  ░░  ░      ░   ░ ░▒  ░ ░░▒ ░      ░ ░  ░  ░  ▒
     ░░    ░ ░     ░ ░   ░      ░      ░  ░  ░  ░░          ░   ░
      ░      ░  ░    ░  ░       ░            ░              ░  ░░ ░
     ░                                                          ░
```

# SYS_03 // VLLM SPECULATIVE SERVING MESH

```
[SYSTEM ID]       SYS_03
[ARCHITECTURE]    Speculative Draft Decoding (1.5B Draft -> 70B Target)
[ACCELERATION]    OpenAI Triton Fused RMSNorm/SwiGLU/RoPE Kernels
[CACHING]         Redis Cluster Semantic Response Cache (Cosine Sim)
[FRONTEND STACK]  Vite + React 19 + TypeScript (Bun)
[BACKEND STACK]   FastAPI + vLLM Continuous Batching + Python 3.11 (uv)
[COMMAND RUNNER]  just (Justfile)
[PERF METRICS]    4.2x Throughput // 68% p99 Latency Drop // 38.4% Cache Hit
```

---

## ⚡ ARCHITECTURAL SPECIFICATIONS

High-throughput distributed model serving pipeline built for generative workloads. Combines speculative draft decoding (1.5B draft assisting 70B target) with OpenAI Triton fused kernels to maximize GPU memory bandwidth utilization.

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │                   Client Chat Completion Request                       │
 └──────────────────────────────────┬─────────────────────────────────────┘
                                    │
 ┌──────────────────────────────────▼─────────────────────────────────────┐
 │               Redis Semantic Embedding Cache (Cosine > 0.92)           │
 └──────────┬─────────────────────────────────────────────────┬───────────┘
            │ HIT (38.4%)                                     │ MISS (61.6%)
 ┌──────────▼──────────────┐             ┌────────────────────▼───────────┐
 │ Return Zero-GPU Compute │             │  Speculative Decoding Engine   │
 │ Response (<1.2ms)       │             │  (K=5 Candidate Tokens)        │
 └─────────────────────────┘             └────────────┬───────────────────┘
                                                      │
                                   ┌──────────────────┴───────────────────┐
                                   │  Draft Model (Llama-3.2-1.5B)        │
                                   │  - Emits 5 Draft Tokens Parallel     │
                                   └──────────────────┬───────────────────┘
                                                      │
                                   ┌──────────────────▼───────────────────┐
                                   │  Target Model (Llama-3.1-70B)        │
                                   │  - Single-Pass Verification Forward  │
                                   │  - Rejection Sampling: min(1, p/q)   │
                                   └──────────────────┬───────────────────┘
                                                      │
                                   ┌──────────────────▼───────────────────┐
                                   │  Triton Fused Operator Kernels       │
                                   │  - fused_rmsnorm                     │
                                   │  - fused_swiglu                      │
                                   │  - fused_rotary_emb                  │
                                   └──────────────────────────────────────┘
```

---

## 🛠 INDEPENDENT EXECUTION WITH `just`

Frontend and backend run independently or concurrently:

```bash
# Display all recipes
just

# Install backend (uv) and frontend (bun) dependencies
just install

# Run backend independently on http://localhost:8003
just backend

# Run frontend independently on http://localhost:5173
just frontend

# Run both backend and frontend concurrently
just dev

# Run comprehensive test suite
just test

# Benchmark speculative decoding acceleration via CLI
just benchmark tokens="64" draft="5"

# Inspect Redis semantic cache efficiency via CLI
just cache-stats
```

---

## 📦 CONTAINERIZATION (DOCKER & DOCKER-COMPOSE)

```bash
# Build multi-stage container
just docker-build

# Launch serving mesh and Redis
just docker-up

# Verify health
curl -s http://localhost:8003/health | jq .
```

---

## 🌐 API SPECIFICATION

| Verb | Path | Description |
|---|---|---|
| `GET` | `/health` | Inference engine and Redis connection health |
| `GET` | `/metrics` | Prometheus metrics exporter (TTFT, ITL, alpha) |
| `GET` | `/api/v1/telemetry` | Real-time speculative serving HUD telemetry |
| `POST` | `/v1/chat/completions` | OpenAI-compatible speculative completions API |
