import pytest
import torch
from fastapi.testclient import TestClient

from src.cache.semantic_cache import SemanticCache
from src.server import app
from src.speculative.engine import SpeculativeDecodingEngine


@pytest.fixture
def client():
    return TestClient(app)


def test_speculative_rejection_sampling():
    engine = SpeculativeDecodingEngine(num_speculative_tokens=3)

    # Perfect match between draft and target
    target_probs = torch.tensor(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.95, 0.05],
        ]
    )
    draft_probs = torch.tensor(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.95, 0.05],
        ]
    )
    draft_tokens = [0, 0, 0]

    accepted, count = engine.verify_draft_tokens(target_probs, draft_probs, draft_tokens)
    assert count == 3
    assert accepted == [0, 0, 0]


def test_semantic_cache():
    cache = SemanticCache()
    prompt = "What is the speedup of speculative decoding?"
    response = "Speculative decoding delivers a 2.5x to 3.2x latency speedup."

    assert cache.get(prompt) is None
    cache.set(prompt, response)
    assert cache.get(prompt) == response


def test_api_completions(client):
    payload = {
        "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 16,
        "stream": False,
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "choices" in data
    assert "speculative_telemetry" in data
    assert data["speculative_telemetry"]["acceptance_rate"] > 0.0


def test_api_metrics(client):
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "llm_serving_requests_total" in res.text
