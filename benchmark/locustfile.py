"""Locust Load Test Suite for Streaming Speculative LLM Serving Mesh."""

from locust import HttpUser, between, task


class LLMStreamingUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(3)
    def test_streaming_chat(self):
        payload = {
            "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": "Explain speculative decoding with draft verification in LLMs.",
                }
            ],
            "max_tokens": 30,
            "stream": True,
        }
        with self.client.post(
            "/v1/chat/completions", json=payload, stream=True, catch_response=True
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(1)
    def test_cached_query(self):
        # Repeated prompt to measure Semantic Cache <2ms response time
        payload = {
            "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "messages": [{"role": "user", "content": "What is Triton GPU programming?"}],
            "max_tokens": 20,
            "stream": False,
        }
        self.client.post("/v1/chat/completions", json=payload)
