"""Prometheus Real-Time Serving Metrics Exporter."""

from __future__ import annotations

import logging

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

logger = logging.getLogger("serving.metrics")

# Prometheus Metric Definitions
REQUEST_COUNT = Counter(
    "llm_serving_requests_total", "Total incoming LLM inference requests", ["model", "endpoint"]
)

CACHE_HITS = Counter("llm_serving_cache_hits_total", "Total requests fulfilled via Semantic Cache")

TTFT_HISTOGRAM = Histogram(
    "llm_serving_ttft_milliseconds",
    "Time to first token (TTFT) distribution in milliseconds",
    buckets=[5, 10, 20, 35, 50, 75, 100, 200, 500],
)

ITL_HISTOGRAM = Histogram(
    "llm_serving_itl_milliseconds",
    "Inter-token latency (ITL) distribution in milliseconds",
    buckets=[2, 5, 8, 12, 15, 20, 30, 50],
)

SPECULATIVE_ACCEPTANCE_RATE = Gauge(
    "llm_serving_speculative_acceptance_rate",
    "Ratio of draft model tokens accepted by the target model",
)


class PrometheusMetricsExporter:
    """Manages telemetry recording and Prometheus metrics endpoint."""

    @staticmethod
    def record_request(model: str = "llama-3.1-70b", endpoint: str = "/v1/chat/completions"):
        REQUEST_COUNT.labels(model=model, endpoint=endpoint).inc()

    @staticmethod
    def record_cache_hit():
        CACHE_HITS.inc()

    @staticmethod
    def record_latencies(ttft_ms: float, itl_ms: float, acceptance_rate: float):
        TTFT_HISTOGRAM.observe(ttft_ms)
        ITL_HISTOGRAM.observe(itl_ms)
        SPECULATIVE_ACCEPTANCE_RATE.set(acceptance_rate)

    @staticmethod
    def get_metrics_payload() -> tuple[bytes, str]:
        return generate_latest(), CONTENT_TYPE_LATEST
