"""Tests for inference module: format selector, pipeline, metrics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.format_selector import select_format, LIFECYCLE_TABLE, FORMAT_RATIONALE
from src.inference.metrics import pct, compute_metrics


def test_select_format_production():
    fmt, _ = select_format("production", hardware="gpu")
    assert fmt == "tensorrt"


def test_select_format_local():
    fmt, _ = select_format("local", hardware="gpu")
    assert fmt == "gguf"


def test_select_format_sharing():
    fmt, _ = select_format("sharing", hardware="gpu")
    assert fmt == "safetensors"


def test_select_format_portable():
    fmt, _ = select_format("portable", hardware="mixed")
    assert fmt == "onnx"


def test_format_rationale_keys():
    assert "safetensors" in FORMAT_RATIONALE
    assert "gguf" in FORMAT_RATIONALE
    assert "tensorrt" in FORMAT_RATIONALE
    assert "onnx" in FORMAT_RATIONALE


def test_lifecycle_table():
    assert LIFECYCLE_TABLE["research"] == "safetensors"
    assert LIFECYCLE_TABLE["production"] == "tensorrt"


def test_pct():
    assert pct([1, 2, 3, 4, 5], 50) == 3.0
    import math
    assert math.isnan(pct([], 50))


def test_compute_metrics():
    latencies = [1.0, 1.1, 1.2, 1.3, 1.4]
    ttft = [0.1, 0.11, 0.12, 0.13, 0.14]
    token_counts = [50] * 5
    m = compute_metrics(
        total_latencies=latencies,
        first_token_latencies=ttft,
        token_counts=token_counts,
    )
    assert "p50_latency_s" in m
    assert "p90_latency_s" in m
    assert "p50_ttft_s" in m
    assert "throughput_sustained_tok_s" in m
