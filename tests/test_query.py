"""Tests for query module: prompts, engine (mock NIM)."""

import tempfile
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.query.prompts import SYSTEM_PROMPT, format_user_query
from src.query.engine import TelemetryQueryEngine
from data.synthetic.generate_telemetry import generate_telemetry, _ensure_full_schema


def test_system_prompt_contains_telemetry():
    assert "telemetry" in SYSTEM_PROMPT.lower()
    assert "ROS2" in SYSTEM_PROMPT or "NVIDIA" in SYSTEM_PROMPT


def test_format_user_query():
    ctx = "col1,col2\n1,2\n3,4"
    q = "What is the max of col2?"
    out = format_user_query(q, ctx, max_context_chars=1000)
    assert "col1,col2" in out
    assert "What is the max of col2?" in out


def test_format_user_query_truncation():
    ctx = "x" * 10000
    out = format_user_query("q", ctx, max_context_chars=100)
    assert "truncated" in out


def test_engine_retrieve():
    df = generate_telemetry(n_rows=200, vehicle_count=3)
    df = _ensure_full_schema(df)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    try:
        df.to_parquet(path, index=False)
        engine = TelemetryQueryEngine(path, max_context_rows=50, use_cudf=False)
        retrieved = engine.retrieve(vehicle_ids=["V000", "V001"])
        assert len(retrieved) <= 200
        if len(retrieved) > 0:
            assert set(retrieved["vehicle_id"].unique()).issubset({"V000", "V001"})
    finally:
        Path(path).unlink(missing_ok=True)
