"""Tests for ingest module: schema, synthetic data, cuDF loader."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.telemetry_schema import (
    TELEMETRY_SCHEMA,
    SENSOR_TYPES,
    IMU_SCHEMA,
    validate_schema,
    get_columns_for_sensor,
)
from src.ingest.cudf_loader import load_telemetry, filter_by_time_range, filter_by_vehicle
from src.ingest.benchmark_loader import run_benchmark, benchmark_to_dataframe
from data.synthetic.generate_telemetry import generate_telemetry, _ensure_full_schema


def test_schema_sensor_types():
    assert "imu" in SENSOR_TYPES
    assert "lidar" in SENSOR_TYPES
    assert "can" in SENSOR_TYPES
    assert "gps" in SENSOR_TYPES
    assert "camera" in SENSOR_TYPES


def test_get_columns_for_sensor():
    imu_cols = get_columns_for_sensor("imu")
    assert "accel_x" in imu_cols
    assert "gyro_x" in imu_cols
    can_cols = get_columns_for_sensor("can")
    assert "brake_pressure_pct" in can_cols


def test_validate_schema():
    df = pd.DataFrame({"timestamp_ns": [1], "vehicle_id": ["V001"], "accel_x": [0.0]})
    assert not validate_schema(df, TELEMETRY_SCHEMA)
    df_full = pd.DataFrame({k: [0] for k in TELEMETRY_SCHEMA.keys()})
    assert validate_schema(df_full, TELEMETRY_SCHEMA)


def test_generate_telemetry():
    df = generate_telemetry(n_rows=1000, vehicle_count=3, seed=42)
    assert len(df) == 1000
    assert "timestamp_ns" in df.columns
    assert "vehicle_id" in df.columns
    assert "sensor_type" in df.columns
    assert set(df["sensor_type"].unique()).issubset({"imu", "lidar", "can", "gps", "camera"})


def test_ensure_full_schema():
    df = generate_telemetry(n_rows=100, vehicle_count=2)
    df = _ensure_full_schema(df)
    assert "latitude" in df.columns
    assert "brake_pressure_pct" in df.columns


def test_load_telemetry_pandas():
    df = generate_telemetry(n_rows=500, vehicle_count=2)
    df = _ensure_full_schema(df)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    try:
        df.to_parquet(path, index=False)
        loaded = load_telemetry(path, use_cudf=False)
        assert len(loaded) == 500
        assert set(loaded.columns) == set(df.columns)
    finally:
        Path(path).unlink(missing_ok=True)


def test_filter_by_time_range():
    df = pd.DataFrame({
        "timestamp_ns": [100, 200, 300, 400, 500],
        "vehicle_id": ["V1"] * 5,
    })
    filtered = filter_by_time_range(df, start_ns=200, end_ns=400)
    assert len(filtered) == 3
    assert filtered["timestamp_ns"].min() >= 200
    assert filtered["timestamp_ns"].max() <= 400


def test_filter_by_vehicle():
    df = pd.DataFrame({
        "vehicle_id": ["V1", "V2", "V1", "V3", "V2"],
        "x": [1, 2, 3, 4, 5],
    })
    filtered = filter_by_vehicle(df, ["V1", "V3"])
    assert len(filtered) == 3
    assert set(filtered["vehicle_id"].unique()) == {"V1", "V3"}


def test_run_benchmark():
    """Benchmark runs and returns pandas at minimum; cuDF/cudf.pandas when available."""
    df = generate_telemetry(n_rows=1000, vehicle_count=3)
    df = _ensure_full_schema(df)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    try:
        df.to_parquet(path, index=False)
        results = run_benchmark(path)
        assert "pandas" in results
        assert all(op in results["pandas"] for op in ["load", "groupby", "filter", "sort"])
        bm_df = benchmark_to_dataframe(results)
        assert "backend" in bm_df.columns and "operation" in bm_df.columns
        assert len(bm_df[bm_df["backend"] == "pandas"]) == 4
    finally:
        Path(path).unlink(missing_ok=True)
