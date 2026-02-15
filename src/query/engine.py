"""
Telemetry query engine: cuDF retrieval + NIM natural-language summarization.
"""

from pathlib import Path
from typing import Any, Optional, Union

from .prompts import SYSTEM_PROMPT, format_user_query

try:
    import cudf

    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False

import pandas as pd


class TelemetryQueryEngine:
    """
    Orchestrates: load telemetry with cuDF, filter/aggregate, format context,
    send to NIM for natural-language answer.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        nim_base_url: str = "http://localhost:8000",
        nim_model: str = "meta/llama3-8b-instruct",
        max_context_rows: int = 1000,
        use_cudf: Optional[bool] = None,
    ):
        self.data_path = Path(data_path)
        self.nim_base_url = nim_base_url
        self.nim_model = nim_model
        self.max_context_rows = max_context_rows
        self.use_cudf = use_cudf if use_cudf is not None else CUDF_AVAILABLE

        self._df: Optional[Union[pd.DataFrame, "cudf.DataFrame"]] = None

    def _load_data(self) -> Union[pd.DataFrame, "cudf.DataFrame"]:
        """Load telemetry with cuDF or pandas."""
        from ..ingest.cudf_loader import load_telemetry

        return load_telemetry(
            self.data_path,
            spill=True,
            use_cudf=self.use_cudf,
        )

    def _ensure_loaded(self) -> None:
        if self._df is None:
            self._df = self._load_data()

    def _data_to_context(
        self,
        df: Union[pd.DataFrame, "cudf.DataFrame"],
        max_rows: Optional[int] = None,
    ) -> str:
        """Convert dataframe slice to string context for prompt."""
        n = max_rows or self.max_context_rows
        subset = df.head(n)
        if hasattr(subset, "to_pandas"):
            subset = subset.to_pandas()
        return subset.to_string(max_rows=n)

    def retrieve(
        self,
        vehicle_ids: Optional[list[str]] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
        sensor_type: Optional[str] = None,
        brake_threshold: Optional[float] = None,
    ) -> Union[pd.DataFrame, "cudf.DataFrame"]:
        """
        Retrieve telemetry with filters. Uses cuDF when available.
        """
        from ..ingest.cudf_loader import (
            filter_by_time_range,
            filter_by_vehicle,
            get_anomaly_windows,
        )

        self._ensure_loaded()
        df = self._df

        if vehicle_ids:
            df = filter_by_vehicle(df, vehicle_ids)
        if start_ns is not None or end_ns is not None:
            df = filter_by_time_range(df, start_ns, end_ns)
        if sensor_type and "sensor_type" in df.columns:
            df = df[df["sensor_type"] == sensor_type]
        if brake_threshold is not None and "brake_pressure_pct" in df.columns:
            df = df[df["brake_pressure_pct"] > brake_threshold]

        return df

    def query(
        self,
        user_query: str,
        vehicle_ids: Optional[list[str]] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
        sensor_type: Optional[str] = None,
    ) -> str:
        """
        Execute natural-language query over telemetry.

        1. Retrieve data with filters (inferred from query or explicit).
        2. Format as context.
        3. Send to NIM for summarization.
        """
        df = self.retrieve(
            vehicle_ids=vehicle_ids,
            start_ns=start_ns,
            end_ns=end_ns,
            sensor_type=sensor_type,
        )
        context = self._data_to_context(df)

        from ..deploy.nim_client import NIMClient

        client = NIMClient(
            base_url=self.nim_base_url,
            model=self.nim_model,
        )
        user_msg = format_user_query(user_query, context)
        return client.ask(user_msg, system_context=SYSTEM_PROMPT)
