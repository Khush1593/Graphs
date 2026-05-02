from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List

import duckdb
import pandas as pd


@dataclass
class SchemaInfo:
    columns: List[str]
    dtypes: Dict[str, str]
    date_column: str | None
    numeric_column: str | None
    category_column: str | None


def create_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(database=":memory:")


def load_csv_to_duckdb(con: duckdb.DuckDBPyConnection, uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        raise ValueError("No file uploaded.")

    if not uploaded_file.name.lower().endswith(".csv"):
        raise ValueError("Only CSV files are supported in MVP.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp:
        temp.write(uploaded_file.getbuffer())
        temp_path = temp.name

    try:
        con.execute("DROP TABLE IF EXISTS data")
        con.execute("CREATE TABLE data AS SELECT * FROM read_csv_auto(?)", [temp_path])
        df = con.execute("SELECT * FROM data").df()
    finally:
        os.remove(temp_path)

    if df.empty:
        raise ValueError("Uploaded CSV has no rows.")

    return df


def detect_schema(df: pd.DataFrame) -> SchemaInfo:
    columns = list(df.columns)
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    date_column = None
    numeric_column = None
    category_column = None

    lower_columns = {c.lower(): c for c in columns}

    for col in columns:
        if "date" in col.lower() or "time" in col.lower():
            date_column = col
            break

    if date_column is None:
        for col in columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.8:
                date_column = col
                break

    numeric_candidates = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_candidates:
        preferred_names = ["revenue", "sales", "amount", "price", "total"]
        for name in preferred_names:
            for col in numeric_candidates:
                if name in col.lower():
                    numeric_column = col
                    break
            if numeric_column:
                break
        if numeric_column is None:
            numeric_column = numeric_candidates[0]

    object_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for col in object_cols:
        unique_ratio = df[col].nunique(dropna=True) / max(len(df), 1)
        if 0 < unique_ratio < 0.7:
            category_column = col
            break

    if category_column is None and object_cols:
        category_column = object_cols[0]

    return SchemaInfo(
        columns=columns,
        dtypes=dtypes,
        date_column=date_column,
        numeric_column=numeric_column,
        category_column=category_column,
    )


def execute_select_query(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    if not sql.strip().lower().startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")
    return con.execute(sql).df()
