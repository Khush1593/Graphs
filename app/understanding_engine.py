from __future__ import annotations

import warnings
from typing import Any

import pandas as pd

from app.config import settings
from app.data_engine import SchemaInfo
from app.llm_engine import generate_dataset_understanding, resolve_active_model


SAMPLE_ROW_COUNT = 5
TOP_VALUE_COUNT = 5

IDENTIFIER_UNIQUE_RATIO = 0.95

CARDINALITY_LOW = 10
CARDINALITY_MEDIUM = 100
CARDINALITY_HIGH = 1000


def _cardinality_bucket(unique_count: int) -> str:
    if unique_count <= CARDINALITY_LOW:
        return "low"
    if unique_count <= CARDINALITY_MEDIUM:
        return "medium"
    if unique_count <= CARDINALITY_HIGH:
        return "high"
    return "very_high"


def _suitable_chart_types_for_dimension(cardinality_bucket: str) -> list[str]:
    if cardinality_bucket == "low":
        return ["bar", "pie", "stacked_bar"]
    if cardinality_bucket == "medium":
        return ["bar", "treemap"]
    if cardinality_bucket == "high":
        return ["top_n_bar", "treemap"]
    return ["top_n_bar"]


def _aggregation_hint(column_name: str) -> str:
    name = column_name.lower()
    if any(token in name for token in ["price", "rate", "ratio", "avg", "mean", "score"]):
        return "avg"
    if any(token in name for token in ["count", "qty", "quantity", "units"]):
        return "sum"
    return "sum"


def _build_column_profiles(df: pd.DataFrame) -> list[dict[str, Any]]:
    row_count = max(len(df), 1)
    profiles: list[dict[str, Any]] = []
    for col in df.columns:
        col_data = df[col]
        unique_count = int(col_data.nunique(dropna=True))
        unique_ratio = unique_count / row_count
        profile: dict[str, Any] = {
            "name": col,
            "dtype": str(col_data.dtype),
            "non_null_count": int(col_data.notna().sum()),
            "null_pct": round(float(col_data.isna().mean()) * 100.0, 2),
            "unique_count": unique_count,
            "unique_ratio": round(unique_ratio, 4),
            "cardinality_bucket": _cardinality_bucket(unique_count),
            "is_identifier_candidate": unique_ratio >= IDENTIFIER_UNIQUE_RATIO,
            "is_numeric": bool(pd.api.types.is_numeric_dtype(col_data)),
            "is_datetime_parseable": False,
        }

        if profile["is_numeric"] and col_data.notna().any():
            profile["min"] = round(float(col_data.min()), 4)
            profile["max"] = round(float(col_data.max()), 4)
            profile["mean"] = round(float(col_data.mean()), 4)
            profile["zero_count"] = int((col_data.fillna(0) == 0).sum())
            profile["negative_count"] = int((col_data.fillna(0) < 0).sum())
        else:
            top_vals = (
                col_data.dropna().astype(str).value_counts().head(TOP_VALUE_COUNT).to_dict()
            )
            profile["top_values"] = {str(k): int(v) for k, v in top_vals.items()}

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(col_data, errors="coerce")
            if parsed.notna().mean() > 0.8:
                profile["is_datetime_parseable"] = True

        profiles.append(profile)
    return profiles


def _resolve_time_column(profiles: list[dict[str, Any]], schema: SchemaInfo, semantic_hints: dict | None) -> str | None:
    hinted = (semantic_hints or {}).get("time_column")
    if isinstance(hinted, str) and any(p["name"] == hinted for p in profiles):
        return hinted
    if schema.date_column:
        return schema.date_column
    for p in profiles:
        if p.get("is_datetime_parseable"):
            return p["name"]
    return None


def _classify_columns(
    profiles: list[dict[str, Any]],
    schema: SchemaInfo,
    semantic_hints: dict | None,
) -> dict[str, Any]:
    time_col = _resolve_time_column(profiles, schema, semantic_hints)

    identifiers: list[str] = []
    measures: list[dict[str, Any]] = []
    dimensions: list[dict[str, Any]] = []

    hinted_metric = (semantic_hints or {}).get("primary_metric") or schema.numeric_column
    metric_candidates = (semantic_hints or {}).get("metric_candidates") or []

    for p in profiles:
        name = p["name"]
        if name == time_col:
            continue

        if p["is_identifier_candidate"]:
            identifiers.append(name)
            continue

        if p["is_numeric"]:
            role = "primary" if name == hinted_metric else "supporting"
            measures.append(
                {
                    "name": name,
                    "role": role,
                    "aggregation_hint": _aggregation_hint(name),
                    "cardinality_bucket": p["cardinality_bucket"],
                }
            )
        else:
            dimensions.append(
                {
                    "name": name,
                    "cardinality_bucket": p["cardinality_bucket"],
                    "suitable_chart_types": _suitable_chart_types_for_dimension(p["cardinality_bucket"]),
                }
            )

    primary_metric = None
    if hinted_metric and any(m["name"] == hinted_metric for m in measures):
        primary_metric = hinted_metric
    elif measures:
        for candidate in metric_candidates:
            if any(m["name"] == candidate for m in measures):
                primary_metric = candidate
                break
        if primary_metric is None:
            primary_metric = measures[0]["name"]

    if primary_metric:
        for m in measures:
            m["role"] = "primary" if m["name"] == primary_metric else "supporting"

    return {
        "time_column": time_col,
        "identifier_columns": identifiers,
        "measure_columns": measures,
        "dimension_columns": dimensions,
        "primary_metric": primary_metric,
    }


def _compute_time_scope(df: pd.DataFrame, time_col: str | None) -> dict[str, Any]:
    scope: dict[str, Any] = {
        "has_time": False,
        "time_column": time_col,
        "start": None,
        "end": None,
        "span_days": None,
        "granularity_guess": None,
    }
    if not time_col or time_col not in df.columns:
        return scope

    parsed = pd.to_datetime(df[time_col], errors="coerce")
    valid = parsed.dropna()
    if valid.empty:
        return scope

    span_days = int((valid.max() - valid.min()).days)
    if span_days <= 60:
        granularity = "daily"
    elif span_days <= 365:
        granularity = "weekly"
    else:
        granularity = "monthly"

    scope.update(
        {
            "has_time": True,
            "start": str(valid.min().date()),
            "end": str(valid.max().date()),
            "span_days": span_days,
            "granularity_guess": granularity,
        }
    )
    return scope


def _detect_derived_metric_candidates(
    df: pd.DataFrame,
    measure_columns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    measure_names = [m["name"] for m in measure_columns]

    numeric_df = df[measure_names].apply(pd.to_numeric, errors="coerce") if measure_names else pd.DataFrame()
    seen_pairs: set[tuple[str, str, str]] = set()

    for a in measure_names:
        for b in measure_names:
            if a == b:
                continue
            for c in measure_names:
                if c in (a, b):
                    continue
                pair_key = (c, *sorted([a, b]))
                if pair_key in seen_pairs:
                    continue
                product = numeric_df[a] * numeric_df[b]
                target = numeric_df[c]
                if product.notna().sum() < max(5, len(numeric_df) // 2):
                    continue
                if target.notna().sum() < max(5, len(numeric_df) // 2):
                    continue
                diff = (product - target).abs()
                tolerance = target.abs() * 0.01
                matches = (diff <= tolerance).sum()
                if matches >= max(5, len(numeric_df) * 0.9):
                    seen_pairs.add(pair_key)
                    op_a, op_b = sorted([a, b])
                    candidates.append(
                        {
                            "name": f"{c} (formula match)",
                            "formula": f"{op_a} * {op_b}",
                            "why_useful": (
                                f"Detected: {c} ≈ {op_a} * {op_b} for ≥90% of rows. "
                                f"Use {op_a} and {op_b} as independent levers."
                            ),
                            "source": "deterministic_formula_match",
                        }
                    )
    return candidates


def _build_data_quality_signals(
    profiles: list[dict[str, Any]],
    row_count: int,
) -> dict[str, Any]:
    high_null_columns = [
        p["name"] for p in profiles if p["null_pct"] > 30
    ]
    suspicious_negatives = [
        p["name"] for p in profiles if p.get("is_numeric") and p.get("negative_count", 0) > 0
    ]
    very_high_cardinality_dims = [
        p["name"]
        for p in profiles
        if not p["is_numeric"]
        and not p["is_identifier_candidate"]
        and p["cardinality_bucket"] in {"high", "very_high"}
    ]
    return {
        "row_count": int(row_count),
        "small_sample_warning": row_count < 100,
        "high_null_columns": high_null_columns,
        "negative_value_columns": suspicious_negatives,
        "very_high_cardinality_dimensions": very_high_cardinality_dims,
    }


def build_understanding_input(
    df: pd.DataFrame,
    schema: SchemaInfo,
    semantic_hints: dict | None,
) -> dict:
    profiles = _build_column_profiles(df)
    classification = _classify_columns(profiles, schema, semantic_hints)
    time_scope = _compute_time_scope(df, classification["time_column"])
    derived_candidates = _detect_derived_metric_candidates(df, classification["measure_columns"])
    quality_signals = _build_data_quality_signals(profiles, len(df))

    sample_rows = df.head(SAMPLE_ROW_COUNT).astype(str).to_dict(orient="records")

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": profiles,
        "sample_rows": sample_rows,
        "schema": {
            "date_column": schema.date_column,
            "numeric_column": schema.numeric_column,
            "category_column": schema.category_column,
        },
        "semantic_hints": semantic_hints or {},
        "classification": classification,
        "time_scope": time_scope,
        "derived_metric_candidates_deterministic": derived_candidates,
        "data_quality_signals": quality_signals,
    }


def generate_deterministic_understanding(input_data: dict) -> dict:
    classification = input_data["classification"]
    time_scope = input_data["time_scope"]
    quality_signals = input_data["data_quality_signals"]
    derived_candidates = input_data["derived_metric_candidates_deterministic"]

    primary_metric = classification.get("primary_metric")
    measures = classification.get("measure_columns", [])
    dimensions = classification.get("dimension_columns", [])
    identifiers = classification.get("identifier_columns", [])
    time_col = classification.get("time_column")

    parts: list[str] = []
    if primary_metric:
        parts.append(f"primary metric is '{primary_metric}'")
    if time_col:
        parts.append(f"with a time column '{time_col}'")
    if dimensions:
        parts.append(f"slicing by {', '.join(d['name'] for d in dimensions)}")
    description = (
        "Tabular dataset where " + "; ".join(parts) + "."
        if parts
        else "Tabular dataset with no clear metric, time, or dimension columns."
    )

    quality_notes: list[str] = []
    if quality_signals["high_null_columns"]:
        quality_notes.append(
            "High nulls in: " + ", ".join(quality_signals["high_null_columns"])
        )
    if quality_signals["negative_value_columns"]:
        quality_notes.append(
            "Negative values in: " + ", ".join(quality_signals["negative_value_columns"])
        )
    if quality_signals["small_sample_warning"]:
        quality_notes.append(
            f"Small sample size ({quality_signals['row_count']} rows) — statistical robustness limited."
        )
    if quality_signals["very_high_cardinality_dimensions"]:
        quality_notes.append(
            "Very high cardinality dimensions (consider top-N): "
            + ", ".join(quality_signals["very_high_cardinality_dimensions"])
        )
    if not quality_notes:
        quality_notes.append("No major data quality issues detected.")

    return {
        "dataset_description": description,
        "business_domain_guess": "unknown",
        "primary_entity": "row",
        "primary_metric": primary_metric,
        "identifier_columns": identifiers,
        "measure_columns": measures,
        "dimension_columns": dimensions,
        "time_scope": time_scope,
        "temporal_features": {
            "has_time": time_scope.get("has_time", False),
            "time_column": time_col,
            "suggested_granularity": time_scope.get("granularity_guess"),
            "seasonality_candidates": [],
        },
        "derived_metric_opportunities": derived_candidates,
        "data_quality_notes": quality_notes,
        "data_quality_signals": quality_signals,
        "confidence": "low",
    }


_DETERMINISTIC_PROTECTED_FIELDS = {
    "time_scope",
    "identifier_columns",
    "measure_columns",
    "dimension_columns",
    "primary_metric",
    "data_quality_signals",
    "derived_metric_opportunities",
}


def _merge_with_fallback(llm_result: dict, fallback: dict) -> dict:
    merged = dict(fallback)

    for key, value in llm_result.items():
        if key in _DETERMINISTIC_PROTECTED_FIELDS:
            continue
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, dict)) and len(value) == 0:
            continue
        merged[key] = value

    llm_derived = llm_result.get("derived_metric_opportunities")
    if isinstance(llm_derived, list) and llm_derived:
        seen = {
            (d.get("name", ""), d.get("formula", ""))
            for d in fallback.get("derived_metric_opportunities", [])
        }
        for item in llm_derived:
            if not isinstance(item, dict):
                continue
            key = (item.get("name", ""), item.get("formula", ""))
            if key in seen:
                continue
            seen.add(key)
            merged.setdefault("derived_metric_opportunities", []).append(
                {**item, "source": "llm"}
            )

    llm_temporal = llm_result.get("temporal_features")
    if isinstance(llm_temporal, dict):
        seasonality = llm_temporal.get("seasonality_candidates")
        if isinstance(seasonality, list) and seasonality:
            merged.setdefault("temporal_features", {})["seasonality_candidates"] = list(seasonality)

    return merged


def generate_understanding(
    df: pd.DataFrame,
    schema: SchemaInfo,
    semantic_hints: dict | None,
    debug_logger=None,
) -> dict:
    input_data = build_understanding_input(df, schema, semantic_hints)

    if debug_logger:
        debug_logger.log_event(
            "understanding_input_built",
            {
                "row_count": input_data["row_count"],
                "column_count": input_data["column_count"],
                "classification": input_data["classification"],
                "time_scope": input_data["time_scope"],
                "data_quality_signals": input_data["data_quality_signals"],
                "derived_metric_candidates_deterministic": input_data["derived_metric_candidates_deterministic"],
                "schema": input_data["schema"],
                "semantic_hints": input_data["semantic_hints"],
                "sample_row_count_sent": len(input_data["sample_rows"]),
            },
        )

    fallback = generate_deterministic_understanding(input_data)

    if debug_logger:
        debug_logger.log_event(
            "understanding_deterministic_baseline",
            {"summary": fallback},
        )

    provider = settings.llm_provider
    model_name = resolve_active_model()

    try:
        if debug_logger:
            debug_logger.log_event(
                "understanding_llm_called",
                {"provider": provider, "model": model_name},
            )

        llm_result = generate_dataset_understanding(input_data)

        if debug_logger:
            debug_logger.log_event(
                "understanding_llm_parsed",
                {"provider": provider, "model": model_name, "parsed": llm_result},
            )

        merged = _merge_with_fallback(llm_result, fallback)

        result = {
            "source": "llm",
            "provider": provider,
            "model": model_name,
            "summary": merged,
            "input": input_data,
            "fallback_baseline": fallback,
            "error": None,
        }

        if debug_logger:
            debug_logger.log_event(
                "understanding_final",
                {
                    "source": result["source"],
                    "provider": provider,
                    "model": model_name,
                    "summary": merged,
                },
            )

        return result

    except Exception as exc:
        if debug_logger:
            debug_logger.log_event(
                "understanding_fallback_used",
                {
                    "provider": provider,
                    "model": model_name,
                    "reason": str(exc),
                    "exception_type": type(exc).__name__,
                },
            )

        result = {
            "source": "deterministic",
            "provider": provider,
            "model": model_name,
            "summary": fallback,
            "input": input_data,
            "fallback_baseline": fallback,
            "error": str(exc),
        }

        if debug_logger:
            debug_logger.log_event(
                "understanding_final",
                {
                    "source": result["source"],
                    "provider": provider,
                    "model": model_name,
                    "summary": fallback,
                },
            )

        return result
