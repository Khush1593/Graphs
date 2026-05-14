from __future__ import annotations

from typing import Any

import pandas as pd

from app.data_engine import SchemaInfo


GRANULARITY_FREQ_CODE = {
    "daily": "D",
    "weekly": "W",
    "monthly": "M",
    "quarterly": "Q",
}

SECONDARY_GRANULARITY = {
    "daily": "weekly",
    "weekly": "monthly",
    "monthly": "quarterly",
    "quarterly": "yearly",
}

BUCKETS_PER_YEAR = {
    "daily": 365,
    "weekly": 52,
    "monthly": 12,
    "quarterly": 4,
    "yearly": 1,
}

GROWTH_THRESHOLDS = {
    "daily": {"opportunity": 5, "decline": -3, "improvement": -8, "unwanted": 5, "instability": 7},
    "weekly": {"opportunity": 8, "decline": -5, "improvement": -10, "unwanted": 5, "instability": 8},
    "monthly": {"opportunity": 12, "decline": -8, "improvement": -12, "unwanted": 8, "instability": 10},
    "quarterly": {"opportunity": 20, "decline": -12, "improvement": -20, "unwanted": 12, "instability": 15},
}

# --- Statistical guardrails -------------------------------------------------
# Below this many valid rows we don't compute outliers at all. IQR is robust
# but still nonsense on a handful of points; small samples produce false
# positives that erode trust in the narrator.
MIN_ROWS_FOR_STATS = 30

# Tukey's fence multiplier. 1.5 * IQR is the textbook "mild outlier" bound;
# anything beyond that gets surfaced. Values >= 3 IQRs from the median are
# treated as extreme for priority scoring.
ANOMALY_IQR_MULTIPLIER = 1.5
EXTREME_IQR_DISTANCE = 3.0


PERIOD_PCT_LABELS = {
    "daily": "DoD",
    "weekly": "WoW",
    "monthly": "MoM",
    "quarterly": "QoQ",
    "yearly": "YoY",
}


def _get_metric_column(df: pd.DataFrame, schema: SchemaInfo, semantic_hints: dict | None) -> str | None:
    hinted_metric = (semantic_hints or {}).get("primary_metric")
    if isinstance(hinted_metric, str) and hinted_metric in df.columns:
        return hinted_metric
    return schema.numeric_column if schema.numeric_column in df.columns else None


def _get_time_column(df: pd.DataFrame, schema: SchemaInfo, semantic_hints: dict | None) -> str | None:
    hinted_time = (semantic_hints or {}).get("time_column")
    if isinstance(hinted_time, str) and hinted_time in df.columns:
        return hinted_time
    return schema.date_column if schema.date_column in df.columns else None


def _get_dimensions(df: pd.DataFrame, schema: SchemaInfo, semantic_hints: dict | None) -> list[str]:
    hinted_dims = (semantic_hints or {}).get("dimensions") or []
    dimensions = [d for d in hinted_dims if isinstance(d, str) and d in df.columns]
    if not dimensions and schema.category_column and schema.category_column in df.columns:
        dimensions = [schema.category_column]
    return dimensions[:3]


def _resolve_columns(
    df: pd.DataFrame,
    schema: SchemaInfo,
    semantic_hints: dict | None,
    context=None,
) -> tuple[str | None, str | None, list[str]]:
    """Context (user-confirmed) beats semantic_hints (auto-detected) beats schema defaults."""
    if context is not None:
        metric_col = context.focus_metric
        if not metric_col or metric_col not in df.columns:
            metric_col = _get_metric_column(df, schema, semantic_hints)

        time_col = context.time_column
        if not time_col or time_col not in df.columns:
            time_col = _get_time_column(df, schema, semantic_hints)

        ctx_dims = [d for d in (context.focus_dimensions or []) if d in df.columns]
        dimensions = ctx_dims if ctx_dims else _get_dimensions(df, schema, semantic_hints)
    else:
        metric_col = _get_metric_column(df, schema, semantic_hints)
        time_col = _get_time_column(df, schema, semantic_hints)
        dimensions = _get_dimensions(df, schema, semantic_hints)

    return metric_col, time_col, dimensions


def _pct_change(current: float, previous: float) -> float | None:
    """Percentage change, robust to the edges that used to crash callers.

    Returns None for:
      - missing inputs (None / NaN on either side)
      - previous == 0 (would be ZeroDivisionError; conceptually undefined or
        "infinite" growth — we report None rather than ``inf`` so JSON stays
        finite. Callers that need to distinguish the two cases should use
        ``_pct_change_with_status``).
    """
    if current is None or previous is None:
        return None
    try:
        if pd.isna(current) or pd.isna(previous):
            return None
    except (TypeError, ValueError):
        return None
    if previous == 0:
        return None
    return ((current - previous) / previous) * 100.0


def _pct_change_with_status(current: float, previous: float) -> tuple[float | None, str]:
    """Same math as ``_pct_change`` but reports *why* the result is None.

    Status codes:
      - ``ok``                — value is a finite percentage
      - ``missing_input``     — current or previous was None/NaN
      - ``prev_zero``         — previous was exactly zero (growth undefined)
    """
    if current is None or previous is None:
        return None, "missing_input"
    try:
        if pd.isna(current) or pd.isna(previous):
            return None, "missing_input"
    except (TypeError, ValueError):
        return None, "missing_input"
    if previous == 0:
        return None, "prev_zero"
    return ((current - previous) / previous) * 100.0, "ok"


def _trend_slope(series: pd.Series) -> float | None:
    values = series.dropna().tolist()
    n = len(values)
    if n < 2:
        return None
    x = list(range(n))
    x_mean = sum(x) / n
    y_mean = sum(values) / n
    num = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
    den = sum((x[i] - x_mean) ** 2 for i in range(n))
    if den == 0:
        return None
    return num / den


def _priority_label(score: int) -> str:
    if score >= 80:
        return "high"
    if score >= 50:
        return "medium"
    return "low"


def _goal_consensus_direction(metric: str | None, active_goals: list[dict]) -> str | None:
    if not metric or not active_goals:
        return None

    relevant = [g for g in active_goals if g.get("metric") == metric]
    if not relevant:
        return None

    weights: dict[str, int] = {"increase": 0, "decrease": 0, "stabilize": 0}
    priority_weight = {"high": 3, "medium": 2, "low": 1}
    for g in relevant:
        d = g.get("direction")
        if d in weights:
            weights[d] += priority_weight.get(g.get("priority", "medium"), 2)

    if all(v == 0 for v in weights.values()):
        return None

    return max(weights, key=weights.get)


def _evaluate_growth_against_direction(
    pct_change: float,
    metric_col: str,
    consensus_direction: str | None,
    period_label: str = "MoM",
    thresholds: dict | None = None,
) -> dict | None:
    th = thresholds or GROWTH_THRESHOLDS["monthly"]

    if consensus_direction == "decrease":
        if pct_change >= th["unwanted"]:
            return {
                "type": "growth_unwanted_increase",
                "score": 85,
                "message": f"{metric_col} grew {pct_change:.1f}% {period_label} — opposite of your stated direction (decrease).",
                "recommendation": "Investigate what drove the increase and confirm whether goal direction needs revision.",
                "wanted_direction": "decrease",
            }
        if pct_change <= th["improvement"]:
            return {
                "type": "growth_improvement",
                "score": 65,
                "message": f"{metric_col} dropped {abs(pct_change):.1f}% {period_label} — aligned with your decrease goal.",
                "recommendation": "Confirm this drop is from the right driver (not data quality) and continue the strategy.",
                "wanted_direction": "decrease",
            }
        return None

    if consensus_direction == "stabilize":
        if abs(pct_change) >= th["instability"]:
            return {
                "type": "growth_instability",
                "score": 75,
                "message": f"{metric_col} moved {pct_change:+.1f}% {period_label} — your goal is to stabilize this metric.",
                "recommendation": "Identify what introduced the swing; tighten controls on the contributing drivers.",
                "wanted_direction": "stabilize",
            }
        return None

    if pct_change <= th["decline"]:
        return {
            "type": "growth_decline",
            "score": 85,
            "message": f"{metric_col} declined {abs(pct_change):.1f}% {period_label}.",
            "recommendation": "Investigate pricing, conversion funnel, and stock availability for weak periods.",
            "wanted_direction": "increase",
        }
    if pct_change >= th["opportunity"]:
        return {
            "type": "growth_opportunity",
            "score": 65,
            "message": f"{metric_col} grew {pct_change:.1f}% {period_label}.",
            "recommendation": "Consider scaling inventory and marketing in high-performing segments.",
            "wanted_direction": "increase",
        }

    return None


def _resolve_granularity(context, time_scope: dict | None) -> tuple[str, str]:
    if context is not None and context.time_granularity in GRANULARITY_FREQ_CODE:
        primary = context.time_granularity
    else:
        guess = (time_scope or {}).get("granularity_guess")
        primary = guess if guess in GRANULARITY_FREQ_CODE else "monthly"
    secondary = SECONDARY_GRANULARITY.get(primary, "monthly")
    return primary, secondary


def _aggregate_by_period(
    df: pd.DataFrame,
    time_col: str,
    metric_col: str,
    granularity: str,
) -> pd.DataFrame:
    freq = GRANULARITY_FREQ_CODE.get(granularity, "M")
    agg = (
        df.assign(period=df[time_col].dt.to_period(freq).astype(str))
        .groupby("period", as_index=False)[metric_col]
        .sum()
        .sort_values("period")
        .reset_index(drop=True)
    )
    return agg


def _compute_baseline(
    period_df: pd.DataFrame,
    metric_col: str,
    granularity: str,
    comparison_window: str | None,
) -> dict[str, Any]:
    if period_df.empty:
        return {
            "method": "no_data",
            "recent_value": None,
            "baseline_value": None,
            "delta_vs_baseline_pct": None,
            "status": "no_data",
            "requested_window": comparison_window,
        }

    recent = float(period_df.iloc[-1][metric_col])
    requested = (comparison_window or "").strip().lower()

    method: str
    baseline: float | None = None

    if requested in {"previous period", "previous_period", "previous"}:
        if len(period_df) < 2:
            method = "fallback_no_previous_period"
        else:
            baseline = float(period_df.iloc[-2][metric_col])
            method = "previous_period"
    elif requested in {"same period last year", "year over year", "yoy"}:
        offset = BUCKETS_PER_YEAR.get(granularity, 12)
        if len(period_df) > offset:
            baseline = float(period_df.iloc[-(offset + 1)][metric_col])
            method = "year_over_year"
        else:
            tail = period_df[metric_col].tail(6)
            baseline = float(tail.mean()) if not tail.empty else None
            method = "fallback_6_period_rolling_yoy_data_short"
    else:
        tail = period_df[metric_col].tail(6)
        baseline = float(tail.mean()) if not tail.empty else None
        method = "6_period_rolling"

    delta = _pct_change(recent, baseline) if baseline not in (None, 0) else None
    if delta is None:
        status = "neutral"
    elif delta >= 5:
        status = "above_baseline"
    elif delta <= -5:
        status = "below_baseline"
    else:
        status = "neutral"

    return {
        "method": method,
        "recent_value": round(recent, 2),
        "baseline_value": round(baseline, 2) if baseline is not None else None,
        "delta_vs_baseline_pct": round(delta, 2) if delta is not None else None,
        "status": status,
        "requested_window": comparison_window,
    }


def _priority_item(
    item_type: str,
    score: int,
    message: str,
    recommendation: str,
    metric: str | None = None,
    dimension: str | None = None,
    direction: str | None = None,
) -> dict[str, Any]:
    return {
        "type": item_type,
        "priority_score": score,
        "priority": _priority_label(score),
        "message": message,
        "recommendation": recommendation,
        "signal_metric": metric,
        "signal_dimension": dimension,
        "signal_direction": direction,
        "goal_alignment_score": 0,
        "goal_alignment_with": [],
    }


def generate_insight_signals_v2(
    df: pd.DataFrame,
    schema: SchemaInfo,
    semantic_hints: dict | None = None,
    context=None,
) -> dict:
    metric_col, time_col, dimensions = _resolve_columns(df, schema, semantic_hints, context)

    active_goals = list(context.active_goals) if context is not None else []
    goal_consensus = _goal_consensus_direction(metric_col, active_goals)
    contributing_goal_ids = [
        g.get("id") for g in active_goals if g.get("metric") == metric_col
    ]

    signals = {
        "primary_metric": metric_col,
        "time_column": time_col,
        "dimensions": dimensions,
        "goal_consensus_direction": goal_consensus,
        "goal_consensus_contributing_goal_ids": contributing_goal_ids,
        "growth": {},
        "benchmarks": {},
        "top_contributors": [],
        "declining_segments": [],
        "concentration_risk": {},
        "anomalies": {},
        "priority_insights": [],
        "action_items": [],
        "key_narratives": [],
        "quality": {
            "rows": int(len(df)),
            "valid_rows": 0,
            "warnings": [],
        },
    }

    if not metric_col:
        signals["quality"]["warnings"].append("No numeric metric found.")
        return signals

    local_df = df.copy()
    local_df[metric_col] = pd.to_numeric(local_df[metric_col], errors="coerce")

    if time_col:
        local_df[time_col] = pd.to_datetime(local_df[time_col], errors="coerce")
        local_df = local_df.dropna(subset=[metric_col, time_col])
    else:
        local_df = local_df.dropna(subset=[metric_col])
        signals["quality"]["warnings"].append("No time column found. Time-based signals are limited.")

    signals["quality"]["valid_rows"] = int(len(local_df))
    if local_df.empty:
        signals["quality"]["warnings"].append("No valid rows after cleaning.")
        return signals

    if time_col:
        time_scope_for_resolve = (
            (context.understanding_summary or {}).get("time_scope") or {}
        ) if context else {}
        primary_granularity, secondary_granularity = _resolve_granularity(context, time_scope_for_resolve)
        comparison_window = context.comparison_window if context else None
        period_label_primary = PERIOD_PCT_LABELS.get(primary_granularity, "PoP")
        period_label_secondary = PERIOD_PCT_LABELS.get(secondary_granularity, "PoP")
        thresholds = GROWTH_THRESHOLDS.get(primary_granularity, GROWTH_THRESHOLDS["monthly"])

        signals["time_granularity"] = primary_granularity
        signals["secondary_granularity"] = secondary_granularity
        signals["comparison_window_requested"] = comparison_window

        primary_periods = _aggregate_by_period(local_df, time_col, metric_col, primary_granularity)
        secondary_periods = _aggregate_by_period(local_df, time_col, metric_col, secondary_granularity)

        primary_pct: float | None = None
        primary_pct_status = "insufficient_data"
        if len(primary_periods) >= 2:
            primary_pct, primary_pct_status = _pct_change_with_status(
                float(primary_periods.iloc[-1][metric_col]),
                float(primary_periods.iloc[-2][metric_col]),
            )

        secondary_pct: float | None = None
        secondary_pct_status = "insufficient_data"
        if len(secondary_periods) >= 2:
            secondary_pct, secondary_pct_status = _pct_change_with_status(
                float(secondary_periods.iloc[-1][metric_col]),
                float(secondary_periods.iloc[-2][metric_col]),
            )

        slope = _trend_slope(primary_periods[metric_col].tail(6))

        baseline_block = _compute_baseline(
            primary_periods, metric_col, primary_granularity, comparison_window
        )
        baseline_block["period"] = str(primary_periods.iloc[-1]["period"])
        baseline_block["granularity"] = primary_granularity
        signals["benchmarks"] = baseline_block

        signals["growth"] = {
            "type": "growth",
            "metric": metric_col,
            "primary_period": primary_granularity,
            "primary_period_pct": round(primary_pct, 2) if primary_pct is not None else None,
            "primary_period_label": period_label_primary,
            "secondary_period": secondary_granularity,
            "secondary_period_pct": round(secondary_pct, 2) if secondary_pct is not None else None,
            "secondary_period_label": period_label_secondary,
            "trend_slope": round(slope, 4) if slope is not None else None,
            "direction": "up" if (primary_pct or 0) >= 0 else "down",
            # Explicit status flags so the narrator can distinguish "we don't
            # know" (insufficient_data / missing_input) from "growth is
            # mathematically undefined" (prev_zero) — both manifest as a
            # null pct but mean very different things downstream.
            "primary_period_pct_status": primary_pct_status,
            "secondary_period_pct_status": secondary_pct_status,
            "thresholds_used": thresholds,
            # backwards-compatible aliases for the existing narrator/text functions
            "mom_pct": round(primary_pct, 2) if primary_pct is not None and primary_granularity == "monthly" else None,
            "wow_pct": round(primary_pct, 2) if primary_pct is not None and primary_granularity == "weekly" else None,
        }

        if primary_pct is not None:
            growth_eval = _evaluate_growth_against_direction(
                pct_change=primary_pct,
                metric_col=metric_col,
                consensus_direction=goal_consensus,
                period_label=period_label_primary,
                thresholds=thresholds,
            )
            if growth_eval:
                signals["priority_insights"].append(
                    _priority_item(
                        growth_eval["type"],
                        growth_eval["score"],
                        growth_eval["message"],
                        growth_eval["recommendation"],
                        metric=metric_col,
                        direction=growth_eval["wanted_direction"],
                    )
                )

        if len(primary_periods) >= 2 and dimensions:
            last_period = primary_periods.iloc[-1]["period"]
            prev_period = primary_periods.iloc[-2]["period"]
            dim_col = dimensions[0]

            by_dim_period = (
                local_df.assign(
                    period=local_df[time_col]
                    .dt.to_period(GRANULARITY_FREQ_CODE.get(primary_granularity, "M"))
                    .astype(str)
                )
                .groupby(["period", dim_col], as_index=False)[metric_col]
                .sum()
            )

            latest = by_dim_period[by_dim_period["period"] == last_period][[dim_col, metric_col]]
            previous = by_dim_period[by_dim_period["period"] == prev_period][[dim_col, metric_col]]
            merged = latest.merge(previous, on=dim_col, how="outer", suffixes=("_last", "_prev")).fillna(0.0)

            merged["growth_pct"] = merged.apply(
                lambda row: _pct_change(row[f"{metric_col}_last"], row[f"{metric_col}_prev"]),
                axis=1,
            )

            declining = merged[merged["growth_pct"].notna() & (merged["growth_pct"] < 0)].copy()
            declining = declining.sort_values("growth_pct").head(3)

            signals["declining_segments"] = [
                {
                    "dimension": dim_col,
                    "name": str(row[dim_col]),
                    "growth_pct": round(float(row["growth_pct"]), 2),
                    "last_value": round(float(row[f"{metric_col}_last"]), 2),
                    "prev_value": round(float(row[f"{metric_col}_prev"]), 2),
                    "last_period": str(last_period),
                    "prev_period": str(prev_period),
                    "granularity": primary_granularity,
                }
                for _, row in declining.iterrows()
            ]

            if signals["declining_segments"]:
                worst = signals["declining_segments"][0]
                severity = 80 if worst["growth_pct"] <= -15 else 60
                signals["priority_insights"].append(
                    _priority_item(
                        "declining_segment",
                        severity,
                        f"{worst['name']} is declining ({worst['growth_pct']:.1f}% {period_label_primary} on {primary_granularity} buckets).",
                        "Review pricing, discount strategy, and demand drivers for this segment.",
                        metric=metric_col,
                        dimension=worst.get("dimension"),
                        direction="increase",
                    )
                )

    if dimensions:
        dim_col = dimensions[0]
        by_dim = local_df.groupby(dim_col, as_index=False)[metric_col].sum().sort_values(metric_col, ascending=False)
        total = float(by_dim[metric_col].sum())
        top = by_dim.head(3)

        top_contributors = []
        for _, row in top.iterrows():
            share_pct = (float(row[metric_col]) / total * 100.0) if total else 0.0
            top_contributors.append(
                {
                    "dimension": dim_col,
                    "name": str(row[dim_col]),
                    "total": round(float(row[metric_col]), 2),
                    "share_pct": round(share_pct, 2),
                }
            )
        signals["top_contributors"] = top_contributors

        top_share = top_contributors[0]["share_pct"] if top_contributors else 0.0
        if top_share >= 80:
            risk_level = "high"
        elif top_share >= 60:
            risk_level = "medium"
        else:
            risk_level = "low"
        signals["concentration_risk"] = {
            "dimension": dim_col,
            "top_share_pct": round(top_share, 2),
            "risk_level": risk_level,
            "message": (
                f"High dependency on {top_contributors[0]['name']}"
                if top_contributors and risk_level in {"high", "medium"}
                else "No major concentration dependency"
            ),
        }

        if risk_level == "high":
            signals["priority_insights"].append(
                _priority_item(
                    "concentration_risk",
                    90,
                    f"{top_contributors[0]['name']} contributes {top_share:.1f}% of {metric_col}.",
                    "Diversify revenue by investing in secondary categories and targeted campaigns.",
                    metric=metric_col,
                    dimension=dim_col,
                    direction="stabilize",
                )
            )
        elif risk_level == "medium":
            signals["priority_insights"].append(
                _priority_item(
                    "concentration_risk",
                    60,
                    f"Top category concentration is {top_share:.1f}%.",
                    "Create a diversification plan to reduce single-category dependency.",
                    metric=metric_col,
                    dimension=dim_col,
                    direction="stabilize",
                )
            )
    else:
        signals["quality"]["warnings"].append("No categorical dimension available for contributor analysis.")

    # ------------------------------------------------------------------
    # Anomaly detection — robust IQR method with sample-size gatekeeping.
    #
    # Why we changed this:
    #   * Global Z-scores assume normal data and use the mean/std, both of
    #     which are pulled around by the very outliers we're trying to find.
    #     On small or seasonal series this produces alert fatigue.
    #   * IQR (Tukey's fences) is computed from quantiles, so a single huge
    #     spike does not move the bounds.
    #   * On <30 rows even IQR is unreliable; we skip entirely and flag it.
    #
    # Output contract: the anomalies dict keeps its existing keys
    # (method, threshold, count, pct_of_rows, events) plus additive fields.
    # Event dicts keep date / value / event_type / zscore (now a *robust
    # score* in units of IQR-from-median) / deviation_pct_from_mean.
    # ------------------------------------------------------------------
    anomaly_events: list[dict[str, Any]] = []
    valid_rows = int(len(local_df))

    if valid_rows < MIN_ROWS_FOR_STATS:
        signals["anomalies"] = {
            "method": "iqr",
            "threshold": ANOMALY_IQR_MULTIPLIER,
            "count": 0,
            "pct_of_rows": 0.0,
            "events": [],
            "anomalies_skipped_due_to_low_volume": True,
            "min_rows_required": MIN_ROWS_FOR_STATS,
            "valid_rows": valid_rows,
        }
        signals["quality"]["warnings"].append(
            f"Anomaly detection skipped: only {valid_rows} valid rows "
            f"(need at least {MIN_ROWS_FOR_STATS})."
        )
    else:
        # When we have a time column, anomalies are most meaningful at the
        # daily aggregate level (one bar per day) rather than per-row.
        daily: pd.DataFrame | None = None
        if time_col:
            daily = (
                local_df.assign(day=local_df[time_col].dt.date)
                .groupby("day", as_index=False)[metric_col]
                .sum()
                .sort_values("day")
                .reset_index(drop=True)
            )
            target_series = daily[metric_col]
        else:
            target_series = local_df[metric_col]

        series = target_series.dropna()

        if len(series) < MIN_ROWS_FOR_STATS:
            signals["anomalies"] = {
                "method": "iqr",
                "threshold": ANOMALY_IQR_MULTIPLIER,
                "count": 0,
                "pct_of_rows": 0.0,
                "events": [],
                "anomalies_skipped_due_to_low_volume": True,
                "min_rows_required": MIN_ROWS_FOR_STATS,
                "valid_rows": int(len(series)),
            }
            signals["quality"]["warnings"].append(
                f"Anomaly detection skipped: aggregated to {len(series)} "
                f"buckets, below the {MIN_ROWS_FOR_STATS}-bucket minimum."
            )
        else:
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            median = float(series.median())
            mean_val = float(series.mean())
            lower = q1 - ANOMALY_IQR_MULTIPLIER * iqr
            upper = q3 + ANOMALY_IQR_MULTIPLIER * iqr

            if iqr <= 0:
                # Flat-ish series: no spread, no outliers worth surfacing.
                signals["anomalies"] = {
                    "method": "iqr",
                    "threshold": ANOMALY_IQR_MULTIPLIER,
                    "count": 0,
                    "pct_of_rows": 0.0,
                    "events": [],
                    "anomalies_skipped_due_to_low_volume": False,
                    "q1": round(q1, 4),
                    "q3": round(q3, 4),
                    "iqr": 0.0,
                    "median": round(median, 4),
                    "note": "iqr_zero_no_variance",
                }
            else:
                outlier_mask = (series < lower) | (series > upper)
                outlier_count = int(outlier_mask.sum())

                if daily is not None:
                    daily_outliers = daily[
                        (daily[metric_col] < lower) | (daily[metric_col] > upper)
                    ].copy()
                    daily_outliers["robust_score"] = (
                        daily_outliers[metric_col] - median
                    ) / iqr
                    daily_outliers["abs_score"] = daily_outliers["robust_score"].abs()
                    daily_outliers = daily_outliers.sort_values(
                        "abs_score", ascending=False
                    ).head(5)

                    for _, row in daily_outliers.iterrows():
                        value = float(row[metric_col])
                        robust = float(row["robust_score"])
                        deviation_pct = (
                            ((value - mean_val) / mean_val) * 100.0
                            if mean_val
                            else None
                        )
                        anomaly_events.append(
                            {
                                "date": str(row["day"]),
                                "value": round(value, 2),
                                "event_type": "spike" if value > upper else "drop",
                                # Kept as ``zscore`` for back-compat with the
                                # narrator; semantically it is now an
                                # IQR-distance from the median (signed).
                                "zscore": round(robust, 2),
                                "robust_score": round(robust, 2),
                                "deviation_pct_from_mean": round(deviation_pct, 2)
                                if deviation_pct is not None
                                else None,
                                "deviation_pct_from_median": (
                                    round(((value - median) / median) * 100.0, 2)
                                    if median
                                    else None
                                ),
                            }
                        )

                signals["anomalies"] = {
                    "method": "iqr",
                    "threshold": ANOMALY_IQR_MULTIPLIER,
                    "count": outlier_count,
                    "pct_of_rows": round(
                        (outlier_count / max(len(series), 1)) * 100.0, 2
                    ),
                    "events": anomaly_events,
                    "anomalies_skipped_due_to_low_volume": False,
                    "q1": round(q1, 4),
                    "q3": round(q3, 4),
                    "iqr": round(iqr, 4),
                    "median": round(median, 4),
                    "lower_bound": round(lower, 4),
                    "upper_bound": round(upper, 4),
                }

                known_events_text = ""
                if context is not None:
                    clarif = context.user_clarifications or {}
                    known_events_text = (clarif.get("known_events") or "").strip()

                if anomaly_events:
                    highest = anomaly_events[0]
                    # Threshold rescaled from Z-score (3.5σ) to IQR units;
                    # ~3 IQRs from the median is the standard "extreme" bound.
                    high_anomaly = abs(highest.get("zscore", 0)) >= EXTREME_IQR_DISTANCE
                    base_anomaly_score = 85 if high_anomaly else 55
                    stabilize_boost = 15 if goal_consensus == "stabilize" else 0

                    if known_events_text:
                        message = (
                            f"Unusual {highest['event_type']} on {highest['date']} "
                            f"({metric_col}={highest['value']}). "
                            f"Note: user reported known events in this period — may be expected."
                        )
                        recommendation = (
                            f"Cross-check against reported events: \"{known_events_text[:120]}\". "
                            "If explained, no action needed."
                        )
                        base_anomaly_score = max(30, base_anomaly_score - 20)
                    else:
                        message = (
                            f"Unusual {highest['event_type']} on {highest['date']} "
                            f"({metric_col}={highest['value']})."
                        )
                        recommendation = (
                            "Validate if this was campaign-driven, operational disruption, or data-quality issue."
                        )

                    signals["anomalies"]["known_events_context"] = known_events_text or None
                    signals["priority_insights"].append(
                        _priority_item(
                            "anomaly_event",
                            min(100, base_anomaly_score + stabilize_boost),
                            message,
                            recommendation,
                            metric=metric_col,
                            direction="stabilize",
                        )
                    )

    top_contributors = signals.get("top_contributors", [])
    growth = signals.get("growth", {})
    concentration = signals.get("concentration_risk", {})
    declining = signals.get("declining_segments", [])

    mom_pct = growth.get("mom_pct")
    if mom_pct is not None and top_contributors:
        top = top_contributors[0]
        if mom_pct > 0 and concentration.get("risk_level") in {"high", "medium"}:
            signals["key_narratives"].append(
                (
                    f"Recent growth appears driven by {top['name']} ({top['share_pct']:.1f}% share), "
                    "which also increases dependency risk."
                )
            )
        elif mom_pct < 0 and declining:
            worst = declining[0]
            signals["key_narratives"].append(
                f"Overall decline aligns with weakness in {worst['name']}, indicating a concentrated performance issue."
            )

    benchmark = signals.get("benchmarks", {})
    benchmark_delta = benchmark.get("delta_vs_baseline_pct")
    if benchmark_delta is not None:
        signals["key_narratives"].append(
            f"Current period is {benchmark_delta:.1f}% versus 6-period baseline ({benchmark.get('status')})."
        )

    if context is not None and context.active_goals:
        from app.context_engine import goal_alignment_score

        active_goals = context.active_goals
        for item in signals["priority_insights"]:
            score, matched_ids = goal_alignment_score(
                signal_metric=item.get("signal_metric"),
                signal_dimension=item.get("signal_dimension"),
                signal_direction=item.get("signal_direction"),
                goals=active_goals,
            )
            item["goal_alignment_score"] = score
            item["goal_alignment_with"] = matched_ids
            base = item.get("priority_score", 0)
            item["priority_score"] = min(100, base + score)
            item["priority"] = _priority_label(item["priority_score"])

        signals["goal_alignment_applied"] = True
        signals["active_goal_count"] = len(active_goals)
    else:
        signals["goal_alignment_applied"] = False
        signals["active_goal_count"] = 0

    signals["priority_insights"] = sorted(
        signals["priority_insights"], key=lambda x: x.get("priority_score", 0), reverse=True
    )
    signals["action_items"] = [
        {
            "priority": item["priority"],
            "type": item["type"],
            "recommendation": item["recommendation"],
            "goal_alignment_with": item.get("goal_alignment_with", []),
        }
        for item in signals["priority_insights"][:5]
    ]

    return signals


def generate_deterministic_insight_text(signals: dict) -> str:
    metric = signals.get("primary_metric") or "metric"
    growth = signals.get("growth", {})
    top_contributors = signals.get("top_contributors", [])
    declining = signals.get("declining_segments", [])
    risk = signals.get("concentration_risk", {})
    anomalies = signals.get("anomalies", {})
    warnings = signals.get("quality", {}).get("warnings", [])

    lines = []

    mom = growth.get("mom_pct")
    if mom is not None:
        direction = "up" if mom >= 0 else "down"
        lines.append(f"{metric} is {direction} by {abs(mom):.1f}% month-over-month.")

    wow = growth.get("wow_pct")
    if wow is not None:
        direction = "up" if wow >= 0 else "down"
        lines.append(f"Week-over-week change is {direction} {abs(wow):.1f}%.")

    if top_contributors:
        top = top_contributors[0]
        lines.append(
            f"Top contributor is {top['name']} with {top['share_pct']:.1f}% share of total {metric}."
        )

    if declining:
        d = declining[0]
        lines.append(
            f"Declining segment: {d['name']} ({d['growth_pct']:.1f}% vs previous period)."
        )

    if risk:
        lines.append(
            f"Concentration risk is {risk.get('risk_level', 'unknown')} (top share {risk.get('top_share_pct', 0)}%)."
        )

    outlier_count = anomalies.get("count")
    if outlier_count is not None:
        lines.append(f"Detected {outlier_count} potential anomaly records based on z-score threshold.")

    if warnings:
        lines.append("Data quality note: " + "; ".join(warnings))

    benchmark = signals.get("benchmarks", {})
    if benchmark:
        delta = benchmark.get("delta_vs_baseline_pct")
        if delta is not None:
            lines.append(
                f"Benchmark context: current period is {delta:.1f}% vs 6-period baseline ({benchmark.get('status')})."
            )

    priority_insights = signals.get("priority_insights", [])
    if priority_insights:
        lines.append("Priority actions:")
        for item in priority_insights[:4]:
            badge = "[HIGH]" if item["priority"] == "high" else "[MEDIUM]" if item["priority"] == "medium" else "[LOW]"
            lines.append(f"{badge} {item['message']} Action: {item['recommendation']}")

    narratives = signals.get("key_narratives", [])
    if narratives:
        lines.append("Key combined narratives:")
        for n in narratives[:2]:
            lines.append(str(n))

    if not lines:
        return "Insight: Not enough signal to generate business insights."

    return "\n".join(f"- {line}" for line in lines)
