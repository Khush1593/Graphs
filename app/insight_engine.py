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
    if previous == 0:
        return None
    return ((current - previous) / previous) * 100.0


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

        primary_pct = None
        if len(primary_periods) >= 2:
            primary_pct = _pct_change(
                float(primary_periods.iloc[-1][metric_col]),
                float(primary_periods.iloc[-2][metric_col]),
            )

        secondary_pct = None
        if len(secondary_periods) >= 2:
            secondary_pct = _pct_change(
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

    mean_val = float(local_df[metric_col].mean())
    std_val = float(local_df[metric_col].std(ddof=0))
    anomaly_events: list[dict[str, Any]] = []
    if std_val > 0:
        z_scores = (local_df[metric_col] - mean_val) / std_val
        outlier_mask = z_scores.abs() >= 2.5
        outlier_count = int(outlier_mask.sum())

        if time_col:
            daily = (
                local_df.assign(day=local_df[time_col].dt.date)
                .groupby("day", as_index=False)[metric_col]
                .sum()
                .sort_values("day")
            )
            daily_mean = float(daily[metric_col].mean())
            daily_std = float(daily[metric_col].std(ddof=0))
            if daily_std > 0:
                daily["z"] = (daily[metric_col] - daily_mean) / daily_std
                unusual = daily[daily["z"].abs() >= 2.0].copy()
                unusual["abs_z"] = unusual["z"].abs()
                unusual = unusual.sort_values("abs_z", ascending=False).head(5)
                anomaly_events = [
                    {
                        "date": str(row["day"]),
                        "value": round(float(row[metric_col]), 2),
                        "event_type": "spike" if float(row["z"]) > 0 else "drop",
                        "zscore": round(float(row["z"]), 2),
                        "deviation_pct_from_mean": round(((float(row[metric_col]) - daily_mean) / daily_mean) * 100, 2)
                        if daily_mean
                        else None,
                    }
                    for _, row in unusual.iterrows()
                ]

        signals["anomalies"] = {
            "method": "zscore",
            "threshold": 2.5,
            "count": outlier_count,
            "pct_of_rows": round((outlier_count / max(len(local_df), 1)) * 100.0, 2),
            "events": anomaly_events,
        }

        known_events_text = ""
        if context is not None:
            clarif = context.user_clarifications or {}
            known_events_text = (clarif.get("known_events") or "").strip()

        if anomaly_events:
            highest = anomaly_events[0]
            high_anomaly = abs(highest.get("zscore", 0)) >= 3.5
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
    else:
        signals["anomalies"] = {
            "method": "zscore",
            "threshold": 2.5,
            "count": 0,
            "pct_of_rows": 0.0,
            "events": [],
        }

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
