from __future__ import annotations

from typing import Any

import pandas as pd

from app.data_engine import SchemaInfo


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


def _priority_item(item_type: str, score: int, message: str, recommendation: str) -> dict[str, Any]:
    return {
        "type": item_type,
        "priority_score": score,
        "priority": _priority_label(score),
        "message": message,
        "recommendation": recommendation,
    }


def generate_insight_signals_v2(
    df: pd.DataFrame,
    schema: SchemaInfo,
    semantic_hints: dict | None = None,
) -> dict:
    metric_col = _get_metric_column(df, schema, semantic_hints)
    time_col = _get_time_column(df, schema, semantic_hints)
    dimensions = _get_dimensions(df, schema, semantic_hints)

    signals = {
        "primary_metric": metric_col,
        "time_column": time_col,
        "dimensions": dimensions,
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
        monthly = (
            local_df.assign(period=local_df[time_col].dt.to_period("M").astype(str))
            .groupby("period", as_index=False)[metric_col]
            .sum()
            .sort_values("period")
        )
        weekly = (
            local_df.assign(period=local_df[time_col].dt.to_period("W").astype(str))
            .groupby("period", as_index=False)[metric_col]
            .sum()
            .sort_values("period")
        )

        mom = None
        if len(monthly) >= 2:
            mom = _pct_change(float(monthly.iloc[-1][metric_col]), float(monthly.iloc[-2][metric_col]))

        wow = None
        if len(weekly) >= 2:
            wow = _pct_change(float(weekly.iloc[-1][metric_col]), float(weekly.iloc[-2][metric_col]))

        slope = _trend_slope(monthly[metric_col].tail(6))

        recent_value = float(monthly.iloc[-1][metric_col])
        baseline_window = monthly[metric_col].tail(6)
        baseline_avg = float(baseline_window.mean()) if not baseline_window.empty else None
        baseline_delta = _pct_change(recent_value, baseline_avg) if baseline_avg not in (None, 0) else None

        benchmark_status = "neutral"
        if baseline_delta is not None and baseline_delta >= 5:
            benchmark_status = "above_baseline"
        elif baseline_delta is not None and baseline_delta <= -5:
            benchmark_status = "below_baseline"

        signals["benchmarks"] = {
            "period": str(monthly.iloc[-1]["period"]),
            "recent_value": round(recent_value, 2),
            "baseline_avg_6": round(baseline_avg, 2) if baseline_avg is not None else None,
            "delta_vs_baseline_pct": round(baseline_delta, 2) if baseline_delta is not None else None,
            "status": benchmark_status,
        }

        signals["growth"] = {
            "type": "growth",
            "metric": metric_col,
            "mom_pct": round(mom, 2) if mom is not None else None,
            "wow_pct": round(wow, 2) if wow is not None else None,
            "trend_slope": round(slope, 4) if slope is not None else None,
            "direction": "up" if (mom or 0) >= 0 else "down",
        }

        if mom is not None:
            if mom >= 12:
                signals["priority_insights"].append(
                    _priority_item(
                        "growth_opportunity",
                        65,
                        f"{metric_col} grew {mom:.1f}% MoM.",
                        "Consider scaling inventory and marketing in high-performing segments.",
                    )
                )
            elif mom <= -8:
                signals["priority_insights"].append(
                    _priority_item(
                        "growth_decline",
                        85,
                        f"{metric_col} declined {abs(mom):.1f}% MoM.",
                        "Investigate pricing, conversion funnel, and stock availability for weak periods.",
                    )
                )

        if len(monthly) >= 2 and dimensions:
            last_period = monthly.iloc[-1]["period"]
            prev_period = monthly.iloc[-2]["period"]
            dim_col = dimensions[0]

            by_dim_period = (
                local_df.assign(period=local_df[time_col].dt.to_period("M").astype(str))
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
                        f"{worst['name']} is declining ({worst['growth_pct']:.1f}% vs previous period).",
                        "Review pricing, discount strategy, and demand drivers for this segment.",
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
                )
            )
        elif risk_level == "medium":
            signals["priority_insights"].append(
                _priority_item(
                    "concentration_risk",
                    60,
                    f"Top category concentration is {top_share:.1f}%.",
                    "Create a diversification plan to reduce single-category dependency.",
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

        if anomaly_events:
            highest = anomaly_events[0]
            high_anomaly = abs(highest.get("zscore", 0)) >= 3.5
            signals["priority_insights"].append(
                _priority_item(
                    "anomaly_event",
                    85 if high_anomaly else 55,
                    f"Unusual {highest['event_type']} on {highest['date']} ({metric_col}={highest['value']}).",
                    "Validate if this was campaign-driven, operational disruption, or data-quality issue.",
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

    signals["priority_insights"] = sorted(
        signals["priority_insights"], key=lambda x: x.get("priority_score", 0), reverse=True
    )
    signals["action_items"] = [
        {
            "priority": item["priority"],
            "type": item["type"],
            "recommendation": item["recommendation"],
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
