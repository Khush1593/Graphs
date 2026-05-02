from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from app.data_engine import SchemaInfo


@dataclass
class SemanticHints:
    primary_metric: str | None
    time_column: str | None
    dimensions: List[str]
    metric_candidates: List[str]

    def to_dict(self) -> dict:
        return {
            "primary_metric": self.primary_metric,
            "time_column": self.time_column,
            "dimensions": self.dimensions,
            "metric_candidates": self.metric_candidates,
        }


def _score_metric_name(column_name: str) -> int:
    name = column_name.lower()
    strong = ["revenue", "sales", "amount", "gmv", "profit", "value", "total"]
    medium = ["price", "cost", "margin", "income", "earning", "net"]
    score = 0
    for token in strong:
        if token in name:
            score += 3
    for token in medium:
        if token in name:
            score += 1
    return score


def _score_dimension_name(column_name: str) -> int:
    name = column_name.lower()
    strong = ["region", "category", "segment", "channel", "product", "brand", "country", "city", "state"]
    return sum(1 for token in strong if token in name)


def infer_semantic_hints(df: pd.DataFrame, schema: SchemaInfo) -> SemanticHints:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    object_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    metric_candidates = sorted(numeric_cols, key=lambda c: _score_metric_name(c), reverse=True)

    primary_metric = schema.numeric_column
    if metric_candidates and _score_metric_name(metric_candidates[0]) > 0:
        primary_metric = metric_candidates[0]

    time_column = schema.date_column

    ranked_dimensions = []
    for col in object_cols:
        unique_ratio = df[col].nunique(dropna=True) / max(len(df), 1)
        if unique_ratio <= 0 or unique_ratio >= 0.8:
            continue
        score = _score_dimension_name(col)
        ranked_dimensions.append((col, score, unique_ratio))

    ranked_dimensions.sort(key=lambda x: (x[1], -x[2]), reverse=True)
    dimensions = [col for col, _, _ in ranked_dimensions[:3]]

    if schema.category_column and schema.category_column not in dimensions:
        dimensions.insert(0, schema.category_column)
        dimensions = dimensions[:3]

    if primary_metric and primary_metric in dimensions:
        dimensions = [d for d in dimensions if d != primary_metric]
    if time_column and time_column in dimensions:
        dimensions = [d for d in dimensions if d != time_column]

    return SemanticHints(
        primary_metric=primary_metric,
        time_column=time_column,
        dimensions=dimensions,
        metric_candidates=metric_candidates[:5],
    )
