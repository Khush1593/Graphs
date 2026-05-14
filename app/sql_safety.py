"""AST-based safety layer for LLM-generated SQL.

The LLM is a powerful but unreliable code generator: it can emit destructive
statements, hallucinate column names, reference unknown tables, or return
unbounded result sets that crash Streamlit. Substring checks (`"DROP" in sql`)
catch the obvious cases but miss things like quoted identifiers, comments, and
nested subqueries — and they also reject legitimate queries (a column literally
named ``updated_at`` contains ``UPDATE``). We parse the query with sqlglot,
walk the tree, and only return SQL we've structurally verified.

Public surface:
    SQLValidationError       — raised on any rejection, with a ``code`` attr
                               so the caller can decide whether to surface a
                               retry to the LLM or a hard error to the user.
    validate_select_sql(...) — returns a sanitized, LIMIT-capped SQL string.
"""
from __future__ import annotations

from typing import Iterable

import sqlglot
from sqlglot import expressions as exp


DEFAULT_MAX_ROWS = 1000
DEFAULT_DIALECT = "duckdb"
DEFAULT_ALLOWED_TABLES: tuple[str, ...] = ("data",)


# Any of these appearing anywhere in the AST is an immediate rejection.
_FORBIDDEN_TYPES: tuple[type, ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.Create,
    exp.TruncateTable,
    exp.Merge,
    exp.Command,  # sqlglot's catch-all for unparseable DDL/DCL statements
)


class SQLValidationError(Exception):
    """Raised when an LLM-generated query fails AST validation.

    ``code`` is a short machine-readable token so callers can branch on the
    failure mode (e.g. give the LLM a retry hint for ``unknown_column`` but
    refuse outright for ``destructive_statement``).
    """

    def __init__(self, code: str, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


def validate_select_sql(
    sql: str,
    allowed_columns: Iterable[str],
    *,
    allowed_tables: Iterable[str] = DEFAULT_ALLOWED_TABLES,
    max_rows: int = DEFAULT_MAX_ROWS,
    dialect: str = DEFAULT_DIALECT,
) -> str:
    """Parse, validate, and return a safe, LIMIT-capped SQL string.

    Raises SQLValidationError on any of: empty / unparseable input, multiple
    statements, non-SELECT root, destructive nodes anywhere in the tree,
    references to disallowed tables, or references to columns outside the
    provided schema.
    """
    if not sql or not sql.strip():
        raise SQLValidationError("empty_query", "SQL string is empty.")

    try:
        parsed = sqlglot.parse(sql, read=dialect)
    except sqlglot.errors.ParseError as e:
        raise SQLValidationError("parse_error", f"Could not parse SQL: {e}") from e

    statements = [s for s in parsed if s is not None]
    if len(statements) == 0:
        raise SQLValidationError("empty_query", "No statement found.")
    if len(statements) > 1:
        raise SQLValidationError(
            "multiple_statements",
            "Only a single SELECT statement is allowed.",
        )

    tree = statements[0]

    if not isinstance(tree, exp.Select):
        raise SQLValidationError(
            "non_select_root",
            f"Top-level statement must be SELECT, got {type(tree).__name__}.",
        )

    for node in tree.find_all(*_FORBIDDEN_TYPES):
        raise SQLValidationError(
            "destructive_statement",
            f"Destructive operation not permitted ({type(node).__name__}).",
        )

    allowed_tables_lower = {t.lower() for t in allowed_tables}
    cte_names = {cte.alias.lower() for cte in tree.find_all(exp.CTE) if cte.alias}

    for tbl in tree.find_all(exp.Table):
        name = (tbl.name or "").lower()
        if not name:
            continue
        if name in cte_names:
            continue
        if name not in allowed_tables_lower:
            raise SQLValidationError(
                "disallowed_table",
                f"Table {tbl.name!r} is not in the allowed set.",
                details={"table": tbl.name, "allowed": sorted(allowed_tables_lower)},
            )

    allowed_cols_lower = {c.lower() for c in allowed_columns}
    local_aliases = _collect_aliases(tree)
    unknown: list[str] = []
    for col in tree.find_all(exp.Column):
        name = (col.name or "").lower()
        if not name or name == "*":
            continue
        if name in allowed_cols_lower or name in local_aliases or name in cte_names:
            continue
        unknown.append(col.name)
    if unknown:
        raise SQLValidationError(
            "unknown_column",
            f"Unknown column(s): {sorted(set(unknown))}",
            details={"columns": sorted(set(unknown))},
        )

    safe_tree = _enforce_limit(tree, max_rows)
    return safe_tree.sql(dialect=dialect)


def _collect_aliases(tree: exp.Expression) -> set[str]:
    """Names that are valid column references because the query itself
    introduces them (column aliases, table aliases, CTE outputs)."""
    aliases: set[str] = set()
    for node in tree.find_all(exp.Alias):
        alias = node.alias
        if alias:
            aliases.add(alias.lower())
    for node in tree.find_all(exp.TableAlias):
        if node.name:
            aliases.add(node.name.lower())
    return aliases


def _enforce_limit(tree: exp.Select, max_rows: int) -> exp.Select:
    """Inject a LIMIT, or tighten an existing LIMIT that's above the cap.

    We replace rather than wrap so the emitted SQL stays readable and the
    DuckDB planner doesn't see a redundant outer SELECT.
    """
    existing = tree.args.get("limit")
    if existing is None:
        return tree.limit(max_rows)

    try:
        current = int(existing.expression.this)
    except (AttributeError, TypeError, ValueError):
        # Non-integer LIMIT (e.g. parameter, expression) — force the cap.
        return tree.limit(max_rows)

    if current > max_rows:
        return tree.limit(max_rows)
    return tree
