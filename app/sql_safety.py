def is_safe_query(sql: str) -> bool:
    forbidden = ["DELETE", "UPDATE", "DROP", "ALTER", "INSERT", "TRUNCATE", "CREATE", "REPLACE"]
    upper_sql = sql.upper()
    if ";" in sql.strip().rstrip(";"):
        return False
    return upper_sql.strip().startswith("SELECT") and not any(word in upper_sql for word in forbidden)
