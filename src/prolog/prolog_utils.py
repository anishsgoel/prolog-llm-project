def is_variable(s: str) -> bool:
    """Prolog-ish variable check: starts with uppercase letter or '_'."""
    s = s.strip()
    return bool(s) and (s[0].isupper() or s[0] == "_")
