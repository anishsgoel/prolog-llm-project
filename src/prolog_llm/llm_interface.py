"""LLM interface for Prolog reasoning."""

import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import ollama

from prolog_llm.prolog_utils import extract_first_json
import config


client = ollama.Client()


def ask_llm(prompt: str) -> str:
    """
    Drop-in patch: prevents 'reasoning' rambles and forces a JSON payload.
    - Adds stop tokens to cut off commentary early.
    - Falls back to `thinking` field if `response` is empty (common with gpt-oss).
    - If client supports format='json', it will be used; otherwise it will ignore it.
    """
    try:
        kwargs = dict(
            model=config.MODEL,
            prompt=prompt,
            options={
                "temperature": config.LLM_TEMPERATURE,
                "num_predict": config.LLM_NUM_PREDICT,
                "stop": config.LLM_STOP_TOKENS,
            },
        )

        kwargs["format"] = "json"

        resp = client.generate(**kwargs)

    except TypeError:
        try:
            kwargs.pop("format", None)
            resp = client.generate(**kwargs)
        except Exception as e:
            print("[ask_llm] Ollama generate() exception:", repr(e))
            return ""
    except Exception as e:
        print("[ask_llm] Ollama generate() exception:", repr(e))
        return ""

    answer = (resp.get("response") or "").strip()
    if not answer:
        answer = (resp.get("thinking") or "").strip()

    if not answer:
        print("[ask_llm] EMPTY response+thinking. Raw resp:", resp)
        return ""

    if "...done thinking." in answer:
        answer = answer.split("...done thinking.")[-1].strip()

    return answer


def llm_json_only(prompt: str, repair_schema: str) -> str:
    """
    Calls ask_llm. If no JSON object is present, retries once with a strict JSON-only repair prompt.
    Returns raw text (which should contain JSON).
    """
    raw = ask_llm(prompt).strip()
    try:
        _ = extract_first_json(raw)
        return raw
    except Exception:
        pass

    repair_prompt = f"""
Return ONLY valid JSON. No explanation, no prose, no markdown.

You MUST output JSON that matches this schema exactly:
{repair_schema}

If you cannot comply, output:
{{"by_atom":{{}}}}
""".strip()

    raw2 = ask_llm(repair_prompt).strip()
    return raw2


def nl_to_prolog_kb(nl_kb_text: str, start_index: int = 1) -> list[str]:
    """Convert natural language KB description to Prolog clauses."""
    nl_kb_text = (nl_kb_text or "").strip()
    if not nl_kb_text:
        return []

    from .prolog_utils import parse_predicate
    import re

    prompt = f"""
You are a Prolog formalization assistant.

The user will give you a natural-language description of a small domain,
including objects, relationships, and logical rules.

Your job is to convert that description into a set of Prolog clauses
(facts and rules).

Guidelines:
- Use lowercase atoms for concrete entities (e.g. union_square, times_square,
  grand_central, bryant_park).
- Use uppercase identifiers for variables (e.g. X, Y, Z).
- Choose predicate names that are short, descriptive, and consistent,
  for example: connected/2, reachable/2, located_in/2, etc.
- A fact must look like:
    connected(times_square, bryant_park).
- A rule must look like:
    reachable(X, Z) :- connected(X, Y), reachable(Y, Z).
- Every clause MUST end with a single period '.'.
- Do NOT include any line numbers in your output clauses.
- Do NOT add explanations or comments in the Prolog code.

Here is the NATURAL LANGUAGE description of the knowledge base:

\"\"\"{nl_kb_text}\"\"\"

Respond ONLY in this JSON format (and nothing else):

{{
  "clauses": [
    {{
      "clause": "connected(union_square, times_square)."
    }},
    {{
      "clause": "reachable(X, Y) :- connected(X, Y)."
    }}
  ]
}}
""".strip()

    raw = ask_llm(prompt).strip()
    try:
        data = json.loads(extract_first_json(raw))
    except Exception as e:
        print("[nl_to_prolog_kb] JSON parse error:", e)
        print("Raw LLM output:", raw[:800])
        return []

    raw_clauses = data.get("clauses", [])
    if not isinstance(raw_clauses, list):
        print("[nl_to_prolog_kb] 'clauses' field is not a list:", raw_clauses)
        return []

    cleaned_clauses = []
    for item in raw_clauses:
        if isinstance(item, str):
            clause = item.strip()
        elif isinstance(item, dict):
            clause = (item.get("clause") or "").strip()
        else:
            continue

        if not clause:
            continue

        m_num = re.match(r"^\s*(\d+)\.\s*(.+)$", clause)
        if m_num:
            clause = m_num.group(2).strip()

        clause = clause.rstrip()
        if not clause.endswith("."):
            clause = clause + "."
        else:
            clause = re.sub(r"\.+$", ".", clause)

        body_str = clause[:-1].strip()
        if ":-" in body_str:
            head_part, _ = body_str.split(":-", 1)
            head = head_part.strip()
        else:
            head = body_str

        parsed_head = parse_predicate(head)
        if parsed_head is None:
            print("[nl_to_prolog_kb] Discarding unparsable clause:", clause)
            continue

        cleaned_clauses.append(clause)

    numbered_clauses = []
    next_num = start_index
    for clause in cleaned_clauses:
        numbered_clauses.append(f"{next_num}. {clause}")
        next_num += 1

    return numbered_clauses
