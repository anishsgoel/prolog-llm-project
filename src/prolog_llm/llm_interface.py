"""LLM interface for Prolog reasoning."""

import json
import os
import re
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import ollama

from prolog_llm.prolog_utils import extract_first_json, parse_predicate
import config


class LLMInterface:
    """
    Encapsulates the Ollama LLM client with configurable parameters.
    Provides methods for generating completions and handling JSON responses.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
        stop_tokens: Optional[list] = None,
        host: Optional[str] = None,
    ):
        """
        Initialize the LLM interface.
        
        Args:
            model: Model name (defaults to config.MODEL)
            temperature: Sampling temperature (defaults to config.LLM_TEMPERATURE)
            num_predict: Max tokens to predict (defaults to config.LLM_NUM_PREDICT)
            stop_tokens: Stop tokens (defaults to config.LLM_STOP_TOKENS)
            host: Ollama host URL (optional)
        """
        self.model = model or config.MODEL
        self.temperature = temperature if temperature is not None else config.LLM_TEMPERATURE
        self.num_predict = num_predict or config.LLM_NUM_PREDICT
        self.stop_tokens = stop_tokens or config.LLM_STOP_TOKENS
        
        if host:
            self.client = ollama.Client(host=host)
        else:
            self.client = ollama.Client()
    
    def generate(
        self, 
        prompt: str, 
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
        stop_tokens: Optional[list] = None,
        format_json: bool = True,
    ) -> str:
        """
        Generate a completion from the LLM.
        
        Args:
            prompt: The prompt to send
            temperature: Override default temperature
            num_predict: Override default max tokens
            stop_tokens: Override default stop tokens
            format_json: Whether to request JSON format
            
        Returns:
            The generated text response
        """
        temp = temperature if temperature is not None else self.temperature
        num_pred = num_predict or self.num_predict
        stops = stop_tokens or self.stop_tokens
        
        try:
            kwargs = dict(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": temp,
                    "num_predict": num_pred,
                    "stop": stops,
                },
            )
            
            if format_json:
                kwargs["format"] = "json"
            
            resp = self.client.generate(**kwargs)
            
        except TypeError:
            try:
                if format_json:
                    kwargs.pop("format", None)
                resp = self.client.generate(**kwargs)
            except Exception as e:
                print("[LLMInterface.generate] Ollama generate() exception:", repr(e))
                return ""
        except Exception as e:
            print("[LLMInterface.generate] Ollama generate() exception:", repr(e))
            return ""
        
        answer = (resp.get("response") or "").strip()
        if not answer:
            answer = (resp.get("thinking") or "").strip()
        
        if not answer:
            print("[LLMInterface.generate] EMPTY response+thinking. Raw resp:", resp)
            return ""
        
        if "...done thinking." in answer:
            answer = answer.split("...done thinking.")[-1].strip()
        
        return answer
    
    def ask(self, prompt: str) -> str:
        """
        Simple alias for generate() with JSON format.
        """
        return self.generate(prompt, format_json=True)
    
    def ask_with_retry(self, prompt: str, repair_schema: Optional[str] = None) -> str:
        """
        Generate a completion, retrying with a strict JSON-only prompt if needed.
        
        Args:
            prompt: The initial prompt
            repair_schema: JSON schema to use in the repair prompt (if needed)
            
        Returns:
            The generated text (should contain valid JSON)
        """
        raw = self.ask(prompt).strip()
        
        try:
            _ = extract_first_json(raw)
            return raw
        except Exception:
            pass
        
        if not repair_schema:
            return raw
        
        repair_prompt = f"""
Return ONLY valid JSON. No explanation, no prose, no markdown.

You MUST output JSON that matches this schema exactly:
{repair_schema}

If you cannot comply, output:
{{"by_atom":{{}}}}
""".strip()
        
        return self.ask(repair_prompt).strip()
    
    def nl_to_prolog(self, nl_text: str, start_index: int = 1) -> list[str]:
        """
        Convert natural language KB description to Prolog clauses.
        
        Args:
            nl_text: Natural language description of the KB
            start_index: Starting line number for clauses
            
        Returns:
            List of numbered Prolog clauses
        """
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

\"\"\"{nl_text}\"\"\"

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
        
        raw = self.ask(prompt).strip()
        
        try:
            data = json.loads(extract_first_json(raw))
        except Exception as e:
            print("[LLMInterface.nl_to_prolog] JSON parse error:", e)
            print("Raw LLM output:", raw[:800])
            return []
        
        raw_clauses = data.get("clauses", [])
        if not isinstance(raw_clauses, list):
            print("[LLMInterface.nl_to_prolog] 'clauses' field is not a list:", raw_clauses)
            return []
        
        cleaned = []
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
                clause += "."
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
                print("[LLMInterface.nl_to_prolog] Discarding unparsable clause:", clause)
                continue
            
            cleaned.append(clause)
        
        numbered = [f"{i}. {clause}" for i, clause in enumerate(cleaned, start=start_index)]
        return numbered
    
    def __repr__(self) -> str:
        return f"LLMInterface(model={self.model!r}, temp={self.temperature})"


# Global instance for backwards compatibility
_llm_interface: Optional[LLMInterface] = None


def get_llm_interface() -> LLMInterface:
    """Get or create the global LLM interface instance."""
    global _llm_interface
    if _llm_interface is None:
        _llm_interface = LLMInterface()
    return _llm_interface


# Backwards compatibility functions
def ask_llm(prompt: str) -> str:
    """Backwards-compatible ask_llm function."""
    return get_llm_interface().ask(prompt)


def llm_json_only(prompt: str, repair_schema: str) -> str:
    """Backwards-compatible llm_json_only function."""
    return get_llm_interface().ask_with_retry(prompt, repair_schema)


def nl_kb_to_prolog_kb(nl_kb_text: str, start_index: int = 1) -> list[str]:
    """Backwards-compatible nl_kb_to_prolog_kb function."""
    return get_llm_interface().nl_to_prolog(nl_kb_text, start_index)


# Alias for backwards compatibility
nl_to_prolog_kb = nl_kb_to_prolog_kb
