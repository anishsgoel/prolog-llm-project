"""LLM interface for model querying."""

from typing import Optional

import ollama

from prolog_llm.prolog_utils import extract_first_json
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