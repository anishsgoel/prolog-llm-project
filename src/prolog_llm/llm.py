"""LLM interface for model querying."""

import os
import time
from typing import Optional

import ollama

from prolog_llm.prolog_utils import extract_first_json
import config
from log_setup import ERROR_LOGGER, LLM_LOGGER, get_logger

_llm_log = get_logger(LLM_LOGGER)
_err_log = get_logger(ERROR_LOGGER)

# Retry/abort policy for backend failures. Env-overridable so it can be tuned on
# a cluster without editing global.yaml (the values are not model behaviour, they
# are infrastructure tolerances).
_MAX_ATTEMPTS = int(os.environ.get("PLLM_LLM_MAX_ATTEMPTS", "3"))
_BACKOFF_S = float(os.environ.get("PLLM_LLM_BACKOFF_S", "2.0"))
_MAX_CONSECUTIVE_EMPTY = int(os.environ.get("PLLM_LLM_MAX_EMPTY", "8"))


class LLMError(RuntimeError):
    """Base class for LLM failures that must NOT be mistaken for model output."""


class LLMTransportError(LLMError):
    """The backend could not be reached, or spoke a malformed protocol.

    Raised only once retries are exhausted. Deliberately *not* swallowed: the
    20260804 baseline run produced 68 all-zero gpt-oss:120b rows because the
    ollama runner had been SIGKILLed at load, every ``/api/chat`` returned HTTP
    500, and ``generate()`` turned that into ``""``. Every caller then read the
    empty string as "the model proposed nothing", so an infrastructure outage
    entered the results CSV as data, with an empty ``error`` column.
    """


class LLMEmptyResponseError(LLMError):
    """The backend answered OK but with no usable text, repeatedly.

    A single empty completion is a model quirk and is tolerated (returns ``""``).
    ``_MAX_CONSECUTIVE_EMPTY`` in a row means the backend is not doing useful
    work, which is again something that must fail loudly rather than silently
    score as zero hypotheses.
    """


class LLMInterface:
    """
    Encapsulates the Ollama LLM client with configurable parameters.
    Provides methods for generating completions and handling JSON responses.

    Failure contract: transport/protocol failures raise :class:`LLMTransportError`
    after ``max_attempts`` tries; a run of empty-but-successful responses raises
    :class:`LLMEmptyResponseError`. Both propagate up through ``run_method`` to
    ``batch_run._evaluate``, which records them in the CSV's ``error`` column —
    so a broken cell is visibly broken instead of looking like a bad result.
    Per-instance counters (``n_calls``, ``n_empty_responses``,
    ``n_transport_failures``) are available for diagnostics.
    """

    def __init__(
            self,
            model: Optional[str] = None,
            temperature: Optional[float] = None,
            num_predict: Optional[int] = None,
            stop_tokens: Optional[list] = None,
            host: Optional[str] = None,
            max_attempts: Optional[int] = None,
            backoff_s: Optional[float] = None,
            max_consecutive_empty: Optional[int] = None,
    ):
        """
        Initialize the LLM interface.

        Args:
            model: Model name (defaults to config.MODEL)
            temperature: Sampling temperature (defaults to config.LLM_TEMPERATURE)
            num_predict: Max tokens to predict (defaults to config.LLM_NUM_PREDICT)
            stop_tokens: Stop tokens (defaults to config.LLM_STOP_TOKENS)
            host: Ollama host URL (optional)
            max_attempts: Tries per call before raising LLMTransportError
                (default 3, or $PLLM_LLM_MAX_ATTEMPTS)
            backoff_s: Base delay between retries, doubled each attempt
                (default 2.0, or $PLLM_LLM_BACKOFF_S)
            max_consecutive_empty: Empty-but-OK responses in a row before raising
                LLMEmptyResponseError; 0 disables the check
                (default 8, or $PLLM_LLM_MAX_EMPTY)
        """
        self.model = model or config.MODEL
        self.temperature = temperature if temperature is not None else config.LLM_TEMPERATURE
        self.num_predict = num_predict or config.LLM_NUM_PREDICT
        self.stop_tokens = stop_tokens or config.LLM_STOP_TOKENS

        self.max_attempts = max(1, int(max_attempts if max_attempts is not None else _MAX_ATTEMPTS))
        self.backoff_s = float(backoff_s if backoff_s is not None else _BACKOFF_S)
        self.max_consecutive_empty = int(
            max_consecutive_empty if max_consecutive_empty is not None else _MAX_CONSECUTIVE_EMPTY
        )

        # Diagnostics: how healthy was the backend during this run?
        self.n_calls = 0
        self.n_empty_responses = 0
        self.n_transport_failures = 0
        self._consecutive_empty = 0

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
            The generated text response, or "" if the model answered with nothing
            usable (an occasional model quirk, tolerated).

        Raises:
            LLMTransportError: the backend failed on all ``max_attempts`` tries.
            LLMEmptyResponseError: ``max_consecutive_empty`` empty answers in a row.
        """
        temp = temperature if temperature is not None else self.temperature
        num_pred = num_predict or self.num_predict
        stops = stop_tokens or self.stop_tokens

        _llm_log.debug("PROMPT (model=%s, temperature=%s, num_predict=%s):\n%s",
                       self.model, temp, num_pred, prompt)

        # NOTE: `options` and `format` stay disabled, exactly as before, so this
        # change does not alter what the model is asked and results stay
        # comparable with earlier runs. Only FAILURE handling changed. (Side
        # effect worth knowing: temperature/num_predict/stop are therefore NOT
        # applied -- see run_self_consistency's note on sampler nondeterminism.)
        options = {
            "temperature": temp,
            "num_predict": num_pred,
            "stop": stops,
        }

        self.n_calls += 1
        msg = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                resp = self.client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    #options=options,
                    #format="json" if format_json else None,
                )
                msg = resp["message"]   # KeyError here = malformed protocol -> retry
                break
            except Exception as exc:
                self.n_transport_failures += 1
                _llm_log.error("Ollama chat() failed (attempt %d/%d): %r",
                               attempt, self.max_attempts, exc)
                if attempt >= self.max_attempts:
                    _err_log.exception(
                        "LLMInterface.generate: giving up after %d attempt(s); "
                        "model=%s host=%s", self.max_attempts, self.model,
                        os.environ.get("OLLAMA_HOST", "<default>"))
                    raise LLMTransportError(
                        f"Ollama chat() failed {self.max_attempts}x for model "
                        f"{self.model!r} on {os.environ.get('OLLAMA_HOST', '<default host>')}: "
                        f"{exc.__class__.__name__}: {exc}"
                    ) from exc
                time.sleep(self.backoff_s * (2 ** (attempt - 1)))

        answer = (msg.get("content") or "").strip()
        if not answer:
            answer = (msg.get("thinking") or "").strip()

        if not answer:
            self.n_empty_responses += 1
            self._consecutive_empty += 1
            _llm_log.warning("EMPTY response+thinking (%d in a row). Raw resp: %s",
                             self._consecutive_empty, msg)
            if self.max_consecutive_empty and self._consecutive_empty >= self.max_consecutive_empty:
                _err_log.error(
                    "LLMInterface.generate: %d consecutive empty responses from "
                    "model=%s -- treating the backend as dead.",
                    self._consecutive_empty, self.model)
                raise LLMEmptyResponseError(
                    f"{self._consecutive_empty} consecutive empty responses from "
                    f"model {self.model!r}; refusing to report this as a result"
                )
            return ""

        self._consecutive_empty = 0

        if "...done thinking." in answer:
            answer = answer.split("...done thinking.")[-1].strip()

        _llm_log.debug("RESPONSE:\n%s", answer)

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
            _err_log.exception("ask_with_retry: no valid JSON in first response; retrying with repair prompt. raw=%r", raw)

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
