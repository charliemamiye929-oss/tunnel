"""Claude client used by the breeder.

Thin wrapper over the official ``anthropic`` SDK that adds the things this
project needs: structured JSON responses, retry/robustness, token accounting,
and a hard cost ceiling so a 1000-generation run cannot quietly run up a bill.

If the SDK is not installed or no credentials are configured, :class:`Claude`
reports ``available == False`` and the evolution loop falls back to its
non-LLM genetic operators instead of failing.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:  # the SDK is optional: offline runs do not need it
    import anthropic
except ImportError:  # pragma: no cover - exercised only without the dependency
    anthropic = None  # type: ignore

DEFAULT_MODEL = "claude-opus-5"

#: USD per million tokens (input, output), for run cost accounting.
PRICING: Dict[str, tuple] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


class LLMError(RuntimeError):
    pass


class BudgetExceeded(LLMError):
    """Raised when the configured USD ceiling for a run is reached."""


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    errors: int = 0
    models: Dict[str, int] = field(default_factory=dict)

    def add(self, model: str, usage: Any) -> float:
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        cache_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        price_in, price_out = PRICING.get(model, (0.0, 0.0))
        cost = ((in_tok + cache_write * 1.25 + cache_read * 0.1) * price_in
                + out_tok * price_out) / 1_000_000.0
        self.calls += 1
        self.input_tokens += in_tok + cache_read + cache_write
        self.output_tokens += out_tok
        self.cache_read_tokens += cache_read
        self.cost_usd += cost
        self.models[model] = self.models.get(model, 0) + 1
        return cost

    def to_dict(self) -> Dict[str, Any]:
        return {"calls": self.calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cache_read_tokens": self.cache_read_tokens,
                "cost_usd": round(self.cost_usd, 4), "errors": self.errors,
                "models": dict(self.models)}


@dataclass
class Response:
    text: str
    data: Optional[Any]        # parsed JSON when a schema was requested
    model: str
    cost_usd: float
    stop_reason: str = ""


class Claude:
    """Minimal, retrying Claude client with cost accounting."""

    def __init__(self, model: str = DEFAULT_MODEL, *, max_tokens: int = 16_000,
                 effort: str = "high", api_key: Optional[str] = None,
                 budget_usd: float = 0.0, timeout: float = 600.0,
                 max_retries: int = 4, verbose: bool = False):
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.budget_usd = float(budget_usd)
        self.max_retries = max_retries
        self.verbose = verbose
        self.usage = Usage()
        self._client = None
        self._unavailable_reason = ""

        if anthropic is None:
            self._unavailable_reason = (
                "the 'anthropic' package is not installed (pip install anthropic)")
            return
        if not (api_key or _has_credentials()):
            self._unavailable_reason = (
                "no Anthropic credentials found (set ANTHROPIC_API_KEY, or run "
                "`ant auth login`); breeding will fall back to mutation only")
            return
        try:
            kwargs: Dict[str, Any] = {"timeout": timeout, "max_retries": 2}
            if api_key:
                kwargs["api_key"] = api_key
            self._client = anthropic.Anthropic(**kwargs)
        except Exception as exc:  # credentials malformed
            self._unavailable_reason = f"could not construct client: {exc}"

    # ------------------------------------------------------------------ state
    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def unavailable_reason(self) -> str:
        return self._unavailable_reason

    def budget_left(self) -> float:
        if self.budget_usd <= 0:
            return float("inf")
        return max(0.0, self.budget_usd - self.usage.cost_usd)

    # ------------------------------------------------------------------ calls
    def complete(self, prompt: str, *, system: str = "",
                 schema: Optional[Dict[str, Any]] = None,
                 max_tokens: Optional[int] = None,
                 effort: Optional[str] = None,
                 model: Optional[str] = None) -> Response:
        """One request.  With ``schema``, the reply is guaranteed valid JSON."""
        if not self.available:
            raise LLMError(f"Claude is unavailable: {self._unavailable_reason}")
        if self.budget_left() <= 0:
            raise BudgetExceeded(
                f"LLM budget of ${self.budget_usd:.2f} exhausted "
                f"(spent ${self.usage.cost_usd:.2f})")

        model = model or self.model
        output_config: Dict[str, Any] = {"effort": effort or self.effort}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}

        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "adaptive"},
            "output_config": output_config,
        }
        if system:
            # The system prompt is identical across generations, so caching it
            # makes every breeding call after the first substantially cheaper.
            kwargs["system"] = [{"type": "text", "text": system,
                                 "cache_control": {"type": "ephemeral"}}]

        last: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                # Streaming avoids HTTP timeouts on long, high-effort replies.
                with self._client.messages.stream(**kwargs) as stream:
                    message = stream.get_final_message()
                break
            except Exception as exc:  # noqa: BLE001 - retry policy below
                last = exc
                self.usage.errors += 1
                if not _retryable(exc) or attempt == self.max_retries - 1:
                    raise LLMError(f"Claude request failed: {exc}") from exc
                delay = min(2 ** attempt + 0.5 * attempt, 30.0)
                if self.verbose:
                    print(f"  [llm] retry {attempt + 1}/{self.max_retries} in {delay:.0f}s: {exc}")
                time.sleep(delay)
        else:  # pragma: no cover - loop always breaks or raises
            raise LLMError(f"Claude request failed: {last}")

        cost = self.usage.add(model, message.usage)
        text = "".join(b.text for b in message.content if getattr(b, "type", "") == "text")
        stop_reason = getattr(message, "stop_reason", "") or ""
        if stop_reason == "refusal":
            raise LLMError("Claude declined this request "
                           f"({getattr(getattr(message, 'stop_details', None), 'category', 'unknown')})")

        data = None
        if schema is not None:
            data = extract_json(text)
            if data is None:
                raise LLMError("structured response was not valid JSON")
        return Response(text=text, data=data, model=model, cost_usd=cost,
                        stop_reason=stop_reason)


def _has_credentials() -> bool:
    """Env credentials, or a profile written by ``ant auth login``.

    An unset ANTHROPIC_API_KEY does not mean there are no credentials: the SDK
    also reads ANTHROPIC_AUTH_TOKEN, workload identity federation env vars, and
    OAuth profiles on disk.
    """
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    if os.environ.get("ANTHROPIC_IDENTITY_TOKEN") or os.environ.get("ANTHROPIC_IDENTITY_TOKEN_FILE"):
        return True
    config = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
        os.path.expanduser("~"), ".config")
    profile_dir = os.path.join(config, "anthropic")
    return os.path.isdir(profile_dir) and bool(os.listdir(profile_dir))


def _retryable(exc: Exception) -> bool:
    if anthropic is None:  # pragma: no cover
        return False
    retry_types = tuple(
        t for t in (getattr(anthropic, "RateLimitError", None),
                    getattr(anthropic, "APIConnectionError", None),
                    getattr(anthropic, "APITimeoutError", None),
                    getattr(anthropic, "InternalServerError", None))
        if t is not None
    )
    if retry_types and isinstance(exc, retry_types):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and (status == 429 or status >= 500)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Optional[Any]:
    """Best-effort JSON recovery from a model reply."""
    if not text:
        return None
    for candidate in _candidates(text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _candidates(text: str) -> List[str]:
    out = [text.strip()]
    for m in _FENCE.finditer(text):
        out.append(m.group(1).strip())
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if 0 <= start < end:
            out.append(text[start:end + 1])
    return out
