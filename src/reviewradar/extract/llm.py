"""The model, behind a Protocol.

`OfflineLlm` is the default and is what the entire test suite and CI run against. It is a
**test double, not a second classifier**: it returns canned payloads keyed on a hash of
the prompt and otherwise abstains. That is deliberate. If the offline implementation
extracted anything by rule, it would become a second extraction path competing with the
one being measured, and the scoreboard's baseline-versus-model delta would stop meaning
what it says.

`VertexLlm` is the real one, imported lazily so that a clone with no cloud account still
runs everything (D-006).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

if TYPE_CHECKING:
    # Type-only: never imported at runtime, so D-006 holds - a clone without the
    # `vertex` extra still runs everything. Without the extra installed, the
    # `anthropic.*` ignore_missing_imports override makes this resolve to Any.
    from anthropic.types import ToolParam

SYSTEM_PROMPT = """\
You extract corporate-action facts from SEC 8-K filings for an index calculation team.

Rules, in order of importance:

1. Every field you return MUST carry a span: the doc_id, the character offsets into the
   plain text you were given, and the exact substring at those offsets. If you cannot
   point at the words, do not return the field.
2. Return only what the filing states. Do not infer an ex-date from a record date, do not
   convert between ratio conventions, and do not resolve a ticker you were not given.
3. If the filing announces no corporate action that an index calculator would act on,
   return event_type "no_index_action". That is a correct and common answer.
4. Ratios are exact fractions of new shares to old: a three-for-one forward split is
   "3/1"; a one-for-eight reverse split is "1/8".
5. Prefer the exhibit press release over the filing body when they disagree, and cite
   whichever you used.
"""


@dataclass(frozen=True, slots=True)
class LlmResponse:
    payload: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model: str = "offline"


@runtime_checkable
class LlmClient(Protocol):
    """The only surface the pipeline knows about."""

    name: str

    def extract(self, prompt: str, schema: Mapping[str, Any]) -> LlmResponse: ...


def prompt_key(prompt: str) -> str:
    """Stable key for a prompt. Used by the offline double and by response caching."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


@dataclass
class OfflineLlm:
    """Deterministic double. No network, no credentials, no extraction.

    Returns a canned payload when one is registered for the prompt's key, and abstains
    otherwise. Abstention is `no_index_action` with no span, which the pipeline treats as
    an unsupported classification and scores accordingly - so the offline path exercises
    the whole scaffolding including its failure handling.
    """

    canned: dict[str, dict[str, Any]] = field(default_factory=dict)
    name: str = "offline"
    calls: int = 0

    def register(self, prompt: str, payload: dict[str, Any]) -> None:
        self.canned[prompt_key(prompt)] = payload

    def extract(self, prompt: str, schema: Mapping[str, Any]) -> LlmResponse:
        del schema
        self.calls += 1
        payload = self.canned.get(
            prompt_key(prompt),
            {"event_type": {"value": "no_index_action", "span": None}},
        )
        return LlmResponse(
            payload=payload,
            input_tokens=len(prompt) // 4,
            output_tokens=len(json.dumps(payload)) // 4,
            latency_ms=0.0,
            model="offline",
        )


class VertexLlm:
    """Claude on Vertex AI.

    Vertex rather than the direct API (D-005): one GCP project, service-account
    authentication, no second vendor relationship, and no key to leak into a repository.

    Imported lazily. `pip install reviewradar` without the `vertex` extra must still run
    the full pipeline offline.
    """

    def __init__(
        self,
        *,
        project: str,
        region: str = "us-east5",
        model: str = "claude-haiku-4-5@20251001",
        max_tokens: int = 1500,
        max_retries: int = 3,
    ) -> None:
        try:
            from anthropic import AnthropicVertex
        except ImportError as exc:  # pragma: no cover - exercised only with the extra
            raise RuntimeError(
                "the Vertex client needs the `vertex` extra: `uv sync --extra vertex`. "
                "The offline pipeline does not require it."
            ) from exc
        self.name = f"vertex:{model}"
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._client = AnthropicVertex(project_id=project, region=region)

    def extract(self, prompt: str, schema: Mapping[str, Any]) -> LlmResponse:
        """One structured-output call, with retries on transient failure.

        Schema-invalid output is retried, then allowed to fail. A partial parse of
        malformed JSON is never returned - a half-read ratio is worse than no ratio.
        """
        tool: ToolParam = {
            "name": "record_corporate_action",
            "description": "Record the corporate action this filing announces.",
            "input_schema": dict(schema),
        }
        last: Exception | None = None
        for attempt in range(self.max_retries):
            started = time.monotonic()
            try:
                message = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=SYSTEM_PROMPT,
                    tools=[tool],
                    tool_choice={"type": "tool", "name": "record_corporate_action"},
                    messages=[{"role": "user", "content": prompt}],
                )
                for block in message.content:
                    if block.type == "tool_use":
                        # tool_use input is a JSON object by API contract; the SDK
                        # types it `object`.
                        return LlmResponse(
                            payload=dict(cast(Mapping[str, Any], block.input)),
                            input_tokens=message.usage.input_tokens,
                            output_tokens=message.usage.output_tokens,
                            latency_ms=(time.monotonic() - started) * 1000,
                            model=self.model,
                        )
                raise RuntimeError("no tool_use block in the response")
            except Exception as exc:
                last = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
        raise RuntimeError(f"Vertex extraction failed after {self.max_retries} attempts: {last}")
