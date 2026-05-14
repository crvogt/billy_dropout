"""Gemma 4 response parsers.

Two thought-channel surface forms are recognized:

  1. The spec'd Gemma 4 channel envelope:
       <|channel>thought   ...   <channel|>
     (pipe BEFORE `channel` on the opener, AFTER on the closer; the
     asymmetry is per the Gemma 4 spec, not a typo).

  2. A literal `<think>...</think>` block. Verified empirically (see
     `uagent/sandbox/gemma4_thinking/`): on `gemma4:e2b`/`e4b` the
     `<|channel>thought` envelope is not emitted via the Ollama HTTP
     surface even when `<|think|>` is included in the system prompt.
     The strong verbal-instruction approach (system prompt directly
     asks for `<think>...</think>`) is what actually elicits reasoning.
     The strict envelope is kept as a forward-compatibility hedge for
     larger Gemma 4 variants and future Ollama releases.

Strict is tried first; loose is the fallback. This means strict-only
responses behave exactly as before (the existing test suite pins the
strict semantics), while the new prompt's `<think>` outputs are picked
up cleanly.

Lives here rather than in `uagent.experiment.harness` because it has
zero experiment-orchestration dependencies and `uagent.agent.runtime`
needs to call it. The harness re-exports `parse_thought_channel` for
backward compatibility with the Step F directive.

Refs: https://ollama.com/library/gemma4
      https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4
"""

from __future__ import annotations

import re

# Strict pattern: `<|channel>thought ... <channel|>` (Gemma 4 spec).
_STRICT_PATTERN = re.compile(
    r"<\|channel>thought\s*(.*?)\s*<channel\|>",
    re.DOTALL,
)

# Loose pattern: literal `<think>...</think>`. The empirically-observed
# shape under the verbal-instruction prompt. Case-insensitive in case the
# model emits `<Think>` or `<THINK>` under unusual sampling.
_LOOSE_PATTERN = re.compile(
    r"<think>\s*(.*?)\s*</think>",
    re.DOTALL | re.IGNORECASE,
)


def parse_thought_channel(content: object) -> str:
    """Extract Gemma 4 thought text from a model response.

    Tries the strict `<|channel>thought ... <channel|>` envelope first;
    falls back to literal `<think>...</think>` only if strict yields
    nothing. Returns the concatenated text of every well-formed block.
    Multiple blocks join with a blank line. Empty / unmatched / non-string
    inputs return ``""`` — never raises. This function sits on the hot
    path of every event in a multi-hour run; a crash here is expensive.
    """
    if not isinstance(content, str) or not content:
        return ""
    blocks = [m.strip() for m in _STRICT_PATTERN.findall(content) if m.strip()]
    if not blocks:
        blocks = [m.strip() for m in _LOOSE_PATTERN.findall(content) if m.strip()]
    return "\n\n".join(blocks)


def strip_thought_channel(content: object) -> str:
    """Return ``content`` with thought blocks removed and whitespace trimmed.

    Strips both the strict `<|channel>thought ... <channel|>` envelope
    and the loose `<think>...</think>` form, since either may appear in
    the model's response under the current prompt. Used at sites that
    need the model's *final answer* text without the thought channel —
    e.g. when the dispatcher receives no tool call and falls back to
    using the natural-language content as a defer reason. Storing the
    raw content there would contaminate the audit trail with internal
    reasoning that the rest of the pipeline assumes lives only in
    ``agent_reasoning_chain``.

    Non-string input yields ``""``. Never raises.
    """
    if not isinstance(content, str) or not content:
        return ""
    stripped = _STRICT_PATTERN.sub("", content)
    stripped = _LOOSE_PATTERN.sub("", stripped)
    return stripped.strip()
