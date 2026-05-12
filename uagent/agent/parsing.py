"""Gemma 4 response parsers.

Currently hosts `parse_thought_channel`, which extracts the thinking-mode
thought channel from a model's natural-language response. Lives here
rather than in `uagent.experiment.harness` because it has zero
experiment-orchestration dependencies and `uagent.agent.runtime` needs
to call it (a parser-in-harness arrangement creates a cycle, since
harness imports `build_runtime` from runtime).

The harness re-exports `parse_thought_channel` for backward compatibility
with the Step F directive, which placed it under
`uagent.experiment.harness`.
"""

from __future__ import annotations

import re

# Non-greedy match between the literal open and close tokens. The open
# token is `<|channel>thought` (pipe BEFORE `channel`); the close token
# is `<channel|>` (pipe AFTER `channel`). Asymmetry is per the Gemma 4
# spec, not a typo.
# Refs: https://ollama.com/library/gemma4
#       https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4
_THOUGHT_PATTERN = re.compile(
    r"<\|channel>thought\s*(.*?)\s*<channel\|>",
    re.DOTALL,
)


def parse_thought_channel(content: object) -> str:
    """Extract Gemma 4 thought-channel text from a model response.

    Returns the concatenated text of every well-formed thought block in
    ``content``. Multiple blocks are joined with a blank line. Any of:
    no block, an unmatched opener, an empty block, or a non-string
    input yields ``""`` — never raises. This function sits on the hot
    path of every event in a multi-hour run; a crash here is expensive.
    """
    if not isinstance(content, str) or not content:
        return ""
    matches = _THOUGHT_PATTERN.findall(content)
    # Strip per-block and drop empties so an explicit empty thought
    # `<|channel>thought\n<channel|>` collapses cleanly.
    blocks = [m.strip() for m in matches if m.strip()]
    return "\n\n".join(blocks)


def strip_thought_channel(content: object) -> str:
    """Return ``content`` with all `<|channel>thought ... <channel|>` blocks
    removed and surrounding whitespace collapsed.

    Used at sites that need the model's *final answer* text without the
    thought channel — e.g. when the dispatcher receives no tool call and
    falls back to using the natural-language content as a defer reason.
    Storing the raw content there would contaminate the audit trail with
    internal reasoning that the rest of the pipeline assumes lives only
    in ``agent_reasoning_chain``.

    Non-string input yields ``""``. Never raises.
    """
    if not isinstance(content, str) or not content:
        return ""
    stripped = _THOUGHT_PATTERN.sub("", content)
    return stripped.strip()
