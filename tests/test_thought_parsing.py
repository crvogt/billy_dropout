"""Parser tests for Gemma 4 thought-channel extraction.

`parse_thought_channel` lives in `uagent.agent.parsing` and is
re-exported from `uagent.experiment.harness` for backward compatibility
with the Step F directive. It is called from
`uagent.agent.runtime.agent_reason` to pull the
`<|channel>thought ... <channel|>` block out of a model response before
storing the result as `agent_reasoning_chain`.

These tests pin behavior on the edge cases enumerated in the Step F
directive plus a handful of paranoid defensive cases (the parser sits
on the hot path of every event in the experiment, so a crash here
takes down a 7-hour run).

Tag convention in this file:
    <|channel>thought      open token (pipe BEFORE `channel`)
    <channel|>             close token (pipe AFTER  `channel`)
"""

from __future__ import annotations

from uagent.agent.parsing import parse_thought_channel, strip_thought_channel
from uagent.experiment.harness import (
    parse_thought_channel as parse_thought_channel_via_harness,
)


# ---------------------------------------------------------------------------
# Directive-mandated cases (5)
# ---------------------------------------------------------------------------


def test_well_formed_single_block():
    """Standard happy-path response: one thought block, post-channel text."""
    content = (
        "<|channel>thought\n"
        "Detection confidence is 0.91; commit forward.\n"
        "<channel|>\n"
        "Calling move_forward(0.3)."
    )
    assert parse_thought_channel(content) == (
        "Detection confidence is 0.91; commit forward."
    )


def test_no_channel_block_yields_empty():
    """Text-only response with no thinking tokens at all returns ''."""
    content = "Plain natural-language reply, no thinking tokens."
    assert parse_thought_channel(content) == ""


def test_truncated_open_without_closer_yields_empty():
    """Open token but no `<channel|>` closer → defensive empty string.

    Don't return a half-parsed thought even if a long thought-looking
    blob was emitted; an unterminated block is ambiguous.
    """
    content = (
        "<|channel>thought\n"
        "model started thinking but the stream got cut here..."
    )
    assert parse_thought_channel(content) == ""


def test_multiple_blocks_are_concatenated():
    """Two complete blocks → concatenated with a blank-line separator.

    Defensive over taking-the-first: if the model emits more than one
    thought block (re-thinking after a tool result, for example) we
    preserve all of it so the reasoning audit is faithful.
    """
    content = (
        "<|channel>thought\nfirst thought\n<channel|>\n"
        "intermediate filler\n"
        "<|channel>thought\nsecond thought\n<channel|>\n"
        "final"
    )
    assert parse_thought_channel(content) == "first thought\n\nsecond thought"


def test_empty_thought_block_yields_empty():
    """`<|channel>thought\\n<channel|>` (no inner text) → empty."""
    content = "<|channel>thought\n<channel|>\nfinal answer"
    assert parse_thought_channel(content) == ""


# ---------------------------------------------------------------------------
# Paranoid defensive cases
# ---------------------------------------------------------------------------


def test_empty_string_returns_empty():
    assert parse_thought_channel("") == ""


def test_non_string_returns_empty():
    """Caller passing None / list / int should not crash the run.

    LangChain occasionally returns content as a list of typed blocks
    rather than a flat string; the runtime str()-stringifies before
    calling us, but belt-and-suspenders this layer too.
    """
    assert parse_thought_channel(None) == ""        # type: ignore[arg-type]
    assert parse_thought_channel(["foo", "bar"]) == ""  # type: ignore[arg-type]
    assert parse_thought_channel(42) == ""          # type: ignore[arg-type]


def test_closer_before_opener_yields_empty():
    """A stray `<channel|>` before any opener is not a block."""
    content = "stray <channel|> then text <|channel>thought no closer"
    assert parse_thought_channel(content) == ""


def test_multiline_thought_preserves_internal_newlines():
    """Internal line breaks inside the thought are preserved verbatim."""
    content = (
        "<|channel>thought\n"
        "line one\n"
        "line two\n"
        "\n"
        "line four after blank\n"
        "<channel|>\n"
        "answer"
    )
    expected = "line one\nline two\n\nline four after blank"
    assert parse_thought_channel(content) == expected


def test_surrounding_whitespace_does_not_break_match():
    """Models sometimes pad tokens with extra whitespace."""
    content = "<|channel>thought   \n  the actual thought  \n<channel|>"
    assert parse_thought_channel(content) == "the actual thought"


def test_only_whitespace_inside_block_collapses_to_empty():
    """A block containing only whitespace is treated as empty."""
    content = "<|channel>thought\n   \n\t\n<channel|>"
    assert parse_thought_channel(content) == ""


def test_open_without_thought_suffix_yields_empty():
    """`<|channel>` without the literal `thought` suffix is not a match.

    The Gemma 4 spec uses `<|channel>thought` for the thinking channel
    specifically; other channels (e.g. a hypothetical `<|channel>tool`)
    must NOT spuriously land in reasoning_chain.
    """
    content = "<|channel>\nsome text\n<channel|>"
    assert parse_thought_channel(content) == ""


def test_nested_looking_tokens_first_close_wins():
    """Non-greedy match stops at the first `<channel|>`.

    Real Gemma 4 won't nest thought blocks, but pin the deterministic
    behavior so a hallucinated re-opener inside a thought doesn't lose
    the outer content silently.
    """
    content = (
        "<|channel>thought\nouter <|channel>thought inner\n<channel|>"
        " still outer\n<channel|>"
    )
    # First match closes at the first `<channel|>`, capturing
    # "outer <|channel>thought inner".
    assert parse_thought_channel(content) == "outer <|channel>thought inner"


def test_bom_prefix_does_not_break_match():
    """A stray UTF-8 BOM before the opener should not defeat the parser."""
    content = "﻿<|channel>thought\nthinking\n<channel|>"
    assert parse_thought_channel(content) == "thinking"


def test_crlf_line_endings_supported():
    """Windows-style line endings in the response are tolerated."""
    content = "<|channel>thought\r\nline one\r\nline two\r\n<channel|>"
    assert parse_thought_channel(content) == "line one\r\nline two"


def test_very_long_input_does_not_pathologically_backtrack():
    """A multi-MB irrelevant prefix must not blow up the regex.

    re.DOTALL with `.*?` is O(n); this is a smoke test for the
    hot-path performance claim, not a benchmark.
    """
    prefix = "x" * 2_000_000
    content = prefix + "<|channel>thought\nactual\n<channel|>"
    assert parse_thought_channel(content) == "actual"


def test_thought_inside_turn_frame():
    """Tolerate a thought block embedded inside a turn wrapper.

    LangChain's ChatOllama is expected to strip turn frames before
    populating response.content, but if a turn frame ever leaks through
    we should still find the thought.
    """
    content = (
        "<|turn>assistant\n"
        "<|channel>thought\nframed thought\n<channel|>\n"
        "final answer\n"
        "<turn|>"
    )
    assert parse_thought_channel(content) == "framed thought"


def test_list_content_stringified_safely():
    """LangChain occasionally returns content as a list of typed blocks.

    The runtime str()-stringifies before calling us, but the parser
    must not crash when given the resulting `[{'type': ...}]` repr —
    it should simply find no match and return "".
    """
    # Simulate runtime's `str(response.content)` on a multimodal-block list.
    list_repr = str([{"type": "text", "text": "hello"}])
    assert parse_thought_channel(list_repr) == ""


# ---------------------------------------------------------------------------
# Harness re-export sanity
# ---------------------------------------------------------------------------


def test_harness_reexport_is_the_same_function():
    """`uagent.experiment.harness.parse_thought_channel` must be identical
    to `uagent.agent.parsing.parse_thought_channel` (re-export, not a
    diverging copy)."""
    assert parse_thought_channel_via_harness is parse_thought_channel


# ---------------------------------------------------------------------------
# strip_thought_channel — complementary stripper used in the dispatcher's
# no-tool-call fallback path to keep final_args["reason"] free of leaked
# thought content.
# ---------------------------------------------------------------------------


def test_strip_removes_single_block():
    content = (
        "<|channel>thought\nhidden\n<channel|>\n"
        "Visible final answer here."
    )
    assert strip_thought_channel(content) == "Visible final answer here."


def test_strip_no_block_returns_input_trimmed():
    """When there's no thought block, the input is returned with edge whitespace removed."""
    assert strip_thought_channel("  just an answer  ") == "just an answer"


def test_strip_removes_multiple_blocks():
    content = (
        "<|channel>thought\nfirst\n<channel|> mid "
        "<|channel>thought\nsecond\n<channel|> tail"
    )
    assert strip_thought_channel(content) == "mid  tail"


def test_strip_empty_or_non_string_returns_empty():
    assert strip_thought_channel("") == ""
    assert strip_thought_channel(None) == ""        # type: ignore[arg-type]
    assert strip_thought_channel(42) == ""          # type: ignore[arg-type]


def test_strip_only_thought_yields_empty():
    """If the entire content is a thought block, stripping leaves nothing."""
    assert strip_thought_channel("<|channel>thought\nonly\n<channel|>") == ""


# ---------------------------------------------------------------------------
# Loose `<think>...</think>` fallback — covers the empirically-observed
# response shape on gemma4 e2b/e4b under the verbal-instruction prompt.
# Strict envelope wins when both are present (forward-compat hedge).
# ---------------------------------------------------------------------------


def test_loose_think_tag_is_picked_up():
    """`<think>...</think>` extracted via the fallback path."""
    content = (
        "<think>Confidence is high; commit forward.</think>\n"
        "Calling move_forward(0.3)."
    )
    assert parse_thought_channel(content) == "Confidence is high; commit forward."


def test_loose_think_tag_case_insensitive():
    """Case variations on the literal tag still match."""
    assert parse_thought_channel("<Think>hello</Think>") == "hello"
    assert parse_thought_channel("<THINK>hello</THINK>") == "hello"


def test_loose_empty_think_tag_yields_empty():
    assert parse_thought_channel("<think></think>tail") == ""


def test_loose_multiline_think_block():
    content = "<think>\nline one\nline two\n</think>\nfinal"
    assert parse_thought_channel(content) == "line one\nline two"


def test_strict_pattern_wins_when_both_present():
    """If a response somehow contains both surface forms, the strict
    envelope is the source of truth (per the spec'd shape)."""
    content = (
        "<|channel>thought\nstrict\n<channel|>\n"
        "<think>loose</think>\nfinal"
    )
    assert parse_thought_channel(content) == "strict"


def test_loose_unclosed_think_yields_empty():
    """Loose pattern also requires a closer."""
    assert parse_thought_channel("<think>never closed") == ""


def test_strip_removes_loose_think_block():
    content = "<think>hidden</think>\nVisible answer."
    assert strip_thought_channel(content) == "Visible answer."


def test_strip_removes_both_surfaces():
    """Strict + loose blocks both stripped from the same content."""
    content = (
        "<|channel>thought\nA\n<channel|> mid "
        "<think>B</think> tail"
    )
    assert strip_thought_channel(content) == "mid  tail"
