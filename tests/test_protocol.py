"""Tests for the A2A message builders (wire-level invariants).

The key invariant these guard: every atrium-built message carries a ``message_id``.
A2A (v0.3) requires it, and a compliant server rejects a message without one with
``INVALID_PARAMS`` — so a missing id silently breaks real cross-agent dispatch.
"""

from __future__ import annotations

from atrium.protocol import data_message, get_message_text, text_message


def test_text_message_sets_a_message_id():
    msg = text_message("hello")
    assert msg.message_id  # non-empty
    assert get_message_text(msg) == "hello"


def test_data_message_sets_a_message_id():
    msg = data_message({"k": "v"})
    assert msg.message_id


def test_message_ids_are_unique_per_build():
    a = text_message("x")
    b = text_message("x")
    assert a.message_id != b.message_id
