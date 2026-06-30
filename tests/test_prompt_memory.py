"""Unit tests for the layered prompt-construction feature.

These are pure / GPU-free: they exercise ``PromptLayer`` / ``PromptMemory``
composition and the ``InferenceAgent.build_system_prompt`` seam without starting
a sandbox or contacting a model.
"""

from __future__ import annotations

import pytest

from atrium.agents.prompt_memory import (
    PromptLayer,
    PromptMemory,
    default_prompt_memory,
    render_tools_block,
    tools_layer,
)

SAMPLE_TOOL = {
    "type": "function",
    "function": {"name": "read_file", "description": "Read a file", "parameters": {}},
}


# --------------------------------------------------------------------------- #
# PromptLayer.body resolution                                                 #
# --------------------------------------------------------------------------- #
def test_body_precedence_render_over_template_over_content():
    layer = PromptLayer(
        "x",
        content="C",
        template="T-{v}",
        render=lambda ctx: f"R-{ctx['v']}",
    )
    assert layer.body({"v": 1}) == "R-1"

    layer.render = None
    assert layer.body({"v": 2}) == "T-2"

    layer.template = None
    assert layer.body({"v": 3}) == "C"


# --------------------------------------------------------------------------- #
# record / remove                                                             #
# --------------------------------------------------------------------------- #
def test_record_adds_and_replaces_by_name():
    mem = PromptMemory()
    mem.record(PromptLayer("a", content="first"))
    mem.record(PromptLayer("a", content="second"))  # same name -> replace
    assert len(mem.layers) == 1
    assert mem.layers["a"].content == "second"


def test_remove_drops_layer():
    mem = PromptMemory().record(PromptLayer("a")).record(PromptLayer("b"))
    mem.remove("a")
    assert set(mem.layers) == {"b"}
    mem.remove("missing")  # no-op


# --------------------------------------------------------------------------- #
# ordering                                                                    #
# --------------------------------------------------------------------------- #
def test_explicit_order_names_come_first_then_numeric():
    mem = PromptMemory(order=("b", "a"))
    mem.record(PromptLayer("a", content="A", order=10))
    mem.record(PromptLayer("b", content="B", order=99))
    mem.record(PromptLayer("c", content="C", order=5))  # not pinned -> by order
    assert mem.compose() == "B\n\nA\n\nC"


def test_numeric_order_with_insertion_order_tiebreak():
    mem = PromptMemory()
    mem.record(PromptLayer("a", content="A", order=50))
    mem.record(PromptLayer("b", content="B", order=50))  # tie -> insertion order
    mem.record(PromptLayer("c", content="C", order=10))
    assert mem.compose() == "C\n\nA\n\nB"


# --------------------------------------------------------------------------- #
# compose: skip rules, include/exclude, title, sep                            #
# --------------------------------------------------------------------------- #
def test_compose_skips_disabled_and_empty():
    mem = PromptMemory()
    mem.record(PromptLayer("a", content="A", order=1))
    mem.record(PromptLayer("off", content="X", order=2, enabled=False))
    mem.record(PromptLayer("blank", content="   ", order=3))  # whitespace -> empty
    mem.record(PromptLayer("b", content="B", order=4))
    assert mem.compose() == "A\n\nB"


def test_compose_include_and_exclude():
    mem = PromptMemory()
    mem.record(PromptLayer("a", content="A", order=1))
    mem.record(PromptLayer("b", content="B", order=2))
    mem.record(PromptLayer("c", content="C", order=3))
    assert mem.compose(include={"a", "c"}) == "A\n\nC"
    assert mem.compose(exclude={"b"}) == "A\n\nC"


def test_compose_title_and_custom_sep():
    mem = PromptMemory(sep="\n---\n")
    mem.record(PromptLayer("a", content="body", order=1, title="## Heading"))
    assert mem.compose() == "## Heading\nbody"
    mem.record(PromptLayer("b", content="more", order=2))
    assert mem.compose() == "## Heading\nbody\n---\nmore"


# --------------------------------------------------------------------------- #
# tools layer                                                                 #
# --------------------------------------------------------------------------- #
def test_render_tools_block_empty_and_populated():
    assert render_tools_block(None) == ""
    assert render_tools_block([]) == ""
    block = render_tools_block([SAMPLE_TOOL])
    assert block.startswith("<tools>") and block.endswith("</tools>")
    assert '"read_file"' in block


def test_tools_layer_json_schema_renders_schemas():
    layer = tools_layer(mode="json_schema", guidance="Use tools wisely.")
    body = layer.body({"tools": [SAMPLE_TOOL]})
    assert "Use tools wisely." in body
    assert "<tools>" in body and '"read_file"' in body
    # No tools -> only guidance survives.
    assert layer.body({"tools": []}) == "Use tools wisely."


def test_tools_layer_native_emits_guidance_only():
    layer = tools_layer(mode="native", guidance="Call one tool at a time.")
    body = layer.body({"tools": [SAMPLE_TOOL]})
    assert body == "Call one tool at a time."
    assert "<tools>" not in body


def test_tools_layer_rejects_bad_mode():
    with pytest.raises(ValueError):
        tools_layer(mode="bogus")


def test_tools_layer_in_memory_at_configured_position():
    mem = default_prompt_memory()
    mem.record(PromptLayer("identity", order=10, content="ID"))
    mem.record(PromptLayer("rules", order=60, content="RULES"))
    out = mem.compose({"tools": [SAMPLE_TOOL]})
    # identity (10) precedes the tools block (40) precedes rules (60).
    assert out.index("ID") < out.index("<tools>") < out.index("RULES")
    # empty default layers (tone, memory, ...) are absent.
    assert "tone" not in out


# --------------------------------------------------------------------------- #
# from_mapping                                                                 #
# --------------------------------------------------------------------------- #
def test_from_mapping_builds_layers_and_tools():
    mem = PromptMemory.from_mapping(
        {
            "order": ["identity", "tools"],
            "layers": {
                "identity": {"order": 10, "content": "You are X."},
                "tools": {"order": 40, "mode": "json_schema"},
            },
        }
    )
    assert mem.order == ("identity", "tools")
    assert mem.layers["tools"].render is not None  # tools_layer built
    out = mem.compose({"tools": [SAMPLE_TOOL]})
    assert out.index("You are X.") < out.index("<tools>")


def test_from_mapping_empty_returns_empty_memory():
    assert PromptMemory.from_mapping(None).layers == {}
    assert PromptMemory.from_mapping({}).layers == {}


def test_from_mapping_rejects_unknown_top_level_key():
    with pytest.raises(ValueError):
        PromptMemory.from_mapping({"orderr": []})


def test_from_mapping_rejects_unknown_layer_key():
    with pytest.raises(ValueError):
        PromptMemory.from_mapping({"layers": {"a": {"contentt": "x"}}})


# --------------------------------------------------------------------------- #
# default_prompt_memory                                                       #
# --------------------------------------------------------------------------- #
def test_default_prompt_memory_is_noop_until_filled():
    mem = default_prompt_memory()
    assert mem.compose() == ""  # all layers empty -> nothing
    mem.record(PromptLayer("identity", order=10, content="Hello."))
    assert mem.compose() == "Hello."
