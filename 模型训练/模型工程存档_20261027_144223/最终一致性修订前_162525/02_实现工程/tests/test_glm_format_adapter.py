import json
from pathlib import Path

import pytest
from jinja2 import Template
from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser

from audit_engine.training.glm_format_adapter import (
    ANSWER_SCHEMA,
    SYSTEM_PROMPT,
    adapted_chat_template,
    adapted_messages,
)


def consume(text):
    parser = JsonSchemaParser(
        ANSWER_SCHEMA, CharacterLevelParserConfig(force_json_field_order=True)
    )
    for char in text:
        assert char in parser.get_allowed_characters()
        parser = parser.add_character(char)
    return parser


@pytest.mark.parametrize("flag, amount", [(0, "0"), (0, "29.55"), (1, "12800.00")])
def test_distinct_decisions_and_amounts_are_allowed(flag, amount):
    text = json.dumps(
        {"FLAG": flag, "max_legal": amount, "problem_reports": ["A concise issue."]},
        separators=(",", ":"),
    )
    assert consume(text).can_end()


@pytest.mark.parametrize(
    "text",
    [
        '{"FLAG":true',
        '{"FLAG":2',
        '{"FLAG":0,"max_legal":12',
        '{"FLAG":0,"max_legal":"NaN',
        "A valid expense",
    ],
)
def test_invalid_types_and_unstructured_text_are_disallowed(text):
    with pytest.raises(AssertionError):
        consume(text)


def test_source_materials_are_preserved_and_not_mutated():
    original = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Original policy"},
                {"type": "image", "image": "/image.png"},
            ],
        }
    ]
    before = json.dumps(original)
    result = adapted_messages(original)
    assert json.dumps(original) == before
    assert result[1]["content"][:-1] == original[0]["content"]
    assert result[0]["role"] == "system"


def test_native_system_template_renders_multimodal_text_content():
    path = (
        Path(__file__).resolve().parents[1]
        / "models/GLM-4.1V-9B-Thinking/chat_template.jinja"
    )
    if not path.exists():
        pytest.skip("local GLM tokenizer is not installed")
    messages = adapted_messages(
        [{"role": "user", "content": [{"type": "text", "text": "Original source"}]}]
    )
    rendered = Template(adapted_chat_template(path.read_text())).render(
        messages=messages, add_generation_prompt=True
    )
    assert SYSTEM_PROMPT in rendered and "Original source" in rendered
    assert "[{'type'" not in rendered.split("<|user|>")[0]
