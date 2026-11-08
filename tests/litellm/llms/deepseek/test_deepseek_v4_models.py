"""Tests for DeepSeek V4/V4.1 thinking and streaming support.

The dual-mode V4/V4.1 models expose thinking as an on/off flag and stream their
reasoning through the OpenAI-compatible `delta.reasoning_content` field. These
tests pin the model-map metadata (streaming/reasoning/tool-choice flags) and
the shared reasoning-extraction path so that free-tier DeepSeek models like
`deepseek-v4.1-flash` are verified to work for thinking + streaming.
"""

import json
from pathlib import Path

import pytest

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    _extract_reasoning_content,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_PRICES = REPO_ROOT / "model_prices_and_context_window.json"

V4_MODELS = [
    "deepseek/deepseek-v4-pro",
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v4-flash-vision-exp",
    "deepseek/deepseek-v4.1-flash",
]


@pytest.fixture(scope="module")
def model_map() -> dict:
    with MODEL_PRICES.open(encoding="utf-8") as f:
        return json.load(f)


class TestV4ModelMap:
    def test_all_v4_models_declare_streaming_and_tool_choice(self, model_map):
        for name in V4_MODELS:
            entry = model_map[name]
            assert entry.get("supports_streaming") is True, f"{name} missing streaming"
            assert (
                entry.get("supports_reasoning") is True
            ), f"{name} missing reasoning"
            assert (
                entry.get("supports_parallel_function_calling") is True
            ), f"{name} missing parallel function calling"
            assert entry.get("supports_function_calling") is True
            assert entry.get("supports_tool_choice") is True

    def test_vision_exp_model_declares_vision(self, model_map):
        assert (
            model_map["deepseek/deepseek-v4-flash-vision-exp"].get(
                "supports_vision"
            )
            is True
        )

    def test_v4_1_flash_has_a_large_context_window(self, model_map):
        entry = model_map["deepseek/deepseek-v4.1-flash"]
        assert entry["max_input_tokens"] >= 1_000_000


class TestReasoningContentExtraction:
    def test_extracts_dual_mode_reasoning_field(self):
        # DeepSeek V4 streams reasoning in `reasoning_content`, separate from
        # the visible content, in both streaming deltas and final messages.
        message = {"content": "final answer", "reasoning_content": "thinking…"}
        reasoning, content = _extract_reasoning_content(message)
        assert reasoning == "thinking…"
        assert content == "final answer"

    def test_falls_back_to_the_aliased_reasoning_field(self):
        message = {"content": "final answer", "reasoning": "thinking…"}
        reasoning, content = _extract_reasoning_content(message)
        assert reasoning == "thinking…"
        assert content == "final answer"
