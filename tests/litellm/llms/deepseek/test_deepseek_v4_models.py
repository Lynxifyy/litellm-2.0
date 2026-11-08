"""Tests for the FreeHands DeepSeek V4/V4.1 model-map additions.

Upstream litellm has `deepseek-v4-pro` and `deepseek-v4-flash` but omits the
vision-preview variant and the newer `deepseek-v4.1-flash`. This fork adds both
so FreeHands can use them. The tests pin the model-map entries (metadata only —
costs are inherited from the sibling `deepseek-v4-flash` entry).
"""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_PRICES = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_MODEL_PRICES = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

FORK_ADDED_MODELS = [
    "deepseek/deepseek-v4-flash-vision-exp",
    "deepseek/deepseek-v4.1-flash",
]


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def model_map() -> dict:
    return _load(MODEL_PRICES)




@pytest.mark.parametrize("path", [MODEL_PRICES, BACKUP_MODEL_PRICES])
class TestForkAddedV4Models:
    def test_models_are_present(self, path):
        model_map = _load(path)
        for name in FORK_ADDED_MODELS:
            assert name in model_map, f"{name} missing from {path.name}"

    def test_models_declare_streaming_and_function_calling(self, path):
        model_map = _load(path)
        for name in FORK_ADDED_MODELS:
            entry = model_map[name]
            assert entry.get("supports_native_streaming") is True
            assert entry.get("supports_function_calling") is True
            assert entry.get("supports_parallel_function_calling") is True
            assert entry.get("supports_tool_choice") is True

    def test_vision_exp_declares_vision(self, path):
        model_map = _load(path)
        assert model_map["deepseek/deepseek-v4-flash-vision-exp"]["supports_vision"] is True

    def test_v4_1_flash_has_large_context(self, path):
        model_map = _load(path)
        assert model_map["deepseek/deepseek-v4.1-flash"]["max_input_tokens"] >= 1_000_000


class TestV4BaseModels:
    def test_existing_v4_models_keep_streaming_support(self, model_map):
        for name in ["deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-flash"]:
            assert model_map[name].get("supports_native_streaming") is True
