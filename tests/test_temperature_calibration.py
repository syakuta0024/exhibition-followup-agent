"""tests/test_temperature_calibration.py — temperature 判定基準の精緻化テスト

LLM 呼び出しはすべてモック（API コスト発生なし）。
判定ロジックは _NEEDS_EXTRACTION_SYSTEM プロンプト内にあるため:
  - プロンプトに判定基準テキストが含まれているか（構造テスト）
  - extract_needs() が LLM の返した temperature を正しく抽出するか（動作テスト）
の2軸で検証する。
"""

import json
from unittest.mock import MagicMock

import pytest

from src.audio_processor import AudioProcessor, _NEEDS_EXTRACTION_SYSTEM


# ── ヘルパー ─────────────────────────────────────────────────────────────

def _make_processor(temperature: str) -> AudioProcessor:
    """指定した temperature を返す LLM モック付き AudioProcessor"""
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "summary": "テスト要約",
        "issues": "テスト課題",
        "needs": "テストニーズ",
        "budget": "",
        "decision_maker": "",
        "temperature": temperature,
    })
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    proc = AudioProcessor(api_key="sk-test", llm=mock_llm)
    return proc


# ── 構造テスト: プロンプトに判定基準が含まれること ──────────────────────────

class TestSystemPromptStructure:

    def test_system_prompt_has_high_criteria(self):
        """_NEEDS_EXTRACTION_SYSTEM に high 判定基準と '2つ以上' ルールが含まれること"""
        assert "high" in _NEEDS_EXTRACTION_SYSTEM
        assert "2つ以上" in _NEEDS_EXTRACTION_SYSTEM

    def test_system_prompt_has_warm_and_cold(self):
        """warm / cold の基準も含まれること"""
        assert "warm" in _NEEDS_EXTRACTION_SYSTEM
        assert "cold" in _NEEDS_EXTRACTION_SYSTEM

    def test_system_prompt_sends_criteria_to_llm(self):
        """extract_needs() 呼び出し時に判定基準がシステムメッセージとして渡されること"""
        proc = _make_processor("high")
        proc.extract_needs("デモ希望。予算400万。自分で決められる。")

        messages = proc._llm.invoke.call_args[0][0]
        system_content = messages[0].content  # SystemMessage
        assert "high" in system_content
        assert "2つ以上" in system_content


# ── 動作テスト: 各シナリオで temperature が正しく抽出されること ───────────────

class TestTemperatureExtraction:

    def test_high_all_three_conditions(self):
        """3条件すべて満たす（デモ希望・予算400万・自己承認権限あり）→ high"""
        proc = _make_processor("high")
        result = proc.extract_needs("デモ希望。予算400万確定。自分で決められる。")
        assert result["temperature"] == "high"

    def test_cold_information_gathering(self):
        """情報収集段階・予算未定・上申必要 → cold"""
        proc = _make_processor("cold")
        result = proc.extract_needs("まだ情報収集中。予算未定。上申必要。")
        assert result["temperature"] == "cold"

    def test_warm_interested_but_uncertain(self):
        """興味あり・検討中・予算不明 → warm"""
        proc = _make_processor("warm")
        result = proc.extract_needs("興味あり。検討中。予算不明。")
        assert result["temperature"] == "warm"

    def test_high_two_conditions_budget_and_demo(self):
        """予算確定＋デモ希望（2つ満たす）→ high"""
        proc = _make_processor("high")
        result = proc.extract_needs("デモ希望。予算400万確定。承認プロセス不明。")
        assert result["temperature"] == "high"

    def test_warm_one_condition_only(self):
        """デモ希望のみ（1条件）→ warm"""
        proc = _make_processor("warm")
        result = proc.extract_needs("デモ希望。予算未定。承認プロセス不明。")
        assert result["temperature"] == "warm"
