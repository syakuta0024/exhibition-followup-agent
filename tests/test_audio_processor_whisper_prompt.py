"""tests/test_audio_processor_whisper_prompt.py — Whisper prompt パラメータの単体テスト"""

from unittest.mock import MagicMock, patch

import pytest

from src.audio_processor import AudioProcessor, resolve_whisper_prompt


# ── resolve_whisper_prompt ────────────────────────────────────────────────

class TestResolveWhisperPrompt:

    def test_user_prompt_takes_priority(self):
        """ユーザー設定 whisper_prompt が product_urls より優先されること"""
        result = resolve_whisper_prompt("プレス機, 異常検知", {"EdgeGuard": "", "DigiMA": ""})
        assert result == "プレス機, 異常検知"

    def test_auto_generate_from_product_urls(self):
        """whisper_prompt が空のとき product_urls のキー名から自動生成されること"""
        result = resolve_whisper_prompt("", {"EdgeGuard": "", "DigiMA": "", "Sorani": ""})
        assert "EdgeGuard" in result
        assert "DigiMA" in result
        assert "Sorani" in result

    def test_empty_when_both_empty(self):
        """whisper_prompt も product_urls も空のとき空文字が返ること"""
        assert resolve_whisper_prompt("", {}) == ""


# ── AudioProcessor.transcribe() の prompt 渡し ───────────────────────────

def _make_processor(whisper_prompt: str = "") -> AudioProcessor:
    """テスト用 AudioProcessor（OpenAI クライアントをモック）"""
    proc = AudioProcessor(api_key="sk-test", whisper_prompt=whisper_prompt)
    mock_response = MagicMock()
    mock_response.text = "テスト文字起こし"
    proc._openai = MagicMock()
    proc._openai.audio.transcriptions.create.return_value = mock_response
    return proc


class TestTranscribeWhisperPrompt:

    def test_prompt_passed_to_api_when_set(self):
        """whisper_prompt が設定済みのとき API kwargs に prompt が含まれること"""
        proc = _make_processor("EdgeGuard, DigiMA")
        proc.transcribe(b"\x00" * 10, "test.mp3")

        kwargs = proc._openai.audio.transcriptions.create.call_args.kwargs
        assert kwargs.get("prompt") == "EdgeGuard, DigiMA"

    def test_prompt_not_in_api_kwargs_when_empty(self):
        """whisper_prompt が空のとき prompt キーが API に渡らないこと"""
        proc = _make_processor("")
        proc.transcribe(b"\x00" * 10, "test.mp3")

        kwargs = proc._openai.audio.transcriptions.create.call_args.kwargs
        assert "prompt" not in kwargs

    def test_no_crash_without_whisper_prompt(self):
        """デフォルト引数（whisper_prompt 未指定）でクラッシュしないこと"""
        proc = AudioProcessor(api_key="sk-test")
        mock_response = MagicMock()
        mock_response.text = "ok"
        proc._openai = MagicMock()
        proc._openai.audio.transcriptions.create.return_value = mock_response

        result = proc.transcribe(b"\x00" * 10, "test.mp3")
        assert result == "ok"
