"""tests/test_audio_processor_integration.py — AudioProcessor の単体テスト

Whisper API 呼び出し・get_audio_metadata()・MIME タイプ解決をモックで検証する。
"""

from io import BytesIO
from unittest.mock import MagicMock, patch
from datetime import datetime

import pytest

from src.audio_processor import AudioProcessor, _extract_recording_time


# ── フィクスチャ ────────────────────────────────────────────────────────

def _make_processor():
    """テスト用 AudioProcessor（OpenAI クライアントはモック）"""
    with patch("src.audio_processor.openai.OpenAI"):
        proc = AudioProcessor(api_key="sk-test", llm=None, whisper_prompt="")
    return proc


# ── transcribe() ───────────────────────────────────────────────────

class TestTranscribe:

    def test_transcribe_returns_text(self):
        """正常時に文字起こしテキストが返ること"""
        proc = _make_processor()

        mock_response = MagicMock()
        mock_response.text = "これは展示会での会話録音です。"
        proc._openai.audio.transcriptions.create.return_value = mock_response

        result = proc.transcribe(b"fake_audio_bytes", "test.mp3")

        assert result == "これは展示会での会話録音です。"
        proc._openai.audio.transcriptions.create.assert_called_once()

    def test_transcribe_raises_value_error_over_25mb(self):
        """25MB 超のファイルは ValueError が発生すること"""
        proc = _make_processor()
        # 26MB のダミーバイト列
        large_bytes = b"x" * (26 * 1024 * 1024)

        with pytest.raises(ValueError, match="25MB"):
            proc.transcribe(large_bytes, "large.mp3")

    def test_transcribe_raises_runtime_error_on_api_failure(self):
        """API 呼び出しが例外を投げたとき RuntimeError に変換されること"""
        proc = _make_processor()
        proc._openai.audio.transcriptions.create.side_effect = Exception("APIタイムアウト")

        with pytest.raises(RuntimeError, match="文字起こしに失敗しました"):
            proc.transcribe(b"audio", "test.m4a")

    def test_transcribe_passes_whisper_prompt_when_set(self):
        """whisper_prompt が設定されているとき API に渡されること"""
        with patch("src.audio_processor.openai.OpenAI"):
            proc = AudioProcessor(api_key="sk-test", llm=None, whisper_prompt="Sorani, EdgeGuard")

        mock_response = MagicMock()
        mock_response.text = "音声テキスト"
        proc._openai.audio.transcriptions.create.return_value = mock_response

        proc.transcribe(b"audio", "test.mp3")

        call_kwargs = proc._openai.audio.transcriptions.create.call_args[1]
        assert call_kwargs.get("prompt") == "Sorani, EdgeGuard"

    def test_transcribe_omits_prompt_when_empty(self):
        """whisper_prompt が空のとき prompt キーが渡されないこと"""
        proc = _make_processor()

        mock_response = MagicMock()
        mock_response.text = "音声テキスト"
        proc._openai.audio.transcriptions.create.return_value = mock_response

        proc.transcribe(b"audio", "test.mp3")

        call_kwargs = proc._openai.audio.transcriptions.create.call_args[1]
        assert "prompt" not in call_kwargs

    @pytest.mark.parametrize("filename,expected_mime", [
        ("rec.mp3",  "audio/mpeg"),
        ("rec.m4a",  "audio/mp4"),
        ("rec.wav",  "audio/wav"),
        ("rec.webm", "audio/webm"),
        ("rec.ogg",  "audio/ogg"),
        ("rec.flac", "audio/flac"),
        ("rec.unknown", "audio/mpeg"),  # 未知の拡張子 → デフォルト
    ])
    def test_transcribe_sets_correct_mime_type(self, filename, expected_mime):
        """拡張子に応じた MIME タイプが API に渡されること"""
        proc = _make_processor()

        mock_response = MagicMock()
        mock_response.text = "テスト"
        proc._openai.audio.transcriptions.create.return_value = mock_response

        proc.transcribe(b"audio", filename)

        call_kwargs = proc._openai.audio.transcriptions.create.call_args[1]
        file_arg = call_kwargs["file"]  # (filename, bio, content_type) のタプル
        assert file_arg[2] == expected_mime


# ── get_audio_metadata() ──────────────────────────────────────────

class TestGetAudioMetadata:

    def test_returns_size_mb(self):
        """size_mb がファイルサイズから正しく計算されること"""
        proc = _make_processor()

        mock_audio = MagicMock()
        mock_audio.info.length = 120.0
        mock_audio.tags = None

        import sys
        fake_mutagen = MagicMock()
        fake_mutagen.File = MagicMock(return_value=mock_audio)
        with patch.dict(sys.modules, {"mutagen": fake_mutagen}):
            result = proc.get_audio_metadata(b"x" * 1024, "test.mp3")

        assert result["size_mb"] == pytest.approx(1024 / (1024 * 1024), rel=1e-3)

    def test_returns_duration_from_mutagen(self):
        """mutagen から duration_sec が取得されること"""
        import sys
        proc = _make_processor()

        mock_audio = MagicMock()
        mock_audio.info.length = 300.5
        mock_audio.tags = None

        fake_mutagen = MagicMock()
        fake_mutagen.File = MagicMock(return_value=mock_audio)

        with patch.dict(sys.modules, {"mutagen": fake_mutagen}):
            result = proc.get_audio_metadata(b"audio_data", "test.mp3")

        assert result["duration_sec"] == 300.5

    def test_returns_zero_duration_when_mutagen_unavailable(self):
        """mutagen がない場合 duration_sec=0.0 になること"""
        proc = _make_processor()

        with patch.dict("sys.modules", {"mutagen": None}):
            # ImportError を発生させるために mutagen をモック削除
            import builtins
            real_import = builtins.__import__

            def import_mock(name, *args, **kwargs):
                if name == "mutagen":
                    raise ImportError("mutagen not installed")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=import_mock):
                result = proc.get_audio_metadata(b"audio_data", "test.mp3")

        assert result["duration_sec"] == 0.0
        assert result["start_time"] is None

    def test_returns_none_start_time_when_no_tags(self):
        """タグがない場合 start_time=None になること"""
        import sys
        proc = _make_processor()

        mock_audio = MagicMock()
        mock_audio.info.length = 60.0
        mock_audio.tags = None

        fake_mutagen = MagicMock()
        fake_mutagen.File = MagicMock(return_value=mock_audio)

        with patch.dict(sys.modules, {"mutagen": fake_mutagen}):
            result = proc.get_audio_metadata(b"audio", "test.m4a")

        assert result["start_time"] is None


# ── _extract_recording_time() ─────────────────────────────────────

class TestExtractRecordingTime:

    def test_returns_none_when_audio_is_none(self):
        result = _extract_recording_time(None)
        assert result is None

    def test_returns_none_when_tags_is_none(self):
        mock_audio = MagicMock()
        mock_audio.tags = None
        result = _extract_recording_time(mock_audio)
        assert result is None

    def test_extracts_from_tdrc_tag(self):
        """TDRC タグから日時が抽出されること"""
        mock_audio = MagicMock()
        tdrc = MagicMock()
        tdrc.text = ["2026-04-24 10:30:00"]
        mock_audio.tags = {"TDRC": tdrc}

        result = _extract_recording_time(mock_audio)

        assert result is not None
        assert result.year == 2026
        assert result.month == 4
        assert result.day == 24

    def test_extracts_from_day_tag_m4a(self):
        """©day タグ（M4A/iPhone）から日時が抽出されること"""
        mock_audio = MagicMock()
        mock_audio.tags = {"©day": ["2026-04-24T10:30:00"]}

        result = _extract_recording_time(mock_audio)

        assert result is not None
        assert result.year == 2026

    def test_returns_none_when_no_known_tags(self):
        """既知タグが何もない場合 None が返ること"""
        mock_audio = MagicMock()
        mock_audio.tags = {}

        result = _extract_recording_time(mock_audio)

        assert result is None


# ── estimate_cost() ──────────────────────────────────────────────

class TestEstimateCost:

    def test_cost_calculation(self):
        """コスト計算が正しいこと（$0.006/分）"""
        cost = AudioProcessor.estimate_cost(duration_sec=60.0)
        assert cost == pytest.approx(0.006, rel=1e-6)

    def test_cost_for_5_minutes(self):
        """5分の音声コストが $0.03 になること"""
        cost = AudioProcessor.estimate_cost(duration_sec=300.0)
        assert cost == pytest.approx(0.03, rel=1e-6)

    def test_zero_duration_returns_zero(self):
        cost = AudioProcessor.estimate_cost(duration_sec=0.0)
        assert cost == 0.0
