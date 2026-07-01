"""tests/test_pdf_processor.py — pdf_processor の単体テスト

fitz (PyMuPDF) と OpenAI Vision API をモックして
extract_text_from_pdf_vlm() の正常系・異常系を検証する。
"""

import sys
from unittest.mock import MagicMock, patch, call

import pytest

from src.pdf_processor import is_pdf


# ── is_pdf() ────────────────────────────────────────────────────────

class TestIsPdf:

    @pytest.mark.parametrize("path", [
        "document.pdf",
        "PATH/TO/FILE.PDF",
        "report.PDF",
        "/absolute/path/data.pdf",
    ])
    def test_returns_true_for_pdf_extension(self, path):
        assert is_pdf(path) is True

    @pytest.mark.parametrize("path", [
        "document.md",
        "document.txt",
        "document.docx",
        "document",
        "",
        "notapdf",
    ])
    def test_returns_false_for_non_pdf(self, path):
        assert is_pdf(path) is False


# ── extract_text_from_pdf_vlm() ─────────────────────────────────────

class TestExtractTextFromPdfVlm:

    def _make_mock_fitz(self, page_count: int = 2):
        """ダミーの fitz モジュールを返す"""
        fake_fitz = MagicMock()

        # ページのモック
        fake_page = MagicMock()
        fake_pix = MagicMock()
        fake_pix.tobytes.return_value = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # PNG ダミーバイト
        fake_page.get_pixmap.return_value = fake_pix

        # ドキュメントのモック
        fake_doc = MagicMock()
        fake_doc.__len__ = MagicMock(return_value=page_count)
        fake_doc.__getitem__ = MagicMock(return_value=fake_page)

        fake_fitz.open.return_value = fake_doc
        fake_fitz.Matrix = MagicMock(return_value=MagicMock())

        return fake_fitz, fake_doc, fake_page

    def _make_mock_client(self, page_text: str = "ページの内容"):
        """ダミーの OpenAI クライアントを返す"""
        fake_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = page_text
        fake_client.chat.completions.create.return_value = mock_response
        return fake_client

    def test_raises_import_error_when_fitz_unavailable(self):
        """fitz がインストールされていない場合 ImportError が発生すること"""
        import sys
        from src.pdf_processor import extract_text_from_pdf_vlm

        # sys.modules["fitz"] を None にすると import fitz が ImportError を起こす
        with patch.dict(sys.modules, {"fitz": None}):
            with pytest.raises(ImportError, match="PyMuPDF"):
                extract_text_from_pdf_vlm("test.pdf", MagicMock(), "gpt-test")

    def test_processes_each_page(self):
        """ページ数分だけ Vision API が呼ばれること"""
        from src.pdf_processor import extract_text_from_pdf_vlm

        fake_fitz, fake_doc, _ = self._make_mock_fitz(page_count=3)
        fake_client = self._make_mock_client("ページテキスト")

        with patch.dict(sys.modules, {"fitz": fake_fitz}):
            result = extract_text_from_pdf_vlm("test.pdf", fake_client, "gpt-test")

        assert fake_client.chat.completions.create.call_count == 3

    def test_joins_pages_with_separator(self):
        """複数ページの結果が --- で結合されること"""
        from src.pdf_processor import extract_text_from_pdf_vlm

        fake_fitz, fake_doc, _ = self._make_mock_fitz(page_count=2)

        call_count = [0]
        def create_side_effect(**kwargs):
            call_count[0] += 1
            mock_resp = MagicMock()
            mock_resp.choices[0].message.content = f"ページ{call_count[0]}の内容"
            return mock_resp

        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = create_side_effect

        with patch.dict(sys.modules, {"fitz": fake_fitz}):
            result = extract_text_from_pdf_vlm("test.pdf", fake_client, "gpt-test")

        assert "---" in result
        assert "ページ1の内容" in result
        assert "ページ2の内容" in result

    def test_skips_failed_page_and_continues(self):
        """特定ページで例外が発生しても残りのページが処理されること"""
        from src.pdf_processor import extract_text_from_pdf_vlm

        fake_fitz, fake_doc, _ = self._make_mock_fitz(page_count=3)

        call_count = [0]
        def create_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("ページ2の処理に失敗")
            mock_resp = MagicMock()
            mock_resp.choices[0].message.content = f"ページ{call_count[0]}の内容"
            return mock_resp

        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = create_side_effect

        with patch.dict(sys.modules, {"fitz": fake_fitz}):
            result = extract_text_from_pdf_vlm("test.pdf", fake_client, "gpt-test")

        # 成功したページ1と3が含まれること
        assert "ページ1の内容" in result
        assert "ページ3の内容" in result

    def test_closes_document_after_processing(self):
        """処理後にドキュメントが close されること"""
        from src.pdf_processor import extract_text_from_pdf_vlm

        fake_fitz, fake_doc, _ = self._make_mock_fitz(page_count=1)
        fake_client = self._make_mock_client()

        with patch.dict(sys.modules, {"fitz": fake_fitz}):
            extract_text_from_pdf_vlm("test.pdf", fake_client, "gpt-test")

        fake_doc.close.assert_called_once()

    def test_warning_printed_for_large_pdf(self, capsys):
        """10ページ超の PDF では警告が表示されること"""
        from src.pdf_processor import extract_text_from_pdf_vlm

        fake_fitz, fake_doc, _ = self._make_mock_fitz(page_count=15)
        fake_client = self._make_mock_client()

        with patch.dict(sys.modules, {"fitz": fake_fitz}):
            extract_text_from_pdf_vlm("large.pdf", fake_client, "gpt-test")

        captured = capsys.readouterr()
        assert "警告" in captured.out or "15" in captured.out

    def test_returns_empty_string_when_all_pages_fail(self):
        """全ページで例外が発生したとき空文字が返ること"""
        from src.pdf_processor import extract_text_from_pdf_vlm

        fake_fitz, fake_doc, _ = self._make_mock_fitz(page_count=2)

        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = RuntimeError("全ページ失敗")

        with patch.dict(sys.modules, {"fitz": fake_fitz}):
            result = extract_text_from_pdf_vlm("test.pdf", fake_client, "gpt-test")

        assert result == ""
