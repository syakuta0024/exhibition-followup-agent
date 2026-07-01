"""tests/test_load_csv_encoding.py — utils.load_csv_with_encoding() の単体テスト

UTF-8 BOM / UTF-8 / Shift_JIS のエンコーディング自動判定と、
NaN補完・空白トリム・全失敗時の ValueError を検証する。
"""

import io
from unittest.mock import patch

import pandas as pd
import pytest

from src.utils import load_csv_with_encoding


class TestLoadCsvWithEncoding:

    def _make_file(self, text: str, encoding: str) -> io.BytesIO:
        return io.BytesIO(text.encode(encoding))

    def test_reads_utf8_bom_csv(self):
        """UTF-8 BOM（Excel エクスポート形式）が正しく読み込めること"""
        f = self._make_file("氏名,会社名\n山田 太郎,ABC社\n", "utf-8-sig")
        df = load_csv_with_encoding(f)
        assert len(df) == 1
        assert df.iloc[0]["氏名"] == "山田 太郎"
        assert df.iloc[0]["会社名"] == "ABC社"

    def test_reads_utf8_csv(self):
        """BOM なし UTF-8 CSV が読み込めること"""
        f = self._make_file("name,company\nAlice,Acme\n", "utf-8")
        df = load_csv_with_encoding(f)
        assert df.iloc[0]["name"] == "Alice"

    def test_reads_shift_jis_csv(self):
        """Shift_JIS CSV（日本語）が読み込めること（UTF-8 失敗後のフォールバック）"""
        f = self._make_file("氏名,会社名\n鈴木 花子,XYZ工業\n", "shift_jis")
        df = load_csv_with_encoding(f)
        assert df.iloc[0]["氏名"] == "鈴木 花子"

    def test_nan_filled_with_empty_string(self):
        """NaN セルが空文字に変換されること"""
        f = self._make_file("name,memo\n山田,\n", "utf-8")
        df = load_csv_with_encoding(f)
        assert df.iloc[0]["memo"] == ""

    def test_whitespace_trimmed(self):
        """セル値の前後の空白が除去されること"""
        f = self._make_file("name,company\n  山田  ,  ABC社  \n", "utf-8")
        df = load_csv_with_encoding(f)
        assert df.iloc[0]["name"] == "山田"
        assert df.iloc[0]["company"] == "ABC社"

    def test_returns_all_string_dtype(self):
        """数値カラムも文字列型で読み込まれること（dtype=str 指定）"""
        f = self._make_file("id,score\n1,100\n2,200\n", "utf-8")
        df = load_csv_with_encoding(f)
        # dtype=str で読み込むため数値も文字列になること
        assert df.iloc[0]["id"] == "1"
        assert df.iloc[0]["score"] == "100"

    def test_multirow_utf8_bom(self):
        """複数行の BOM 付き UTF-8 CSV が全行読み込まれること"""
        content = "name,company\n山田,ABC社\n鈴木,XYZ社\n田中,DEF社\n"
        f = io.BytesIO(content.encode("utf-8-sig"))
        df = load_csv_with_encoding(f)
        assert len(df) == 3
        assert list(df["name"]) == ["山田", "鈴木", "田中"]

    def test_all_encodings_fail_raises_value_error(self):
        """全エンコーディングで読み込み失敗したとき ValueError が発生すること"""
        f = io.BytesIO(b"dummy content")
        with patch("src.utils.pd.read_csv", side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "x")):
            with pytest.raises(ValueError, match="エンコーディング"):
                load_csv_with_encoding(f)
