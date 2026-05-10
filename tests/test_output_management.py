"""tests/test_output_management.py — _manage_output_path の単体テスト"""

import os
import re

import pytest

from src.cli_runner import _manage_output_path


class TestManageOutputPath:

    def test_timestamp_creates_timestamped_path(self, tmp_path):
        """output_naming=timestamp 時にタイムスタンプ付きパスが返ること"""
        base = str(tmp_path / "emails.csv")
        result = _manage_output_path("timestamp", base)
        assert result != base
        assert re.search(r"emails_\d{8}_\d{6}\.csv$", result)

    def test_existing_file_moved_to_legacy(self, tmp_path):
        """既存の emails.csv が legacy/ に退避されること"""
        base = tmp_path / "emails.csv"
        base.write_text("old content")

        _manage_output_path("timestamp", str(base))

        # 元ファイルは消える
        assert not base.exists()
        # legacy/ に .csv ファイルが 1 件できる
        legacy_dir = tmp_path / "legacy"
        assert legacy_dir.is_dir()
        csv_files = list(legacy_dir.glob("*.csv"))
        assert len(csv_files) == 1
        assert csv_files[0].read_text() == "old content"

    def test_no_existing_file_no_legacy_dir(self, tmp_path):
        """元ファイルが存在しない場合は legacy/ ディレクトリを作らないこと"""
        base = str(tmp_path / "emails.csv")
        _manage_output_path("timestamp", base)
        assert not (tmp_path / "legacy").exists()

    def test_overwrite_returns_base_path_unchanged(self, tmp_path):
        """output_naming=overwrite 時はパスが変わらないこと"""
        base = str(tmp_path / "emails.csv")
        result = _manage_output_path("overwrite", base)
        assert result == base

    def test_overwrite_does_not_move_existing_file(self, tmp_path):
        """output_naming=overwrite 時は既存ファイルを退避しないこと"""
        base = tmp_path / "emails.csv"
        base.write_text("old content")
        _manage_output_path("overwrite", str(base))
        assert base.exists()  # 消えていない
        assert not (tmp_path / "legacy").exists()
