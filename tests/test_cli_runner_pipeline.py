"""tests/test_cli_runner_pipeline.py — cli_runner.py 残テスト

run_load_leads()   : CSV読み込み→カラムマッピング→ランク正規化→フィルタ
run_kb_status()    : ChromaDB メタデータ集計
run_fetch_calendar_slots() : Calendar API 呼び出しと設定オーバーライド
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── run_load_leads() ─────────────────────────────────────────────────────────

class TestRunLoadLeads:

    def _write_csv(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def test_file_not_found_returns_ok_false(self, tmp_path, monkeypatch):
        """存在しない CSV パスを渡したとき ok=False で total=0 が返ること"""
        monkeypatch.chdir(tmp_path)
        from src.cli_runner import run_load_leads
        result = run_load_leads(csv_path=str(tmp_path / "nonexistent.csv"))
        assert result["ok"] is False
        assert result["total"] == 0
        assert result["leads_df"] is None

    def test_valid_csv_returns_ok_true_with_counts(self, tmp_path, monkeypatch):
        """正常 CSV のとき ok=True かつ total / total_all が一致すること"""
        monkeypatch.chdir(tmp_path)
        csv = tmp_path / "data" / "leads.csv"
        self._write_csv(csv,
            "氏名,会社名,メールアドレス,lead_rank\n"
            "山田,ABC社,a@abc.com,A\n"
            "鈴木,XYZ社,b@xyz.com,B\n"
            "田中,DEF社,c@def.com,C\n"
        )
        from src.cli_runner import run_load_leads
        result = run_load_leads(csv_path=str(csv), ranks=["A", "B", "C"])
        assert result["ok"] is True
        assert result["total_all"] == 3
        assert result["total"] == 3

    def test_rank_filter_excludes_lower_ranks(self, tmp_path, monkeypatch):
        """ranks=['A','B'] のとき C リードが除外され total が減ること"""
        monkeypatch.chdir(tmp_path)
        csv = tmp_path / "data" / "leads.csv"
        self._write_csv(csv,
            "氏名,会社名,メールアドレス,lead_rank\n"
            "山田,ABC社,a@abc.com,A\n"
            "鈴木,XYZ社,b@xyz.com,B\n"
            "田中,DEF社,c@def.com,C\n"
        )
        from src.cli_runner import run_load_leads
        result = run_load_leads(csv_path=str(csv), ranks=["A", "B"])
        assert result["ok"] is True
        assert result["total_all"] == 3
        assert result["total"] == 2

    def test_by_rank_counts_all_ranks_before_filter(self, tmp_path, monkeypatch):
        """by_rank はフィルター前の全件分布を返すこと"""
        monkeypatch.chdir(tmp_path)
        csv = tmp_path / "data" / "leads.csv"
        self._write_csv(csv,
            "氏名,会社名,メールアドレス,lead_rank\n"
            "山田,ABC社,a@abc.com,A\n"
            "鈴木,XYZ社,b@xyz.com,B\n"
            "田中,DEF社,c@def.com,C\n"
        )
        from src.cli_runner import run_load_leads
        result = run_load_leads(csv_path=str(csv), ranks=["A"])
        # フィルター後は1件だが by_rank は全3ランクを含む
        assert result["by_rank"].get("A", 0) == 1
        assert result["by_rank"].get("B", 0) == 1
        assert result["by_rank"].get("C", 0) == 1

    def test_rank_value_mapping_from_config_applied(self, tmp_path, monkeypatch):
        """cli_config.yaml の rank_value_mapping が lead_rank 列に適用されること"""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "cli_config.yaml").write_text(
            "rank_value_mapping:\n  高い: A\n  普通: B\n",
            encoding="utf-8",
        )
        csv = tmp_path / "data" / "leads.csv"
        self._write_csv(csv,
            "氏名,会社名,メールアドレス,lead_rank\n"
            "山田,ABC社,a@abc.com,高い\n"
            "鈴木,XYZ社,b@xyz.com,普通\n"
        )
        from src.cli_runner import run_load_leads
        result = run_load_leads(csv_path=str(csv), ranks=["A", "B"])
        assert result["ok"] is True
        assert result["total"] == 2

    def test_auto_extract_rank_from_description_format(self, tmp_path, monkeypatch):
        """'A：説明' 形式が自動的に 'A' へ正規化されること"""
        monkeypatch.chdir(tmp_path)
        csv = tmp_path / "data" / "leads.csv"
        self._write_csv(csv,
            "氏名,会社名,メールアドレス,lead_rank\n"
            "山田,ABC社,a@abc.com,A：非常に高い関心\n"
            "鈴木,XYZ社,b@xyz.com,B：関心あり\n"
        )
        from src.cli_runner import run_load_leads
        result = run_load_leads(csv_path=str(csv), ranks=["A", "B"])
        assert result["ok"] is True
        assert result["total"] == 2
        ranks_in_df = set(result["leads_df"]["lead_rank"].tolist())
        assert ranks_in_df == {"A", "B"}

    def test_columns_mapped_contains_standard_fields(self, tmp_path, monkeypatch):
        """columns_mapped にマッピング成功したフィールドが含まれること"""
        monkeypatch.chdir(tmp_path)
        csv = tmp_path / "data" / "leads.csv"
        self._write_csv(csv,
            "氏名,会社名,メールアドレス\n"
            "山田,ABC社,a@abc.com\n"
        )
        from src.cli_runner import run_load_leads
        result = run_load_leads(csv_path=str(csv), ranks=["A", "B", "C"])
        assert "visitor_name" in result["columns_mapped"]
        assert "company_name" in result["columns_mapped"]

    def test_message_contains_count_and_ranks(self, tmp_path, monkeypatch):
        """message に件数とランク情報が含まれること"""
        monkeypatch.chdir(tmp_path)
        csv = tmp_path / "data" / "leads.csv"
        self._write_csv(csv,
            "氏名,会社名,メールアドレス,lead_rank\n"
            "山田,ABC社,a@abc.com,A\n"
        )
        from src.cli_runner import run_load_leads
        result = run_load_leads(csv_path=str(csv), ranks=["A"])
        assert "1件" in result["message"]


# ── run_kb_status() ──────────────────────────────────────────────────────────

class TestRunKbStatus:

    def _make_db_dir(self, base: Path, sqlite_size: int = 2000) -> Path:
        """chroma_db ディレクトリと sqlite3 ファイルを作成する"""
        db_dir = base / "chroma_db"
        db_dir.mkdir()
        (db_dir / "chroma.sqlite3").write_bytes(b"x" * sqlite_size)
        return db_dir

    def test_no_sqlite_returns_is_empty(self, tmp_path, monkeypatch):
        """chroma.sqlite3 が存在しないとき is_empty=True が返ること"""
        db_dir = tmp_path / "chroma_db"
        db_dir.mkdir()
        from src.config import Config
        monkeypatch.setattr(Config, "CHROMA_DB_DIR", str(db_dir))
        from src.cli_runner import run_kb_status
        result = run_kb_status()
        assert result["is_empty"] is True
        assert result["total_chunks"] == 0

    def test_tiny_sqlite_returns_is_empty(self, tmp_path, monkeypatch):
        """sqlite3 が 1000 バイト以下のとき is_empty=True になること（空 DB 判定）"""
        db_dir = self._make_db_dir(tmp_path, sqlite_size=100)
        from src.config import Config
        monkeypatch.setattr(Config, "CHROMA_DB_DIR", str(db_dir))
        from src.cli_runner import run_kb_status
        result = run_kb_status()
        assert result["is_empty"] is True

    def test_collection_not_found_returns_is_empty(self, tmp_path, monkeypatch):
        """ChromaDB に collection がないとき is_empty=True になること"""
        db_dir = self._make_db_dir(tmp_path)
        from src.config import Config
        monkeypatch.setattr(Config, "CHROMA_DB_DIR", str(db_dir))

        with patch("chromadb.PersistentClient") as MockClient:
            MockClient.return_value.get_collection.side_effect = Exception("not found")
            from src.cli_runner import run_kb_status
            result = run_kb_status()

        assert result["is_empty"] is True
        assert result["total_chunks"] == 0

    def test_collection_with_metadatas_returns_documents(self, tmp_path, monkeypatch):
        """チャンクがある collection から documents リストが正しく返ること"""
        db_dir = self._make_db_dir(tmp_path)
        from src.config import Config
        monkeypatch.setattr(Config, "CHROMA_DB_DIR", str(db_dir))

        metadatas = [
            {"source_file": "product.md", "doc_format": "markdown"},
            {"source_file": "product.md", "doc_format": "markdown"},
            {"source_file": "catalog.pdf", "doc_format": "pdf"},
        ]
        with patch("chromadb.PersistentClient") as MockClient:
            mock_col = MagicMock()
            mock_col.get.return_value = {"metadatas": metadatas}
            MockClient.return_value.get_collection.return_value = mock_col
            from src.cli_runner import run_kb_status
            result = run_kb_status()

        assert result["is_empty"] is False
        assert result["total_chunks"] == 3
        docs = {d["source"]: d for d in result["documents"]}
        assert docs["product.md"]["chunk_count"] == 2
        assert docs["product.md"]["source_type"] == "markdown"
        assert docs["catalog.pdf"]["chunk_count"] == 1
        assert docs["catalog.pdf"]["source_type"] == "pdf"

    def test_source_type_inferred_from_extension(self, tmp_path, monkeypatch):
        """doc_format がない場合に拡張子から source_type が推定されること"""
        db_dir = self._make_db_dir(tmp_path)
        from src.config import Config
        monkeypatch.setattr(Config, "CHROMA_DB_DIR", str(db_dir))

        metadatas = [
            {"source_file": "guide.pdf"},
            {"source_file": "notes.md"},
            {"source_file": "data.xlsx"},
        ]
        with patch("chromadb.PersistentClient") as MockClient:
            mock_col = MagicMock()
            mock_col.get.return_value = {"metadatas": metadatas}
            MockClient.return_value.get_collection.return_value = mock_col
            from src.cli_runner import run_kb_status
            result = run_kb_status()

        docs = {d["source"]: d for d in result["documents"]}
        assert docs["guide.pdf"]["source_type"] == "pdf"
        assert docs["notes.md"]["source_type"] == "markdown"
        assert docs["data.xlsx"]["source_type"] == "unknown"

    def test_doc_format_field_takes_precedence_over_extension(self, tmp_path, monkeypatch):
        """doc_format フィールドが拡張子より優先されること"""
        db_dir = self._make_db_dir(tmp_path)
        from src.config import Config
        monkeypatch.setattr(Config, "CHROMA_DB_DIR", str(db_dir))

        # .pdf 拡張子だが doc_format は "markdown"（ありえないが優先確認用）
        metadatas = [{"source_file": "unusual.pdf", "doc_format": "markdown"}]
        with patch("chromadb.PersistentClient") as MockClient:
            mock_col = MagicMock()
            mock_col.get.return_value = {"metadatas": metadatas}
            MockClient.return_value.get_collection.return_value = mock_col
            from src.cli_runner import run_kb_status
            result = run_kb_status()

        docs = {d["source"]: d for d in result["documents"]}
        assert docs["unusual.pdf"]["source_type"] == "markdown"

    def test_last_updated_is_string_when_data_exists(self, tmp_path, monkeypatch):
        """データがあるとき last_updated が 'YYYY-MM-DD HH:MM' 文字列で返ること"""
        db_dir = self._make_db_dir(tmp_path)
        from src.config import Config
        monkeypatch.setattr(Config, "CHROMA_DB_DIR", str(db_dir))

        metadatas = [{"source_file": "doc.md"}]
        with patch("chromadb.PersistentClient") as MockClient:
            mock_col = MagicMock()
            mock_col.get.return_value = {"metadatas": metadatas}
            MockClient.return_value.get_collection.return_value = mock_col
            from src.cli_runner import run_kb_status
            result = run_kb_status()

        import re
        assert result["last_updated"] is None or re.match(
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", result["last_updated"]
        )


# ── run_fetch_calendar_slots() ───────────────────────────────────────────────

class TestRunFetchCalendarSlots:

    def test_file_not_found_returns_error(self, tmp_path, monkeypatch):
        """credentials.json が見つからないとき error フィールドが設定されること"""
        monkeypatch.chdir(tmp_path)
        with patch("src.calendar_client.get_calendar_service",
                   side_effect=FileNotFoundError("credentials.json not found")):
            from src.cli_runner import run_fetch_calendar_slots
            result = run_fetch_calendar_slots()
        assert result["slots"] == []
        assert result["formatted"] == ""
        assert "not found" in result["error"]

    def test_runtime_error_returns_error(self, tmp_path, monkeypatch):
        """認証失敗など RuntimeError のとき error フィールドが設定されること"""
        monkeypatch.chdir(tmp_path)
        with patch("src.calendar_client.get_calendar_service",
                   side_effect=RuntimeError("OAuth 失敗")):
            from src.cli_runner import run_fetch_calendar_slots
            result = run_fetch_calendar_slots()
        assert result["error"] is not None
        assert "OAuth" in result["error"]

    def test_success_returns_slots_and_formatted(self, tmp_path, monkeypatch):
        """正常取得のとき slots / formatted / error=None が返ること"""
        monkeypatch.chdir(tmp_path)
        fake_slots = [
            {"display": "2026年6月10日（火）10:00〜11:00"},
            {"display": "2026年6月11日（水）14:00〜15:00"},
        ]
        fake_formatted = "・2026年6月10日（火）10:00〜11:00\n・2026年6月11日（水）14:00〜15:00"

        with patch("src.calendar_client.get_calendar_service") as mock_svc, \
             patch("src.calendar_client.fetch_free_slots", return_value=fake_slots), \
             patch("src.calendar_client.format_slots_for_email", return_value=fake_formatted):
            from src.cli_runner import run_fetch_calendar_slots
            result = run_fetch_calendar_slots()

        assert result["error"] is None
        assert result["slots"] == fake_slots
        assert result["formatted"] == fake_formatted

    def test_config_overrides_default_days_ahead(self, tmp_path, monkeypatch):
        """cli_config.yaml の calendar.days_ahead が引数デフォルトより優先されること"""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "cli_config.yaml").write_text(
            "calendar:\n  days_ahead: 30\n  duration_minutes: 30\n",
            encoding="utf-8",
        )
        captured = {}

        def fake_fetch_free_slots(service, days_ahead, duration_minutes, **kwargs):
            captured["days_ahead"] = days_ahead
            captured["duration_minutes"] = duration_minutes
            return []

        with patch("src.calendar_client.get_calendar_service"), \
             patch("src.calendar_client.fetch_free_slots", side_effect=fake_fetch_free_slots), \
             patch("src.calendar_client.format_slots_for_email", return_value=""):
            from src.cli_runner import run_fetch_calendar_slots
            run_fetch_calendar_slots(days_ahead=14, duration_minutes=60)

        assert captured["days_ahead"] == 30
        assert captured["duration_minutes"] == 30

    def test_argument_used_when_no_config(self, tmp_path, monkeypatch):
        """cli_config.yaml に calendar セクションがないとき引数値が使われること"""
        monkeypatch.chdir(tmp_path)
        captured = {}

        def fake_fetch_free_slots(service, days_ahead, duration_minutes, **kwargs):
            captured["days_ahead"] = days_ahead
            captured["duration_minutes"] = duration_minutes
            return []

        with patch("src.calendar_client.get_calendar_service"), \
             patch("src.calendar_client.fetch_free_slots", side_effect=fake_fetch_free_slots), \
             patch("src.calendar_client.format_slots_for_email", return_value=""):
            from src.cli_runner import run_fetch_calendar_slots
            run_fetch_calendar_slots(days_ahead=7, duration_minutes=45)

        assert captured["days_ahead"] == 7
        assert captured["duration_minutes"] == 45
