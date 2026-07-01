"""tests/test_cli_runner_utils.py — cli_runner.py の未テスト関数群

run_check() / _manage_output_path() / save_product_knowledge() /
load_product_knowledge() / run_draft_to_gmail() の CSV 読み込み分岐を検証する。

LLM・ChromaDB・Gmail API 呼び出しはすべてモック化する。
"""

import re
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


# ── run_check() ─────────────────────────────────────────────────────────────

class TestRunCheck:
    """run_check() が返す 8 項目のステータス分岐を検証する。"""

    def _item(self, result: dict, label: str) -> dict:
        for item in result["items"]:
            if item["label"] == label:
                return item
        return {"status": "not_found", "detail": ""}

    # -------- API キー --------

    def test_api_key_present_returns_ok(self, tmp_path, monkeypatch):
        """OPENAI_API_KEY が設定済みのとき ok ステータスが返ること"""
        monkeypatch.chdir(tmp_path)
        from src.config import Config
        monkeypatch.setattr(Config, "OPENAI_API_KEY", "sk-test")

        from src.cli_runner import run_check
        result = run_check()

        assert self._item(result, "OPENAI_API_KEY")["status"] == "ok"

    def test_api_key_absent_returns_error_and_ok_false(self, tmp_path, monkeypatch):
        """OPENAI_API_KEY が未設定のとき error ステータスかつ ok=False になること"""
        monkeypatch.chdir(tmp_path)
        from src.config import Config
        monkeypatch.setattr(Config, "OPENAI_API_KEY", "")

        from src.cli_runner import run_check
        result = run_check()

        assert result["ok"] is False
        assert self._item(result, "OPENAI_API_KEY")["status"] == "error"

    # -------- 送信元会社名 --------

    def test_sender_company_set_returns_ok_with_name_in_detail(self, tmp_path, monkeypatch):
        """sender_company が設定済みのとき ok かつ会社名が detail に含まれること"""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "cli_config.yaml").write_text(
            "sender_company: テスト株式会社\n", encoding="utf-8"
        )
        from src.config import Config
        monkeypatch.setattr(Config, "OPENAI_API_KEY", "sk-test")

        from src.cli_runner import run_check
        result = run_check()

        item = self._item(result, "送信元会社名")
        assert item["status"] == "ok"
        assert "テスト株式会社" in item["detail"]

    def test_sender_company_not_set_returns_warning(self, tmp_path, monkeypatch):
        """sender_company 未設定のとき warning になること"""
        monkeypatch.chdir(tmp_path)
        from src.config import Config
        monkeypatch.setattr(Config, "OPENAI_API_KEY", "sk-test")

        from src.cli_runner import run_check
        result = run_check()

        assert self._item(result, "送信元会社名")["status"] == "warning"

    # -------- リードCSV --------

    def test_leads_csv_found_returns_ok(self, tmp_path, monkeypatch):
        """リードCSVが存在するとき ok になること"""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "leads.csv").write_text(
            "visitor_name,company_name\n山田,ABC社\n", encoding="utf-8"
        )
        from src.config import Config
        monkeypatch.setattr(Config, "OPENAI_API_KEY", "sk-test")

        from src.cli_runner import run_check
        result = run_check()

        assert self._item(result, "リードCSV")["status"] == "ok"

    def test_leads_csv_not_found_returns_warning(self, tmp_path, monkeypatch):
        """リードCSVが存在しないとき warning になること"""
        monkeypatch.chdir(tmp_path)
        from src.config import Config
        monkeypatch.setattr(Config, "OPENAI_API_KEY", "sk-test")

        from src.cli_runner import run_check
        result = run_check()

        assert self._item(result, "リードCSV")["status"] == "warning"

    # -------- Gmail credentials --------

    def test_gmail_credentials_present_returns_ok(self, tmp_path, monkeypatch):
        """credentials.json が存在するとき ok になること"""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "credentials").mkdir()
        (tmp_path / "credentials" / "credentials.json").write_text(
            '{"installed": {}}', encoding="utf-8"
        )
        from src.config import Config
        monkeypatch.setattr(Config, "OPENAI_API_KEY", "sk-test")

        from src.cli_runner import run_check
        result = run_check()

        assert self._item(result, "Gmail credentials")["status"] == "ok"

    def test_gmail_credentials_absent_returns_warning(self, tmp_path, monkeypatch):
        """credentials.json が未配置のとき warning になること"""
        monkeypatch.chdir(tmp_path)
        from src.config import Config
        monkeypatch.setattr(Config, "OPENAI_API_KEY", "sk-test")

        from src.cli_runner import run_check
        result = run_check()

        assert self._item(result, "Gmail credentials")["status"] == "warning"

    # -------- overall ok フラグ --------

    def test_overall_ok_true_when_only_warnings(self, tmp_path, monkeypatch):
        """エラーなく警告のみの場合 ok=True になること"""
        monkeypatch.chdir(tmp_path)
        from src.config import Config
        monkeypatch.setattr(Config, "OPENAI_API_KEY", "sk-test")

        from src.cli_runner import run_check
        result = run_check()

        # API キーあり・その他は warning 止まり → ok=True
        assert result["ok"] is True

    def test_result_contains_items_list(self, tmp_path, monkeypatch):
        """戻り値に items リストが含まれること"""
        monkeypatch.chdir(tmp_path)
        from src.config import Config
        monkeypatch.setattr(Config, "OPENAI_API_KEY", "sk-test")

        from src.cli_runner import run_check
        result = run_check()

        assert "items" in result
        assert isinstance(result["items"], list)
        assert len(result["items"]) >= 8


# ── _manage_output_path() ─────────────────────────────────────────────────────

class TestManageOutputPath:

    def test_overwrite_mode_returns_base_path_unchanged(self, tmp_path):
        """overwrite モードのとき base_path をそのまま返すこと"""
        from src.cli_runner import _manage_output_path
        base = str(tmp_path / "output" / "emails.csv")
        result = _manage_output_path("overwrite", base)
        assert result == base

    def test_timestamp_mode_returns_timestamped_path(self, tmp_path):
        """timestamp モードで既存ファイルなしのとき、タイムスタンプ付きパスを返すこと"""
        from src.cli_runner import _manage_output_path
        base = str(tmp_path / "output" / "emails.csv")
        result = _manage_output_path("timestamp", base)
        assert result != base
        assert re.search(r"emails_\d{8}_\d{6}\.csv$", result)

    def test_timestamp_mode_existing_file_moves_to_legacy(self, tmp_path):
        """timestamp モードで既存ファイルがあるとき legacy/ に退避されること"""
        from src.cli_runner import _manage_output_path

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        base = str(out_dir / "emails.csv")
        Path(base).write_text("旧出力データ", encoding="utf-8")

        new_path = _manage_output_path("timestamp", base)

        # 元ファイルが消えていること
        assert not Path(base).exists()
        # 新パスがタイムスタンプ付きであること
        assert re.search(r"emails_\d{8}_\d{6}\.csv$", new_path)
        # legacy/ に退避ファイルがあること
        legacy_dir = out_dir / "legacy"
        assert legacy_dir.is_dir()
        legacy_files = list(legacy_dir.iterdir())
        assert len(legacy_files) == 1
        assert re.search(r"emails_\d{8}_\d{6}\.csv$", legacy_files[0].name)


# ── save_product_knowledge() / load_product_knowledge() ──────────────────────

class TestProductKnowledge:

    @pytest.fixture(autouse=True)
    def isolate_path(self, tmp_path, monkeypatch):
        """保存先を tmp_path に隔離する"""
        import src.cli_runner as cr
        monkeypatch.setattr(
            cr, "PRODUCT_KNOWLEDGE_PATH", str(tmp_path / "product_knowledge.yaml")
        )

    def test_save_creates_yaml_file(self, tmp_path):
        """save_product_knowledge() が YAML ファイルを作成すること"""
        from src.cli_runner import save_product_knowledge
        save_product_knowledge({"Sorani": "クラウド在庫管理ツール"})
        assert (tmp_path / "product_knowledge.yaml").exists()

    def test_load_reads_saved_data(self):
        """保存した内容を load_product_knowledge() で正しく読み込めること"""
        from src.cli_runner import save_product_knowledge, load_product_knowledge
        products = {"Sorani": "在庫管理", "EdgeGuard": "セキュリティ"}
        save_product_knowledge(products)
        loaded = load_product_knowledge()
        assert loaded == products

    def test_load_missing_file_returns_empty_dict(self):
        """ファイルが存在しないとき空 dict を返すこと（クラッシュしない）"""
        from src.cli_runner import load_product_knowledge
        result = load_product_knowledge()
        assert result == {}

    def test_save_multiline_value_uses_literal_block_style(self, tmp_path):
        """改行を含む値がリテラルブロック（|）スタイルで保存されること"""
        from src.cli_runner import save_product_knowledge
        save_product_knowledge({
            "Sorani": "概要: 在庫管理\n価格: 50万円/月\n導入実績: 50社"
        })
        content = (tmp_path / "product_knowledge.yaml").read_text(encoding="utf-8")
        # YAML リテラルブロックスタイルのマーカー
        assert "|" in content

    def test_load_corrupted_yaml_returns_empty_dict(self, tmp_path):
        """壊れた YAML ファイルでも空 dict を返すこと（クラッシュしない）"""
        from src.cli_runner import load_product_knowledge
        (tmp_path / "product_knowledge.yaml").write_text(
            "{broken yaml [[[", encoding="utf-8"
        )
        result = load_product_knowledge()
        assert result == {}


# ── run_draft_to_gmail() CSV 読み込み分岐 ─────────────────────────────────────

class TestRunDraftToGmailCsvBranch:

    def test_nonexistent_csv_returns_error(self, tmp_path, monkeypatch):
        """存在しない CSV パスを渡したとき ok=False が返ること"""
        monkeypatch.chdir(tmp_path)
        from src.cli_runner import run_draft_to_gmail
        result = run_draft_to_gmail(output_csv_path="nonexistent_emails.csv")
        assert result["ok"] is False
        assert "見つかりません" in result["message"]

    def test_valid_csv_calls_drafter_and_returns_ok(self, tmp_path):
        """有効な CSV パスを渡したとき GmailDrafter が呼ばれ ok=True が返ること"""
        csv_path = tmp_path / "emails.csv"
        df = pd.DataFrame([{
            "email_to": "a@a.jp",
            "subject": "【御礼】展示会ご来場ありがとうございました",
            "body": "テスト本文",
            "visitor_name": "山田 太郎",
            "company_name": "ABC社",
        }])
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # run_draft_to_gmail 内で「from src.gmail_drafter import GmailDrafter」
        # を経由するため src.gmail_drafter.GmailDrafter をパッチする
        with patch("src.gmail_drafter.GmailDrafter") as MockDrafter:
            mock_instance = MockDrafter.return_value
            mock_instance.create_drafts_from_results.return_value = {
                "success": 1,
                "errors": 0,
                "draft_ids": ["DRAFT_001"],
                "error_details": [],
            }
            from src.cli_runner import run_draft_to_gmail
            result = run_draft_to_gmail(output_csv_path=str(csv_path))

        assert result["ok"] is True
        assert result["success"] == 1
        mock_instance.create_drafts_from_results.assert_called_once()

    def test_no_args_reads_output_path_from_config(self, tmp_path, monkeypatch):
        """results=None, output_csv_path=None のとき設定ファイルのパスが参照されること"""
        monkeypatch.chdir(tmp_path)
        # 存在しないパスを cli_config.yaml に設定
        (tmp_path / "cli_config.yaml").write_text(
            "output_path: output/emails_not_created.csv\n", encoding="utf-8"
        )
        from src.cli_runner import run_draft_to_gmail
        result = run_draft_to_gmail()
        assert result["ok"] is False
        assert "見つかりません" in result["message"]
