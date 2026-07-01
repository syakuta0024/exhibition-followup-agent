"""tests/test_gmail_drafter.py — GmailDrafter の単体テスト

Gmail API 呼び出しはすべてモック。
create_drafts_from_results() の分岐・エラースキップ・成功カウントを検証する。
"""

from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

import pytest

from src.gmail_drafter import GmailDrafter


# ── フィクスチャ ────────────────────────────────────────────────────────

def _make_drafter_with_mock_service():
    """認証済みサービスをモックで持つ GmailDrafter を返す"""
    drafter = GmailDrafter(
        credentials_path="credentials/credentials.json",
        token_path="credentials/token.json",
    )
    mock_service = MagicMock()
    drafter._service = mock_service
    return drafter, mock_service


def _ok_result(email="test@example.com", name="テスト太郎", company="テスト社"):
    return {
        "email_to": email,
        "subject": "【御礼】展示会ご来場ありがとうございました",
        "body": "テスト本文です。",
        "visitor_name": name,
        "company_name": company,
    }


# ── create_drafts_from_results() 正常系 ──────────────────────────────

class TestCreateDraftsFromResults:

    def test_all_success(self):
        """全件正常時に success カウントが正しいこと"""
        drafter, mock_service = _make_drafter_with_mock_service()

        mock_service.users.return_value.drafts.return_value.create.return_value.execute.return_value = {
            "id": "draft_001"
        }

        results = [_ok_result("a@a.jp"), _ok_result("b@b.jp")]
        out = drafter.create_drafts_from_results(results)

        assert out["success"] == 2
        assert out["errors"] == 0
        assert len(out["draft_ids"]) == 2
        assert out["error_details"] == []

    def test_skips_error_rows(self):
        """subject='ERROR' の行はスキップされること"""
        drafter, mock_service = _make_drafter_with_mock_service()

        mock_service.users.return_value.drafts.return_value.create.return_value.execute.return_value = {
            "id": "draft_002"
        }

        results = [
            _ok_result("a@a.jp"),
            {
                "email_to": "b@b.jp",
                "subject": "ERROR",
                "body": "生成エラー",
                "visitor_name": "エラー太郎",
                "company_name": "エラー社",
            },
        ]
        out = drafter.create_drafts_from_results(results)

        assert out["success"] == 1
        assert out["errors"] == 1
        assert len(out["error_details"]) == 1

    def test_skips_empty_email_to(self):
        """email_to が空の行はスキップされること"""
        drafter, mock_service = _make_drafter_with_mock_service()

        results = [
            {"email_to": "", "subject": "件名", "body": "本文", "visitor_name": "名なし太郎", "company_name": "無名社"},
        ]
        out = drafter.create_drafts_from_results(results)

        assert out["success"] == 0
        assert out["errors"] == 1
        assert "宛先メールアドレスが空" in out["error_details"][0]

    def test_api_error_counted_as_error(self):
        """API 呼び出しが例外を投げた場合 errors にカウントされること"""
        drafter, mock_service = _make_drafter_with_mock_service()

        mock_service.users.return_value.drafts.return_value.create.return_value.execute.side_effect = RuntimeError(
            "Gmail API エラー"
        )

        results = [_ok_result()]
        out = drafter.create_drafts_from_results(results)

        assert out["success"] == 0
        assert out["errors"] == 1
        assert len(out["error_details"]) == 1

    def test_empty_results_returns_zeros(self):
        """空リストを渡したとき全てゼロになること"""
        drafter, _ = _make_drafter_with_mock_service()

        out = drafter.create_drafts_from_results([])

        assert out["success"] == 0
        assert out["errors"] == 0
        assert out["draft_ids"] == []
        assert out["error_details"] == []

    def test_mixed_results_counted_correctly(self):
        """成功・ERROR・空メール・API失敗が混在する場合の集計が正しいこと"""
        drafter, mock_service = _make_drafter_with_mock_service()

        call_count = [0]
        def execute_side_effect():
            call_count[0] += 1
            if call_count[0] == 1:
                return {"id": "draft_ok"}
            raise RuntimeError("API エラー")

        mock_service.users.return_value.drafts.return_value.create.return_value.execute.side_effect = execute_side_effect

        results = [
            _ok_result("ok@ok.jp"),                            # 成功
            {"email_to": "err@err.jp", "subject": "ERROR", "body": "エラー", "visitor_name": "エラー", "company_name": "エラー社"},  # ERROR スキップ
            {"email_to": "", "subject": "件名", "body": "本文", "visitor_name": "空", "company_name": "空社"},  # 空アドレス
            _ok_result("fail@fail.jp"),                        # API エラー
        ]

        out = drafter.create_drafts_from_results(results)

        assert out["success"] == 1
        assert out["errors"] == 3


# ── create_draft() 単体 ──────────────────────────────────────────────

class TestCreateDraft:

    def test_create_draft_returns_id(self):
        """create_draft() が draft ID を返すこと"""
        drafter, mock_service = _make_drafter_with_mock_service()
        mock_service.users.return_value.drafts.return_value.create.return_value.execute.return_value = {
            "id": "DRAFT_XYZ"
        }

        draft_id = drafter.create_draft(
            to="test@example.com",
            subject="テスト件名",
            body="テスト本文",
        )

        assert draft_id == "DRAFT_XYZ"

    def test_create_draft_raises_on_http_error(self):
        """Gmail API の HttpError が RuntimeError に変換されること"""
        from googleapiclient.errors import HttpError
        drafter, mock_service = _make_drafter_with_mock_service()

        resp = MagicMock()
        resp.status = 403
        resp.reason = "Forbidden"
        http_err = HttpError(resp=resp, content=b"Forbidden")
        mock_service.users.return_value.drafts.return_value.create.return_value.execute.side_effect = http_err

        with pytest.raises(RuntimeError, match="Gmail API エラー"):
            drafter.create_draft(to="test@example.com", subject="件名", body="本文")

    def test_create_draft_encodes_subject_utf8(self):
        """日本語件名が正しく渡されること（エンコードエラーなし）"""
        drafter, mock_service = _make_drafter_with_mock_service()
        mock_service.users.return_value.drafts.return_value.create.return_value.execute.return_value = {"id": "OK"}

        # 例外なく完走すること
        draft_id = drafter.create_draft(
            to="test@example.com",
            subject="【御礼】展示会ご来場ありがとうございました",
            body="日本語本文です。",
        )
        assert draft_id == "OK"


# ── _get_service() の認証分岐 ─────────────────────────────────────

class TestGetService:

    def test_raises_file_not_found_when_no_credentials(self, tmp_path):
        """credentials.json がないとき FileNotFoundError が発生すること"""
        drafter = GmailDrafter(
            credentials_path=str(tmp_path / "nonexistent.json"),
            token_path=str(tmp_path / "token.json"),
        )

        with pytest.raises(FileNotFoundError, match="credentials.json"):
            drafter._get_service()

    def test_reuses_existing_service(self):
        """_service が設定済みのとき新たに初期化されないこと"""
        drafter, mock_service = _make_drafter_with_mock_service()

        service1 = drafter._get_service()
        service2 = drafter._get_service()

        assert service1 is service2  # 同一オブジェクトが返ること
