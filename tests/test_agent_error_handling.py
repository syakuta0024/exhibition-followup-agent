"""tests/test_agent_error_handling.py — agent.py の異常系・分岐テスト

Web 検索エラー時の継続動作、vectordb フォールバック、
process_lead() の CRM 分岐などを検証する。
LLM・外部 API 呼び出しはすべてモック。
"""

from unittest.mock import MagicMock, patch

import pytest

from src.email_validator import ValidationResult


# ── 共通ヘルパー ────────────────────────────────────────────────────────

def _make_agent(vectordb_built: bool = False):
    """テスト用 FollowUpAgent（外部依存はすべてモック）"""
    from src.agent import FollowUpAgent

    vectordb = MagicMock()
    vectordb.is_index_built.return_value = vectordb_built
    vectordb.search_tech_docs.return_value = []
    vectordb.search_crm.return_value = []

    email_gen = MagicMock()
    email_gen.generate.return_value = {
        "subject": "【御礼】展示会ご来場ありがとうございました",
        "body": "テスト本文です。EdgeGuard のご紹介をいたします。",
        "cta": "3営業日以内に電話でフォロー",
    }

    return FollowUpAgent(vectordb_manager=vectordb, email_generator=email_gen)


_LEAD = {
    "visitor_name": "山田 太郎",
    "company_name": "株式会社テクノ",
    "email": "yamada@techno.co.jp",
    "lead_rank": "B",
    "interested_products": "EdgeGuard",
    "memo": "導入コスト重視",
}

_PASSED_VALIDATION = ValidationResult(passed=True, errors=[], warnings=[])


# ── Web 検索エラー時の継続動作 ───────────────────────────────────────────

class TestWebSearchErrorHandling:

    def test_web_search_exception_continues(self):
        """Web 検索で例外が発生しても process_lead() が完走すること"""
        agent = _make_agent()

        with patch("src.agent.WebSearcher") as MockSearcher, \
             patch("src.email_validator.validate_email", return_value=_PASSED_VALIDATION):
            mock_instance = MockSearcher.return_value
            mock_instance.search_company.side_effect = RuntimeError("ネットワークエラー")
            # agent はすでに初期化済みなので直接上書き
            agent.web_searcher = mock_instance

            result = agent.process_lead(
                lead=_LEAD,
                enable_web_search=True,
                enable_rank_estimation=False,
            )

        # エラーでも結果が返ること
        assert result["visitor_name"] == "山田 太郎"
        assert result["subject"] != ""
        # web_search_results は空
        assert result["web_search_results"] == []

    def test_web_search_disabled_skips(self):
        """enable_web_search=False のとき Web 検索が呼ばれないこと"""
        agent = _make_agent()
        mock_searcher = MagicMock()
        agent.web_searcher = mock_searcher

        with patch("src.email_validator.validate_email", return_value=_PASSED_VALIDATION):
            agent.process_lead(
                lead=_LEAD,
                enable_web_search=False,
                enable_rank_estimation=False,
            )

        mock_searcher.search_company.assert_not_called()

    def test_web_search_no_results_continues(self):
        """Web 検索が success=False を返しても処理が続くこと"""
        agent = _make_agent()
        mock_searcher = MagicMock()
        mock_searcher.search_company.return_value = {
            "success": False,
            "results": [],
            "summary": "",
            "error": "検索結果なし",
        }
        agent.web_searcher = mock_searcher

        with patch("src.email_validator.validate_email", return_value=_PASSED_VALIDATION):
            result = agent.process_lead(
                lead=_LEAD,
                enable_web_search=True,
                enable_rank_estimation=False,
            )

        assert result["visitor_name"] == "山田 太郎"
        assert result["web_search_results"] == []


# ── vectordb フォールバック ────────────────────────────────────────────

class TestVectordbFallback:

    def test_crm_df_none_falls_back_to_vectordb(self):
        """crm_df=None のとき vectordb.search_crm() が呼ばれること"""
        agent = _make_agent(vectordb_built=True)
        agent.vectordb.search_crm.return_value = []

        with patch("src.email_validator.validate_email", return_value=_PASSED_VALIDATION):
            result = agent.process_lead(
                lead=_LEAD,
                crm_df=None,
                enable_web_search=False,
                enable_rank_estimation=False,
            )

        agent.vectordb.search_crm.assert_called_once()
        assert result["crm_source"] == "none"

    def test_crm_df_none_vectordb_hit(self):
        """vectordb.search_crm() がヒットした場合 crm_source='vectordb' になること"""
        agent = _make_agent(vectordb_built=True)
        agent.vectordb.search_crm.return_value = [
            {
                "text": "過去商談：テクノ社との商談記録",
                "score": 0.9,
                "metadata": {"source_file": "techno.md"},
            }
        ]

        with patch("src.email_validator.validate_email", return_value=_PASSED_VALIDATION):
            result = agent.process_lead(
                lead=_LEAD,
                crm_df=None,
                enable_web_search=False,
                enable_rank_estimation=False,
            )

        assert result["crm_source"] == "vectordb"

    def test_crm_df_empty_falls_back_to_vectordb(self):
        """空 DataFrame でも vectordb フォールバックが動くこと"""
        import pandas as pd
        agent = _make_agent(vectordb_built=True)
        agent.vectordb.search_crm.return_value = []
        empty_df = pd.DataFrame()

        with patch("src.email_validator.validate_email", return_value=_PASSED_VALIDATION):
            result = agent.process_lead(
                lead=_LEAD,
                crm_df=empty_df,
                enable_web_search=False,
                enable_rank_estimation=False,
            )

        agent.vectordb.search_crm.assert_called_once()

    def test_index_not_built_skips_vectordb_crm(self):
        """インデックス未構築のとき vectordb CRM 検索がスキップされること"""
        agent = _make_agent(vectordb_built=False)

        with patch("src.email_validator.validate_email", return_value=_PASSED_VALIDATION):
            result = agent.process_lead(
                lead=_LEAD,
                crm_df=None,
                enable_web_search=False,
                enable_rank_estimation=False,
            )

        agent.vectordb.search_crm.assert_not_called()
        assert result["crm_source"] == "none"


# ── CRM CSV マッチング分岐 ────────────────────────────────────────────

class TestCrmCsvMatching:

    def test_crm_csv_match_returns_csv_source(self):
        """CRM CSV でマッチした場合 crm_source='csv' になること"""
        import pandas as pd
        agent = _make_agent()

        crm_df = pd.DataFrame([{
            "email": "yamada@techno.co.jp",
            "company_name": "株式会社テクノ",
            "lifecycle_stage": "customer",
            "lead_status": "open",
            "last_activity_date": "2026-01-15",
            "contact_owner": "鈴木",
            "original_source": "OFFLINE",
            "create_date": "2025-06-01",
            "record_id": "CRM001",
            "first_name": "太郎",
            "last_name": "山田",
            "phone": "03-1234-5678",
            "job_title": "部長",
        }])

        with patch("src.email_validator.validate_email", return_value=_PASSED_VALIDATION):
            result = agent.process_lead(
                lead=_LEAD,
                crm_df=crm_df,
                enable_web_search=False,
                enable_rank_estimation=False,
            )

        assert result["crm_source"] == "csv"
        assert result["crm_match_score"] == 100

    def test_crm_csv_no_match_falls_back_to_vectordb(self):
        """CRM CSV でマッチしない場合 vectordb にフォールバックすること"""
        import pandas as pd
        agent = _make_agent(vectordb_built=True)
        agent.vectordb.search_crm.return_value = []

        crm_df = pd.DataFrame([{
            "email": "other@other.co.jp",
            "company_name": "無関係株式会社",
        }])

        with patch("src.email_validator.validate_email", return_value=_PASSED_VALIDATION):
            result = agent.process_lead(
                lead=_LEAD,
                crm_df=crm_df,
                enable_web_search=False,
                enable_rank_estimation=False,
            )

        agent.vectordb.search_crm.assert_called_once()


# ── on_step コールバック ─────────────────────────────────────────────

class TestOnStepCallback:

    def test_on_step_called_for_each_step(self):
        """on_step コールバックが各ステップで呼ばれること"""
        agent = _make_agent()
        steps = []

        def capture_step(num, name, status, detail=""):
            steps.append((num, name, status))

        with patch("src.email_validator.validate_email", return_value=_PASSED_VALIDATION):
            agent.process_lead(
                lead=_LEAD,
                enable_web_search=False,
                enable_rank_estimation=False,
                on_step=capture_step,
            )

        step_names = [s[1] for s in steps]
        assert "データ確認" in step_names
        assert "ベクトルDB検索" in step_names
        assert "CRM照合" in step_names
        assert "Web検索" in step_names
        assert "メール生成中" in step_names


# ── 戻り値の構造 ─────────────────────────────────────────────────────

class TestProcessLeadReturnStructure:

    def test_return_dict_contains_required_keys(self):
        """process_lead() の戻り値に必須キーが含まれること"""
        agent = _make_agent()

        required_keys = [
            "lead_id", "visitor_name", "company_name", "lead_rank", "email_to",
            "subject", "body", "cta",
            "ref_tech_docs", "ref_crm", "crm_match_score", "crm_deal_stage", "crm_source",
            "quality_score", "validation_passed", "validation_errors", "validation_warnings",
            "judge_passed", "judge_score", "judge_issues",
            "web_search_results", "rank_info",
        ]

        with patch("src.email_validator.validate_email", return_value=_PASSED_VALIDATION):
            result = agent.process_lead(
                lead=_LEAD,
                enable_web_search=False,
                enable_rank_estimation=False,
            )

        for key in required_keys:
            assert key in result, f"戻り値に '{key}' が含まれていない"

    def test_rank_info_structure(self):
        """rank_info に rank / method / confidence / original が含まれること"""
        agent = _make_agent()

        with patch("src.email_validator.validate_email", return_value=_PASSED_VALIDATION):
            result = agent.process_lead(
                lead=_LEAD,
                enable_web_search=False,
                enable_rank_estimation=False,
            )

        rank_info = result["rank_info"]
        assert "rank" in rank_info
        assert "method" in rank_info
        assert "confidence" in rank_info
        assert "original" in rank_info

    def test_lead_with_no_email_returns_empty_email_to(self):
        """メールアドレスがないリードで email_to が空文字になること"""
        agent = _make_agent()
        lead_no_email = {**_LEAD, "email": ""}

        with patch("src.email_validator.validate_email", return_value=_PASSED_VALIDATION):
            result = agent.process_lead(
                lead=lead_no_email,
                enable_web_search=False,
                enable_rank_estimation=False,
            )

        assert result["email_to"] == ""


# ── _build_audio_context ─────────────────────────────────────────────

class TestBuildAudioContext:

    def test_audio_context_empty_when_no_data(self):
        """transcript も needs もない場合は空文字を返すこと"""
        from src.agent import _build_audio_context
        result = _build_audio_context("", None)
        assert result == ""

    def test_audio_context_with_transcript_only(self):
        """transcript だけある場合は文字起こしセクションが含まれること"""
        from src.agent import _build_audio_context
        result = _build_audio_context("こんにちは、EdgeGuardに興味があります。", None)
        assert "文字起こし" in result
        assert "EdgeGuard" in result

    def test_audio_context_with_needs(self):
        """needs がある場合はニーズ・課題が含まれること"""
        from src.agent import _build_audio_context
        needs = {
            "summary": "導入コスト重視",
            "issues": "既存システムとの連携",
            "needs": "API連携機能",
            "budget": "200万円",
            "decision_maker": "CTO",
            "temperature": "warm",
        }
        result = _build_audio_context("", needs)
        assert "課題" in result
        assert "既存システムとの連携" in result
        assert "200万円" in result

    def test_audio_context_truncates_long_transcript(self):
        """2000文字を超える文字起こしは切り詰められること"""
        from src.agent import _build_audio_context
        long_text = "a" * 5000
        result = _build_audio_context(long_text, None)
        assert len(result) < 5000 + 100  # 文字起こしラベル分を考慮
