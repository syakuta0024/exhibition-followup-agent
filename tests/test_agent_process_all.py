"""tests/test_agent_process_all.py — FollowUpAgent.process_all_leads() のテスト

process_lead() のバルク処理ラッパーを検証する:
  - 入力件数と同数の結果が返ること
  - 1件エラーでも残りの処理が継続されること
  - エラー時の結果構造
  - crm_df が各 process_lead に渡されること
  - 空 DataFrame で空リストが返ること
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ── ヘルパー ─────────────────────────────────────────────────────────────────

def _make_agent():
    """外部依存をすべてモック化したテスト用 FollowUpAgent"""
    from src.agent import FollowUpAgent
    vectordb = MagicMock()
    vectordb.is_index_built.return_value = False
    vectordb.search_tech_docs.return_value = []
    vectordb.search_crm.return_value = []
    email_gen = MagicMock()
    email_gen.generate.return_value = {
        "subject": "テスト件名",
        "body": "テスト本文",
        "cta": "CTA",
    }
    return FollowUpAgent(vectordb_manager=vectordb, email_generator=email_gen)


def _make_leads_df(n: int = 2) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "visitor_name": f"テスト{i + 1}",
            "company_name": f"テスト社{i + 1}",
            "email": f"test{i + 1}@test.co.jp",
            "lead_rank": "B",
            "interested_products": "",
            "memo": "",
        })
    return pd.DataFrame(rows)


# process_lead が返す最小限の成功結果テンプレート
_OK_RESULT_BASE = {
    "lead_id": "", "lead_rank": "B",
    "email_to": "", "subject": "件名", "body": "本文", "cta": "",
    "ref_tech_docs": [], "ref_crm": [], "crm_match_score": 0,
    "crm_deal_stage": "", "crm_source": "none", "quality_score": 60,
    "retrieved_tech_chunks": [], "retrieved_crm_chunks": [],
    "crm_structured": None, "web_search_results": [],
    "rank_info": {}, "validation_passed": True,
    "validation_errors": [], "validation_warnings": [],
    "judge_passed": None, "judge_score": None, "judge_issues": [],
}


# ── テストケース ──────────────────────────────────────────────────────────────

class TestProcessAllLeads:

    def test_returns_same_count_as_input_leads(self):
        """入力リード数と同じ件数の結果リストが返ること"""
        agent = _make_agent()
        leads_df = _make_leads_df(n=3)

        with patch.object(
            agent,
            "process_lead",
            side_effect=lambda lead, **kw: {
                **_OK_RESULT_BASE,
                "visitor_name": lead.get("visitor_name", ""),
                "company_name": lead.get("company_name", ""),
            },
        ):
            results = agent.process_all_leads(
                leads_df,
                enable_web_search=False,
                enable_rank_estimation=False,
            )

        assert len(results) == 3

    def test_empty_dataframe_returns_empty_list(self):
        """空の DataFrame を渡したとき空リストが返ること"""
        agent = _make_agent()
        results = agent.process_all_leads(pd.DataFrame())
        assert results == []

    def test_error_in_one_lead_does_not_stop_others(self):
        """1件でエラーが発生しても残りのリードが処理されること"""
        agent = _make_agent()
        leads_df = _make_leads_df(n=3)

        call_count = [0]

        def side_effect(lead, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("2件目の処理エラー")
            return {
                **_OK_RESULT_BASE,
                "visitor_name": lead.get("visitor_name", ""),
                "company_name": lead.get("company_name", ""),
            }

        with patch.object(agent, "process_lead", side_effect=side_effect):
            results = agent.process_all_leads(
                leads_df,
                enable_web_search=False,
                enable_rank_estimation=False,
            )

        assert len(results) == 3
        # 2件目は ERROR
        assert results[1]["subject"] == "ERROR"
        assert "2件目" in results[1]["body"]
        # 1件目・3件目は正常
        assert results[0]["subject"] == "件名"
        assert results[2]["subject"] == "件名"

    def test_error_result_contains_visitor_and_company_info(self):
        """エラー時の結果に visitor_name と company_name が含まれること"""
        agent = _make_agent()
        leads_df = pd.DataFrame([{
            "visitor_name": "エラー太郎",
            "company_name": "エラー社",
            "email": "err@error.co.jp",
            "lead_rank": "C",
            "interested_products": "",
            "memo": "",
        }])

        with patch.object(
            agent,
            "process_lead",
            side_effect=RuntimeError("テストエラー"),
        ):
            results = agent.process_all_leads(
                leads_df,
                enable_web_search=False,
            )

        assert len(results) == 1
        assert results[0]["visitor_name"] == "エラー太郎"
        assert results[0]["company_name"] == "エラー社"
        assert results[0]["subject"] == "ERROR"

    def test_crm_df_passed_to_each_process_lead_call(self):
        """crm_df が各 process_lead 呼び出しに正しく渡されること"""
        agent = _make_agent()
        leads_df = _make_leads_df(n=2)
        crm_df = pd.DataFrame([{
            "email": "crm@crm.co.jp",
            "company_name": "CRM社",
        }])

        received_crm_dfs = []

        def capture(lead, crm_df=None, **kwargs):
            received_crm_dfs.append(crm_df)
            return {
                **_OK_RESULT_BASE,
                "visitor_name": lead.get("visitor_name", ""),
                "company_name": lead.get("company_name", ""),
            }

        with patch.object(agent, "process_lead", side_effect=capture):
            agent.process_all_leads(leads_df, crm_df=crm_df)

        assert len(received_crm_dfs) == 2
        for received in received_crm_dfs:
            assert received is crm_df
