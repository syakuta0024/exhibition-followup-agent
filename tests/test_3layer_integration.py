"""tests/test_3layer_integration.py — 3層検証アーキテクチャの統合テスト

agent.py の process_lead を通じて:
- Layer 1 (email_validator) と Layer 2 (email_judge) の呼び出し制御を検証する。
- LLM 呼び出しはすべてモック（API コスト発生なし）。
"""

from unittest.mock import MagicMock, patch

import pytest

from src.email_validator import ValidationResult
from src.email_judge import JudgeResult


# ── フィクスチャ ────────────────────────────────────────────────────────

def _make_agent():
    """テスト用の FollowUpAgent（vectordb・email_gen はモック）"""
    from src.agent import FollowUpAgent

    vectordb = MagicMock()
    vectordb.is_index_built.return_value = False

    email_gen = MagicMock()
    email_gen.generate.return_value = {
        "subject": "テスト件名",
        "body": "テスト本文。DigiMA のご紹介をさせてください。",
        "cta": "",
    }

    return FollowUpAgent(vectordb_manager=vectordb, email_generator=email_gen)


LEAD = {
    "visitor_name": "テスト太郎",
    "company_name": "テスト株式会社",
    "email": "test@example.com",
    "lead_rank": "B",
    "interested_products": "DigiMA",
    "memo": "",
}

_PASSED_VALIDATION = ValidationResult(passed=True, errors=[], warnings=[])
_FAILED_VALIDATION = ValidationResult(passed=False, errors=["内部ID混入: ['L001']"], warnings=[])
_PASSED_JUDGE      = JudgeResult(passed=True, score=85, issues=[], recommendation="良好")


# ── テスト ────────────────────────────────────────────────────────────

class TestThreeLayerIntegration:

    def test_layer2_skipped_when_layer1_fails(self):
        """Layer 1 失敗時に Layer 2 (judge_email) が呼ばれないこと"""
        agent = _make_agent()

        # 遅延インポート（from src.email_validator import validate_email）なので
        # ソースモジュールをパッチする
        with patch("src.email_validator.validate_email", return_value=_FAILED_VALIDATION) as mock_v, \
             patch("src.email_judge.judge_email") as mock_j:

            result = agent.process_lead(
                lead=LEAD,
                enable_web_search=False,
                enable_rank_estimation=False,
                enable_llm_judge=True,
            )

        mock_v.assert_called_once()
        mock_j.assert_not_called()
        assert result["validation_passed"] is False
        assert result["judge_passed"] is None
        assert result["judge_score"] is None
        assert result["judge_issues"] == []

    def test_layer2_skipped_when_flag_off(self):
        """enable_llm_judge=False のとき Layer 2 がスキップされること"""
        agent = _make_agent()

        with patch("src.email_validator.validate_email", return_value=_PASSED_VALIDATION), \
             patch("src.email_judge.judge_email") as mock_j:

            result = agent.process_lead(
                lead=LEAD,
                enable_web_search=False,
                enable_rank_estimation=False,
                enable_llm_judge=False,
            )

        mock_j.assert_not_called()
        assert result["judge_passed"] is None
        assert result["judge_score"] is None
        assert result["judge_issues"] == []

    def test_layer2_runs_when_flag_on_and_layer1_passes(self):
        """enable_llm_judge=True かつ Layer 1 通過時に Layer 2 が呼ばれること"""
        agent = _make_agent()

        with patch("src.email_validator.validate_email", return_value=_PASSED_VALIDATION), \
             patch("src.email_judge.judge_email", return_value=_PASSED_JUDGE) as mock_j:

            result = agent.process_lead(
                lead=LEAD,
                enable_web_search=False,
                enable_rank_estimation=False,
                enable_llm_judge=True,
            )

        mock_j.assert_called_once()
        assert result["validation_passed"] is True
        assert result["judge_passed"] is True
        assert result["judge_score"] == 85
        assert result["judge_issues"] == []
