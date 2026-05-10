"""tests/test_email_judge.py — email_judge の単体テスト（LLM 呼び出しはモック）"""

import pytest
from unittest.mock import patch

from src.email_judge import judge_email, _parse_judge_response, JudgeResult


LEAD_A = {"interested_products": "DigiMA", "memo": "IoT基盤に興味あり"}
LEAD_C = {"interested_products": "EdgeGuard", "memo": "情報収集中"}


# ── _parse_judge_response ──────────────────────────────────────────────

class TestParseJudgeResponse:
    def test_valid_passed(self):
        result = _parse_judge_response(
            '{"passed": true, "score": 90, "issues": [], "recommendation": "問題なし"}'
        )
        assert result.passed is True
        assert result.score == 90
        assert result.issues == []
        assert result.recommendation == "問題なし"

    def test_valid_failed(self):
        result = _parse_judge_response(
            '{"passed": false, "score": 40, "issues": ["内部ID混入"], "recommendation": "要修正"}'
        )
        assert result.passed is False
        assert result.score == 40
        assert "内部ID混入" in result.issues
        assert result.recommendation == "要修正"

    def test_json_in_markdown_block(self):
        result = _parse_judge_response(
            '```json\n{"passed": true, "score": 85, "issues": [], "recommendation": "良好"}\n```'
        )
        assert result.passed is True
        assert result.score == 85

    def test_json_in_plain_code_block(self):
        result = _parse_judge_response(
            '```\n{"passed": false, "score": 50, "issues": ["トーン不一致"], "recommendation": "修正"}\n```'
        )
        assert result.passed is False
        assert result.score == 50

    def test_parse_failure_returns_failed_result(self):
        result = _parse_judge_response("これはJSONではありません")
        assert result.passed is False
        assert result.score == 0
        assert any("パース失敗" in issue for issue in result.issues)

    def test_parse_failure_stores_raw_text(self):
        raw = "エラーレスポンスの内容"
        result = _parse_judge_response(raw)
        assert raw in result.recommendation

    def test_missing_fields_use_defaults(self):
        result = _parse_judge_response('{"passed": true}')
        assert result.score == 0
        assert result.issues == []
        assert result.recommendation == ""

    def test_score_is_int(self):
        result = _parse_judge_response(
            '{"passed": true, "score": 75.9, "issues": [], "recommendation": ""}'
        )
        assert isinstance(result.score, int)
        assert result.score == 75


# ── judge_email（_call_llm をモック）─────────────────────────────────

class TestJudgeEmail:
    def test_judge_passed(self):
        mock_response = '{"passed": true, "score": 88, "issues": [], "recommendation": "良好"}'
        with patch("src.email_judge._call_llm", return_value=mock_response):
            result = judge_email(
                subject="ご来場御礼",
                body="先日はご来場いただきありがとうございました。DigiMA のご紹介をさせてください。",
                lead=LEAD_A,
                lead_rank="A",
            )
        assert result.passed is True
        assert result.score == 88
        assert result.issues == []

    def test_judge_failed(self):
        mock_response = '{"passed": false, "score": 30, "issues": ["トーン不適切", "情報不足"], "recommendation": "修正が必要"}'
        with patch("src.email_judge._call_llm", return_value=mock_response):
            result = judge_email(
                subject="件名",
                body="本文",
                lead=LEAD_C,
                lead_rank="C",
            )
        assert result.passed is False
        assert result.score == 30
        assert "トーン不適切" in result.issues

    def test_judge_uses_config_model_when_none(self):
        """llm_model=None のとき Config.LLM_MODEL が使われること"""
        from src.config import Config
        mock_response = '{"passed": true, "score": 80, "issues": [], "recommendation": ""}'
        with patch("src.email_judge._call_llm", return_value=mock_response) as mock_llm:
            judge_email(
                subject="件名", body="本文",
                lead=LEAD_A, lead_rank="B",
            )
        called_model = mock_llm.call_args.kwargs.get("model") or mock_llm.call_args.args[0]
        assert called_model == Config.LLM_MODEL

    def test_judge_uses_custom_model(self):
        """llm_model を明示指定したとき、そのモデルが使われること"""
        mock_response = '{"passed": true, "score": 80, "issues": [], "recommendation": ""}'
        with patch("src.email_judge._call_llm", return_value=mock_response) as mock_llm:
            judge_email(
                subject="件名", body="本文",
                lead=LEAD_A, lead_rank="A",
                llm_model="gpt-4.1",
            )
        called_model = mock_llm.call_args.kwargs.get("model") or mock_llm.call_args.args[0]
        assert called_model == "gpt-4.1"

    def test_judge_llm_parse_failure_does_not_raise(self):
        """LLM が不正レスポンスを返してもクラッシュしない"""
        with patch("src.email_judge._call_llm", return_value="不正なレスポンス"):
            result = judge_email(
                subject="件名", body="本文",
                lead=LEAD_A, lead_rank="B",
            )
        assert result.passed is False
        assert any("パース失敗" in i for i in result.issues)
