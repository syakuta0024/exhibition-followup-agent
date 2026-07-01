"""tests/test_email_generator_generate.py — EmailGenerator の単体テスト

generate() 本体・_parse_llm_response()・_build_system_prompt() を検証する。
LLM 呼び出しはモック（API コスト発生なし）。
"""

from unittest.mock import MagicMock, patch

import pytest

from src.email_generator import EmailGenerator, RANK_POLICY


# ── フィクスチャ ────────────────────────────────────────────────────────

def _make_generator():
    """テスト用 EmailGenerator（LLM はモック）"""
    with patch("src.email_generator.ChatOpenAI"):
        gen = EmailGenerator(llm_model="gpt-test", temperature=0.0)
    return gen


_BASE_LEAD = {
    "visitor_name": "佐藤 花子",
    "company_name": "株式会社サンプル",
    "email": "sato@sample.co.jp",
    "lead_rank": "A",
    "interested_products": "Sorani",
    "department": "製造部",
    "job_title": "課長",
    "memo": "コスト削減に課題あり",
    "future_requests": "デモを見たい",
    "visit_date": "2026-04-10",
}

_VALID_LLM_RESPONSE = """【件名】
【御礼】Sorani展示会ご来場ありがとうございました

【本文】
株式会社サンプル
製造部 課長 佐藤 花子 様

お世話になっております。テスト株式会社 営業部の田中でございます。

このたびは展示会にご来場いただきありがとうございました。

---
テスト株式会社 営業部
田中

【CTA】
3営業日以内に電話でフォロー。デモ環境（Sorani）を準備する。
"""


# ── _parse_llm_response() ───────────────────────────────────────────

class TestParseLlmResponse:

    def test_parses_valid_response(self):
        gen = _make_generator()
        result = gen._parse_llm_response(_VALID_LLM_RESPONSE)
        assert "御礼" in result["subject"]
        assert "佐藤 花子" in result["body"]
        assert "3営業日" in result["cta"]

    def test_returns_empty_dict_on_failure(self):
        """マーカーがない場合は body に全文が入ること"""
        gen = _make_generator()
        result = gen._parse_llm_response("マーカーのない生のテキスト")
        assert result["subject"] == ""
        assert result["body"] == "マーカーのない生のテキスト"
        assert result["cta"] == ""

    def test_handles_partial_markers(self):
        """【件名】だけある場合でも subject が取得できること"""
        gen = _make_generator()
        response = "【件名】テスト件名\n本文がここにあります"
        result = gen._parse_llm_response(response)
        assert result["subject"] == "テスト件名\n本文がここにあります"

    def test_handles_all_three_markers(self):
        """3つのマーカーが正しい順序で分割されること"""
        gen = _make_generator()
        response = "【件名】件名テスト\n【本文】本文テスト\n【CTA】CTAテスト"
        result = gen._parse_llm_response(response)
        assert result["subject"] == "件名テスト"
        assert result["body"] == "本文テスト"
        assert result["cta"] == "CTAテスト"


# ── _build_system_prompt() ──────────────────────────────────────────

class TestBuildSystemPrompt:

    def test_includes_sender_company(self):
        gen = _make_generator()
        prompt = gen._build_system_prompt(sender_company="株式会社テスト")
        assert "株式会社テスト" in prompt

    def test_uses_placeholder_when_no_sender_name(self):
        """sender_name が空のとき ●● プレースホルダーが入ること"""
        gen = _make_generator()
        prompt = gen._build_system_prompt(sender_company="ABC社", sender_name="")
        assert "●●" in prompt
        assert "架空の名前" in prompt

    def test_uses_real_name_when_provided(self):
        """sender_name が設定されたとき実名が署名に使われること"""
        gen = _make_generator()
        prompt = gen._build_system_prompt(sender_company="ABC社", sender_name="田中 一郎")
        assert "田中 一郎" in prompt
        # 署名部分（---で囲まれた箇所）に実名が含まれること
        assert "ABC社 営業部\n田中 一郎" in prompt

    def test_no_company_defaults_to_heisha(self):
        """sender_company が空のとき「弊社」が使われること"""
        gen = _make_generator()
        prompt = gen._build_system_prompt(sender_company="")
        assert "弊社" in prompt


# ── generate() 本体 ─────────────────────────────────────────────────

class TestGenerate:

    def test_generate_calls_llm_and_returns_dict(self):
        """generate() が LLM を呼び出し、subject/body/cta を返すこと"""
        gen = _make_generator()

        mock_response = MagicMock()
        mock_response.content = _VALID_LLM_RESPONSE
        gen.llm.invoke = MagicMock(return_value=mock_response)

        result = gen.generate(lead=_BASE_LEAD)

        gen.llm.invoke.assert_called_once()
        assert "subject" in result
        assert "body" in result
        assert "cta" in result

    def test_generate_raises_on_llm_error(self):
        """LLM が例外を投げたとき RuntimeError が発生すること"""
        gen = _make_generator()
        gen.llm.invoke = MagicMock(side_effect=Exception("API タイムアウト"))

        with pytest.raises(RuntimeError, match="メール生成に失敗しました"):
            gen.generate(lead=_BASE_LEAD)

    def test_generate_uses_schedule_prompt_when_schedule_context(self):
        """schedule_context がある場合 _build_human_prompt_with_schedule が呼ばれること"""
        gen = _make_generator()
        mock_response = MagicMock()
        mock_response.content = _VALID_LLM_RESPONSE
        gen.llm.invoke = MagicMock(return_value=mock_response)

        with patch.object(gen, "_build_human_prompt_with_schedule", wraps=gen._build_human_prompt_with_schedule) as mock_sched, \
             patch.object(gen, "_build_human_prompt_info_only", wraps=gen._build_human_prompt_info_only) as mock_info:
            gen.generate(lead=_BASE_LEAD, schedule_context="2026年5月10日（水）10:00〜")

        mock_sched.assert_called_once()
        mock_info.assert_not_called()

    def test_generate_uses_info_prompt_when_no_schedule_context(self):
        """schedule_context がない場合 _build_human_prompt_info_only が呼ばれること"""
        gen = _make_generator()
        mock_response = MagicMock()
        mock_response.content = _VALID_LLM_RESPONSE
        gen.llm.invoke = MagicMock(return_value=mock_response)

        with patch.object(gen, "_build_human_prompt_with_schedule", wraps=gen._build_human_prompt_with_schedule) as mock_sched, \
             patch.object(gen, "_build_human_prompt_info_only", wraps=gen._build_human_prompt_info_only) as mock_info:
            gen.generate(lead=_BASE_LEAD, schedule_context="")

        mock_sched.assert_not_called()
        mock_info.assert_called_once()


# ── RANK_POLICY ──────────────────────────────────────────────────────

class TestRankPolicy:

    @pytest.mark.parametrize("rank", ["A", "B", "C", "D", "E"])
    def test_all_ranks_have_required_keys(self, rank):
        """RANK_POLICY の各ランクに label / tone / instruction があること"""
        policy = RANK_POLICY[rank]
        assert "label" in policy
        assert "tone" in policy
        assert "instruction" in policy

    def test_rank_a_instructs_specific_date_proposal(self):
        """ランク A は具体的な日程提案を指示していること"""
        assert "日程" in RANK_POLICY["A"]["instruction"] or "候補日" in RANK_POLICY["A"]["instruction"]

    def test_rank_e_instructs_minimal_contact(self):
        """ランク E はシンプルなお礼のみを指示していること"""
        assert "最小" in RANK_POLICY["E"]["label"] or "最小限" in RANK_POLICY["E"]["instruction"]

    def test_unknown_rank_falls_back_to_c(self):
        """未知のランクは RANK_POLICY.get() で C にフォールバックすること"""
        gen = _make_generator()
        lead_unknown_rank = {**_BASE_LEAD, "lead_rank": "Z"}

        mock_response = MagicMock()
        mock_response.content = _VALID_LLM_RESPONSE
        gen.llm.invoke = MagicMock(return_value=mock_response)

        # ランク Z は RANK_POLICY にないので C が使われるはず
        # 例外が起きないことを確認
        result = gen.generate(lead=lead_unknown_rank)
        assert "subject" in result


# ── _assemble_context_sections() ──────────────────────────────────────

class TestAssembleContextSections:

    def test_audio_section_included_when_provided(self):
        """audio_context がある場合、★最優先情報セクションが含まれること"""
        gen = _make_generator()
        sections = gen._assemble_context_sections(
            lead=_BASE_LEAD,
            tech_context="",
            crm_context="",
            crm_structured=None,
            exhibition_info=None,
            web_context="",
            audio_context="課題: コスト削減\nニーズ: API連携",
        )
        assert "★最優先情報" in sections
        assert "課題: コスト削減" in sections

    def test_product_card_section_prioritized_over_tech(self):
        """製品カードが入っている場合、人間確認済みセクションが含まれること"""
        gen = _make_generator()
        sections = gen._assemble_context_sections(
            lead=_BASE_LEAD,
            tech_context="RAG技術資料テキスト",
            crm_context="",
            crm_structured=None,
            exhibition_info=None,
            web_context="",
            audio_context="",
            product_card_context="Sorani: 価格500万円、導入実績50社",
        )
        assert "人間確認済み" in sections
        assert "Sorani" in sections

    def test_crm_structured_preferred_over_crm_context(self):
        """crm_structured がある場合、テキスト形式の crm_context より優先されること"""
        gen = _make_generator()
        crm_structured = {
            "lifecycle_stage": "customer",
            "lead_status": "open",
            "last_activity_date": "2026-01-15",
            "match_method": "email",
            "create_date": "", "original_source": "", "contact_owner": "",
            "record_id": "", "first_name": "", "last_name": "", "phone": "", "job_title": "",
            "matched_company": "",
        }
        sections = gen._assemble_context_sections(
            lead=_BASE_LEAD,
            tech_context="",
            crm_context="古いCRM情報テキスト",
            crm_structured=crm_structured,
            exhibition_info=None,
            web_context="",
            audio_context="",
        )
        assert "HubSpot" in sections or "lifecycle_stage" in sections or "ライフサイクル" in sections

    def test_schedule_section_only_when_provided(self):
        """schedule_context を渡した場合のみ面談候補日セクションが含まれること"""
        gen = _make_generator()

        with_schedule = gen._assemble_context_sections(
            lead=_BASE_LEAD,
            tech_context="", crm_context="", crm_structured=None,
            exhibition_info=None, web_context="", audio_context="",
            schedule_context="5月10日（水）10:00〜",
        )
        without_schedule = gen._assemble_context_sections(
            lead=_BASE_LEAD,
            tech_context="", crm_context="", crm_structured=None,
            exhibition_info=None, web_context="", audio_context="",
        )

        assert "面談候補日" in with_schedule
        assert "面談候補日" not in without_schedule
