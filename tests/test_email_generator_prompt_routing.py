from src.email_generator import EmailGenerator, RANK_POLICY

# EmailGenerator.__new__ で LLM init（API key 要求）をスキップ
_gen = EmailGenerator.__new__(EmailGenerator)

_lead = {
    "lead_rank": "A",
    "visitor_name": "テスト太郎",
    "company_name": "テスト株式会社",
    "department": "営業部",
    "job_title": "部長",
    "email": "test@example.com",
    "visit_date": "2026/4/24",
    "interested_products": "ProductA",
    "future_requests": "",
    "memo": "課題について聞いた",
}


class TestPromptRouting:
    def test_with_schedule_contains_schedule_section(self):
        result = _gen._build_human_prompt_with_schedule(
            lead=_lead,
            policy=RANK_POLICY["A"],
            tech_context="",
            crm_context="",
            schedule_context="以下の候補日...\n  - 2026年5月8日(金) 10:00-12:00",
        )
        assert "面談候補日" in result
        assert "候補日提示型" in result
        assert "情報提供型" not in result

    def test_info_only_excludes_schedule_section(self):
        result = _gen._build_human_prompt_info_only(
            lead=_lead,
            policy=RANK_POLICY["C"],
            tech_context="",
            crm_context="",
        )
        assert "情報提供型として扱う" in result
        assert "面談候補日" not in result
        # "候補日提示型として扱う" は with_schedule 専用のヘッダ。info_only には含まれない
        assert "候補日提示型として扱う" not in result
