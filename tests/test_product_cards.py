"""tests/test_product_cards.py — 製品カード機能のテスト（Phase 5）

対象:
  - src/utils.py:match_product_cards
  - src/cli_runner.py:save_product_knowledge / load_product_knowledge
  - src/email_generator.py:_build_human_prompt_info_only / _build_human_prompt_with_schedule
  - src/agent.py:FollowUpAgent.process_lead の product_card_context 配線

実 API（OpenAI / DDGS / Chroma）は呼ばない。
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import src.cli_runner as cli_runner
from src.cli_runner import load_product_knowledge, save_product_knowledge
from src.email_generator import RANK_POLICY, EmailGenerator
from src.utils import match_product_cards

# ── モジュールレベル: LLM 初期化を回避 ──────────────────────────────
_gen = EmailGenerator.__new__(EmailGenerator)

_lead = {
    "lead_rank": "A",
    "visitor_name": "テスト太郎",
    "company_name": "テスト株式会社",
    "department": "営業部",
    "job_title": "部長",
    "email": "test@example.com",
    "visit_date": "2026/4/24",
    "interested_products": "Sorani",
    "future_requests": "",
    "memo": "",
}


# ══════════════════════════════════════════════════════════════════════
# TestMatchProductCards
# ══════════════════════════════════════════════════════════════════════

class TestMatchProductCards:

    def test_exact_match_multiple(self):
        """カンマ区切りで複数製品を完全一致で引ける"""
        pk = {"Sorani": "Soraniカード", "EdgeGuard": "EdgeGuardカード"}
        result = match_product_cards("Sorani,EdgeGuard", pk)
        assert set(result.keys()) == {"Sorani", "EdgeGuard"}
        assert result["Sorani"] == "Soraniカード"
        assert result["EdgeGuard"] == "EdgeGuardカード"

    def test_partial_match_longer_key(self):
        """カードキーが 'EdgeGuard 異常検知' でも 'EdgeGuard' でマッチする"""
        pk = {"EdgeGuard 異常検知": "EdgeGuardカード本文"}
        result = match_product_cards("EdgeGuard", pk)
        assert "EdgeGuard 異常検知" in result

    def test_no_match(self):
        """辞書に存在しない製品名はマッチしない"""
        pk = {"Sorani": "Soraniカード"}
        result = match_product_cards("UnknownProduct", pk)
        assert result == {}

    def test_empty_interested_products(self):
        """空文字列は {} を返す"""
        pk = {"Sorani": "Soraniカード"}
        assert match_product_cards("", pk) == {}

    def test_none_product_knowledge(self):
        """product_knowledge=None は {} を返す"""
        assert match_product_cards("Sorani", None) == {}

    def test_empty_product_knowledge(self):
        """product_knowledge={} は {} を返す"""
        assert match_product_cards("Sorani", {}) == {}

    def test_multiple_products_all_match(self):
        """3製品すべてマッチする"""
        pk = {"Sorani": "A", "EdgeGuard": "B", "Kioton": "C"}
        result = match_product_cards("Sorani,EdgeGuard,Kioton", pk)
        assert set(result.keys()) == {"Sorani", "EdgeGuard", "Kioton"}

    def test_leading_trailing_spaces_in_csv(self):
        """'Sorani, EdgeGuard' のように前後スペースがあっても正しく分割・マッチ"""
        pk = {"Sorani": "Soraniカード", "EdgeGuard": "EdgeGuardカード"}
        result = match_product_cards(" Sorani , EdgeGuard ", pk)
        assert set(result.keys()) == {"Sorani", "EdgeGuard"}


# ══════════════════════════════════════════════════════════════════════
# TestSaveLoadProductKnowledge
# ══════════════════════════════════════════════════════════════════════

class TestSaveLoadProductKnowledge:

    @pytest.fixture(autouse=True)
    def tmp_product_path(self, tmp_path, monkeypatch):
        """テストごとに一時ディレクトリを使い、PRODUCT_KNOWLEDGE_PATH を書き換える。"""
        path = str(tmp_path / "data" / "product_knowledge.yaml")
        monkeypatch.setattr(cli_runner, "PRODUCT_KNOWLEDGE_PATH", path)
        return path

    def test_roundtrip(self):
        """保存 → 読込で内容が一致する"""
        products = {"Sorani": "Soraniの説明", "EdgeGuard": "EdgeGuardの説明"}
        save_product_knowledge(products)
        loaded = load_product_knowledge()
        assert loaded == products

    def test_load_returns_empty_when_file_missing(self):
        """ファイルが無いとき load → {}"""
        result = load_product_knowledge()
        assert result == {}

    def test_load_broken_yaml_returns_empty(self, tmp_product_path):
        """壊れた YAML でも load → {}（クラッシュしない）"""
        p = Path(tmp_product_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("broken: [yaml: !!!\n  - unclosed", encoding="utf-8")
        result = load_product_knowledge()
        assert result == {}

    def test_multiline_card_uses_literal_block(self, tmp_product_path):
        """複数行カードを保存したとき、ファイルに '|' リテラルブロックと各行が含まれる"""
        products = {"Sorani": "1行目\n2行目\n3行目"}
        save_product_knowledge(products)
        content = Path(tmp_product_path).read_text(encoding="utf-8")
        assert "|" in content
        assert "1行目" in content
        assert "2行目" in content
        assert "3行目" in content


# ══════════════════════════════════════════════════════════════════════
# TestCardInjection
# ══════════════════════════════════════════════════════════════════════

class TestCardInjection:

    def test_info_only_contains_card_section(self):
        """product_card_context を渡すと '人間確認済み' セクションが含まれる"""
        result = _gen._build_human_prompt_info_only(
            lead=_lead,
            policy=RANK_POLICY["A"],
            tech_context="技術資料テキスト",
            crm_context="",
            product_card_context="Soraniの製品カード本文",
        )
        assert "★製品情報（人間確認済み）" in result

    def test_info_only_card_section_before_tech(self):
        """product_card_section が tech_section より前に配置される"""
        result = _gen._build_human_prompt_info_only(
            lead=_lead,
            policy=RANK_POLICY["A"],
            tech_context="技術資料テキスト",
            crm_context="",
            product_card_context="Soraniの製品カード本文",
        )
        card_pos = result.index("★製品情報（人間確認済み）")
        tech_pos = result.index("関心製品の技術資料")
        assert card_pos < tech_pos

    def test_info_only_no_card_context_excludes_section(self):
        """product_card_context='' のとき '人間確認済み' セクションが含まれない"""
        result = _gen._build_human_prompt_info_only(
            lead=_lead,
            policy=RANK_POLICY["C"],
            tech_context="技術資料テキスト",
            crm_context="",
            product_card_context="",
        )
        assert "★製品情報（人間確認済み）" not in result

    def test_with_schedule_contains_card_section(self):
        """_build_human_prompt_with_schedule でも '人間確認済み' セクションが含まれる"""
        result = _gen._build_human_prompt_with_schedule(
            lead=_lead,
            policy=RANK_POLICY["A"],
            tech_context="技術資料テキスト",
            crm_context="",
            schedule_context="2026年5月8日(金) 10:00-12:00",
            product_card_context="Soraniの製品カード本文",
        )
        assert "★製品情報（人間確認済み）" in result

    def test_with_schedule_card_section_before_tech(self):
        """_build_human_prompt_with_schedule でも product_card_section が tech_section より前"""
        result = _gen._build_human_prompt_with_schedule(
            lead=_lead,
            policy=RANK_POLICY["A"],
            tech_context="技術資料テキスト",
            crm_context="",
            schedule_context="2026年5月8日(金) 10:00-12:00",
            product_card_context="Soraniの製品カード本文",
        )
        card_pos = result.index("★製品情報（人間確認済み）")
        tech_pos = result.index("関心製品の技術資料")
        assert card_pos < tech_pos

    def test_with_schedule_no_card_context_excludes_section(self):
        """_build_human_prompt_with_schedule で product_card_context='' のときセクションなし"""
        result = _gen._build_human_prompt_with_schedule(
            lead=_lead,
            policy=RANK_POLICY["A"],
            tech_context="技術資料テキスト",
            crm_context="",
            schedule_context="2026年5月8日(金) 10:00-12:00",
            product_card_context="",
        )
        assert "★製品情報（人間確認済み）" not in result


# ══════════════════════════════════════════════════════════════════════
# TestProcessLeadCardWiring
# ══════════════════════════════════════════════════════════════════════

def _make_agent():
    """テスト用の FollowUpAgent（vectordb・email_gen はモック）"""
    from src.agent import FollowUpAgent

    vectordb = MagicMock()
    vectordb.is_index_built.return_value = False

    email_gen = MagicMock()
    email_gen.generate.return_value = {
        "subject": "テスト件名",
        "body": "テスト本文",
        "cta": "",
    }

    return FollowUpAgent(vectordb_manager=vectordb, email_generator=email_gen)


_LEAD_WIRING = {
    "visitor_name": "配線テスト太郎",
    "company_name": "配線テスト株式会社",
    "email": "wiring@example.com",
    "lead_rank": "B",
    "interested_products": "Sorani",
    "memo": "",
}


class TestProcessLeadCardWiring:

    def test_card_text_passed_to_generate(self):
        """product_knowledge に該当カードがあれば email_gen.generate に product_card_context として渡る"""
        agent = _make_agent()
        pk = {"Sorani": "Soraniカード本文テキスト"}

        with patch("src.email_validator.validate_email") as mock_v:
            mock_v.return_value = MagicMock(passed=True, errors=[], warnings=[])
            agent.process_lead(
                lead=_LEAD_WIRING,
                product_knowledge=pk,
                enable_web_search=False,
                enable_rank_estimation=False,
            )

        call_kwargs = agent.email_gen.generate.call_args[1]
        assert "Soraniカード本文テキスト" in call_kwargs["product_card_context"]

    def test_none_product_knowledge_passes_empty_context(self):
        """product_knowledge=None のとき product_card_context が空文字で渡る"""
        agent = _make_agent()

        with patch("src.email_validator.validate_email") as mock_v:
            mock_v.return_value = MagicMock(passed=True, errors=[], warnings=[])
            agent.process_lead(
                lead=_LEAD_WIRING,
                product_knowledge=None,
                enable_web_search=False,
                enable_rank_estimation=False,
            )

        call_kwargs = agent.email_gen.generate.call_args[1]
        assert call_kwargs["product_card_context"] == ""
