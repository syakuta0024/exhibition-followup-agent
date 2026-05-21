"""tests/test_rank_normalizer.py — ランク値正規化・マッピング推定の単体テスト"""

import json
from unittest.mock import MagicMock

import pytest

from src.utils import extract_unique_rank_values, normalize_rank_values
from src.rank_estimator import infer_rank_mapping_with_llm


# ── extract_unique_rank_values ────────────────────────────────────────────────

class TestExtractUniqueRankValues:

    def test_rx_japan_format(self):
        """RX Japan 形式の値からユニーク値が正しく抽出されること"""
        leads = [
            {"lead_rank": "B：担当者フォロー"},
            {"lead_rank": "A：決裁者商談"},
            {"lead_rank": "B：担当者フォロー"},  # 重複
            {"lead_rank": "C：情報収集"},
        ]
        result = extract_unique_rank_values(leads, "lead_rank")
        assert result == ["B：担当者フォロー", "A：決裁者商談", "C：情報収集"]

    def test_empty_and_none_excluded(self):
        """空文字・None・空白のみはユニーク値から除外されること"""
        leads = [
            {"lead_rank": "A"},
            {"lead_rank": ""},
            {"lead_rank": None},
            {"lead_rank": "  "},
            {"lead_rank": "B"},
        ]
        result = extract_unique_rank_values(leads, "lead_rank")
        assert result == ["A", "B"]

    def test_missing_field_excluded(self):
        """rank_field が存在しないリードはスキップされること"""
        leads = [{"other_field": "x"}, {"lead_rank": "C"}]
        result = extract_unique_rank_values(leads, "lead_rank")
        assert result == ["C"]

    def test_insertion_order_preserved(self):
        """出現順が維持されること"""
        leads = [
            {"lead_rank": "C"},
            {"lead_rank": "A"},
            {"lead_rank": "B"},
        ]
        result = extract_unique_rank_values(leads, "lead_rank")
        assert result == ["C", "A", "B"]


# ── normalize_rank_values ─────────────────────────────────────────────────────

class TestNormalizeRankValues:

    def test_rx_japan_format_normalized(self):
        """ケース1: RX Japan 形式が正規化されること"""
        leads = [
            {"lead_rank": "B：担当者フォロー", "name": "田中"},
            {"lead_rank": "A：決裁者商談",    "name": "鈴木"},
        ]
        mapping = {"B：担当者フォロー": "B", "A：決裁者商談": "A"}
        result = normalize_rank_values(leads, "lead_rank", mapping)
        assert result[0]["lead_rank"] == "B"
        assert result[1]["lead_rank"] == "A"

    def test_unknown_key_passes_through(self):
        """ケース4: mapping にないキーはそのまま通ること（クラッシュしない）"""
        leads = [{"lead_rank": "不明"}, {"lead_rank": "B：担当者フォロー"}]
        mapping = {"B：担当者フォロー": "B"}
        result = normalize_rank_values(leads, "lead_rank", mapping)
        assert result[0]["lead_rank"] == "不明"
        assert result[1]["lead_rank"] == "B"

    def test_does_not_mutate_original(self):
        """ケース5: 元のリストを変更しないこと（副作用なし）"""
        original = [{"lead_rank": "B：担当者フォロー", "name": "田中"}]
        mapping = {"B：担当者フォロー": "B"}
        result = normalize_rank_values(original, "lead_rank", mapping)

        # 別オブジェクトであること
        assert result is not original
        assert result[0] is not original[0]
        # 元のリストが変更されていないこと
        assert original[0]["lead_rank"] == "B：担当者フォロー"
        # 戻り値には正規化値が入ること
        assert result[0]["lead_rank"] == "B"

    def test_empty_mapping_passthrough(self):
        """空のマッピングではすべての値がそのまま通ること"""
        leads = [{"lead_rank": "Hot"}, {"lead_rank": "Cold"}]
        result = normalize_rank_values(leads, "lead_rank", {})
        assert result[0]["lead_rank"] == "Hot"
        assert result[1]["lead_rank"] == "Cold"

    def test_other_fields_preserved(self):
        """lead_rank 以外のフィールドが保持されること"""
        leads = [{"lead_rank": "B：担当者フォロー", "name": "田中", "email": "a@a.com"}]
        mapping = {"B：担当者フォロー": "B"}
        result = normalize_rank_values(leads, "lead_rank", mapping)
        assert result[0]["name"] == "田中"
        assert result[0]["email"] == "a@a.com"


# ── infer_rank_mapping_with_llm ───────────────────────────────────────────────

def _make_mock_client(response_json: dict) -> MagicMock:
    """LangChain ChatOpenAI のモッククライアントを生成する"""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps(response_json, ensure_ascii=False)
    mock_client.invoke.return_value = mock_response
    return mock_client


class TestInferRankMappingWithLlm:

    def test_rx_japan_format(self):
        """ケース1: RX Japan 形式が正しくマッピングされること"""
        llm_output = {"B：担当者フォロー": "B", "A：決裁者商談": "A"}
        client = _make_mock_client(llm_output)

        result = infer_rank_mapping_with_llm(["B：担当者フォロー", "A：決裁者商談"], client)

        assert result == {"B：担当者フォロー": "B", "A：決裁者商談": "A"}
        client.invoke.assert_called_once()

    def test_star_rating_format(self):
        """ケース2: 星評価形式が正しくマッピングされること"""
        llm_output = {"★5": "A", "★3": "C", "★1": "E"}
        client = _make_mock_client(llm_output)

        result = infer_rank_mapping_with_llm(["★5", "★3", "★1"], client)

        assert result == {"★5": "A", "★3": "C", "★1": "E"}

    def test_already_clean_skips_llm(self):
        """ケース3: すでにクリーンな値（A〜E）はそのまま返されること（run_rank_mapping 側でチェックするが念のため）"""
        llm_output = {"A": "A", "B": "B", "C": "C"}
        client = _make_mock_client(llm_output)

        result = infer_rank_mapping_with_llm(["A", "B", "C"], client)

        assert result == {"A": "A", "B": "B", "C": "C"}

    def test_null_value_excluded(self):
        """LLM が null を返した値はマッピングから除外されること"""
        llm_output = {"Hot": "A", "Unknown": None}
        client = _make_mock_client(llm_output)

        result = infer_rank_mapping_with_llm(["Hot", "Unknown"], client)

        assert "Hot" in result
        assert "Unknown" not in result

    def test_invalid_rank_value_excluded(self):
        """A〜E 以外の値（例: 'X'）はマッピングから除外されること"""
        llm_output = {"Hot": "A", "Warm": "X"}
        client = _make_mock_client(llm_output)

        result = infer_rank_mapping_with_llm(["Hot", "Warm"], client)

        assert result == {"Hot": "A"}

    def test_lowercase_rank_normalized_to_upper(self):
        """LLM が小文字（'a', 'b'）を返した場合も大文字に正規化されること"""
        llm_output = {"Hot": "a", "Cold": "e"}
        client = _make_mock_client(llm_output)

        result = infer_rank_mapping_with_llm(["Hot", "Cold"], client)

        assert result == {"Hot": "A", "Cold": "E"}

    def test_llm_error_returns_empty_dict(self):
        """LLM 呼び出しが例外を投げた場合、空の dict を返すこと（クラッシュしない）"""
        client = MagicMock()
        client.invoke.side_effect = Exception("API error")

        result = infer_rank_mapping_with_llm(["Hot"], client)

        assert result == {}

    def test_json_parse_error_returns_empty_dict(self):
        """LLM が不正な JSON を返した場合、空の dict を返すこと"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "これは JSON ではありません"
        mock_client.invoke.return_value = mock_response

        result = infer_rank_mapping_with_llm(["Hot"], mock_client)

        assert result == {}

    def test_markdown_code_block_stripped(self):
        """LLM がコードブロック（```json...```）で囲んで返した場合もパースできること"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '```json\n{"Hot": "A"}\n```'
        mock_client.invoke.return_value = mock_response

        result = infer_rank_mapping_with_llm(["Hot"], mock_client)

        assert result == {"Hot": "A"}

    def test_empty_unique_values_returns_empty_dict(self):
        """空リストを渡したとき、LLM を呼ばずに空の dict を返すこと"""
        client = MagicMock()

        result = infer_rank_mapping_with_llm([], client)

        assert result == {}
        client.invoke.assert_not_called()
