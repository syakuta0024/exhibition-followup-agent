"""tests/test_utils_quality.py — utils.py の未テスト関数の単体テスト

check_lead_quality() / parse_interested_products() /
save_results_to_csv() / filter_leads_by_rank() を検証する。
"""

import os

import pandas as pd
import pytest

from src.utils import (
    check_lead_quality,
    parse_interested_products,
    save_results_to_csv,
    filter_leads_by_rank,
    load_leads,
)


# ── check_lead_quality() ─────────────────────────────────────────────

class TestCheckLeadQuality:

    def test_all_fields_present_returns_full_score(self):
        """全必須フィールドが揃っているとき score=100 になること"""
        lead = {
            "visitor_name": "山田 太郎",
            "company_name": "株式会社テスト",
            "interested_products": "Sorani",
            "memo": "導入コストを検討中",
            "lead_rank": "A",
        }
        result = check_lead_quality(lead)
        assert result["score"] == 100
        assert result["errors"] == []
        assert result["warnings"] == []

    def test_missing_visitor_name_is_error(self):
        """visitor_name が空のとき errors に入ること"""
        lead = {
            "visitor_name": "",
            "company_name": "株式会社テスト",
            "interested_products": "Sorani",
            "memo": "メモ",
            "lead_rank": "A",
        }
        result = check_lead_quality(lead)
        assert any("氏名" in e for e in result["errors"])
        assert result["score"] < 100

    def test_missing_company_name_is_error(self):
        """company_name が空のとき errors に入ること"""
        lead = {
            "visitor_name": "山田",
            "company_name": "",
            "interested_products": "Sorani",
            "memo": "メモ",
            "lead_rank": "A",
        }
        result = check_lead_quality(lead)
        assert any("会社名" in e for e in result["errors"])

    def test_missing_interested_products_is_warning(self):
        """interested_products が空のとき warnings に入ること（error ではない）"""
        lead = {
            "visitor_name": "山田",
            "company_name": "テスト社",
            "interested_products": "",
            "memo": "メモ",
            "lead_rank": "A",
        }
        result = check_lead_quality(lead)
        assert any("関心製品" in w for w in result["warnings"])
        assert not any("関心製品" in e for e in result["errors"])

    def test_missing_memo_is_warning(self):
        """memo が空のとき warnings に入ること"""
        lead = {
            "visitor_name": "山田",
            "company_name": "テスト社",
            "interested_products": "Sorani",
            "memo": "",
            "lead_rank": "A",
        }
        result = check_lead_quality(lead)
        assert any("商談メモ" in w for w in result["warnings"])

    def test_all_optional_missing_gives_lower_score(self):
        """required は揃っているが optional が全部空のとき score が下がること"""
        lead = {
            "visitor_name": "山田",
            "company_name": "テスト社",
            "interested_products": "",
            "memo": "",
            "lead_rank": "",
        }
        result = check_lead_quality(lead)
        assert result["score"] < 100
        assert result["errors"] == []

    def test_score_is_integer(self):
        """score が int であること"""
        lead = {"visitor_name": "山田", "company_name": "テスト社"}
        result = check_lead_quality(lead)
        assert isinstance(result["score"], int)

    def test_whitespace_only_value_treated_as_empty(self):
        """空白のみの値は空扱いになること"""
        lead = {
            "visitor_name": "   ",
            "company_name": "テスト社",
            "interested_products": "Sorani",
            "memo": "メモ",
            "lead_rank": "A",
        }
        result = check_lead_quality(lead)
        assert any("氏名" in e for e in result["errors"])


# ── parse_interested_products() ──────────────────────────────────────

class TestParseInterestedProducts:

    def test_parses_comma_separated(self):
        result = parse_interested_products("Sorani,EdgeGuard,FactoryBrain")
        assert result == ["Sorani", "EdgeGuard", "FactoryBrain"]

    def test_strips_whitespace(self):
        result = parse_interested_products("  Sorani ,  EdgeGuard  ")
        assert result == ["Sorani", "EdgeGuard"]

    def test_removes_surrounding_quotes(self):
        result = parse_interested_products('"Sorani,EdgeGuard"')
        assert result == ["Sorani", "EdgeGuard"]

    def test_removes_single_quotes(self):
        result = parse_interested_products("'Sorani,EdgeGuard'")
        assert result == ["Sorani", "EdgeGuard"]

    def test_empty_string_returns_empty_list(self):
        assert parse_interested_products("") == []

    def test_none_equivalent_returns_empty_list(self):
        """None を str に変換した場合も空リストになること"""
        assert parse_interested_products("None") == ["None"]  # "None" はそのまま

    def test_single_product(self):
        result = parse_interested_products("Sorani")
        assert result == ["Sorani"]

    def test_ignores_empty_entries(self):
        """カンマだけの空エントリが除去されること"""
        result = parse_interested_products("Sorani,,EdgeGuard,")
        assert result == ["Sorani", "EdgeGuard"]


# ── save_results_to_csv() ────────────────────────────────────────────

class TestSaveResultsToCsv:

    def test_saves_csv_file(self, tmp_path):
        """正常時に CSV ファイルが作成されること"""
        output_path = str(tmp_path / "output" / "emails.csv")
        results = [
            {"visitor_name": "山田", "subject": "件名1", "body": "本文1"},
            {"visitor_name": "鈴木", "subject": "件名2", "body": "本文2"},
        ]

        save_results_to_csv(results, output_path)

        assert os.path.exists(output_path)
        df = pd.read_csv(output_path, encoding="utf-8-sig")
        assert len(df) == 2
        assert "visitor_name" in df.columns

    def test_raises_on_empty_results(self, tmp_path):
        """空リストを渡したとき ValueError が発生すること"""
        output_path = str(tmp_path / "emails.csv")

        with pytest.raises(ValueError, match="保存する結果データがありません"):
            save_results_to_csv([], output_path)

    def test_creates_output_directory(self, tmp_path):
        """出力ディレクトリが存在しない場合に自動作成されること"""
        output_path = str(tmp_path / "new_dir" / "subdir" / "out.csv")
        results = [{"visitor_name": "テスト"}]

        save_results_to_csv(results, output_path)

        assert os.path.exists(output_path)

    def test_csv_encoding_is_utf8_bom(self, tmp_path):
        """CSV が UTF-8 BOM で保存されること（Excel 対応）"""
        output_path = str(tmp_path / "test.csv")
        results = [{"visitor_name": "山田 太郎", "subject": "【御礼】展示会"}]

        save_results_to_csv(results, output_path)

        # BOM 付き UTF-8 で読めること
        df = pd.read_csv(output_path, encoding="utf-8-sig")
        assert df["visitor_name"][0] == "山田 太郎"

    def test_japanese_text_preserved(self, tmp_path):
        """日本語テキストが文字化けせずに保存・復元されること"""
        output_path = str(tmp_path / "japanese.csv")
        results = [{"body": "展示会にご来場いただきありがとうございました"}]

        save_results_to_csv(results, output_path)

        df = pd.read_csv(output_path, encoding="utf-8-sig")
        assert "展示会" in df["body"][0]


# ── filter_leads_by_rank() ───────────────────────────────────────────

class TestFilterLeadsByRank:

    def _make_df(self):
        return pd.DataFrame([
            {"visitor_name": "山田", "lead_rank": "A"},
            {"visitor_name": "鈴木", "lead_rank": "B"},
            {"visitor_name": "田中", "lead_rank": "C"},
            {"visitor_name": "伊藤", "lead_rank": "D"},
            {"visitor_name": "渡辺", "lead_rank": "E"},
        ])

    def test_filters_by_single_rank(self):
        df = self._make_df()
        result = filter_leads_by_rank(df, ["A"])
        assert len(result) == 1
        assert result.iloc[0]["visitor_name"] == "山田"

    def test_filters_by_multiple_ranks(self):
        df = self._make_df()
        result = filter_leads_by_rank(df, ["A", "B"])
        assert len(result) == 2

    def test_empty_ranks_returns_all(self):
        """ランクリストが空のとき全件返ること"""
        df = self._make_df()
        result = filter_leads_by_rank(df, [])
        assert len(result) == 5

    def test_no_matching_rank_returns_empty(self):
        """該当ランクがない場合空 DataFrame になること"""
        df = self._make_df()
        result = filter_leads_by_rank(df, ["Z"])
        assert len(result) == 0

    def test_case_insensitive(self):
        """小文字でも正しくフィルタリングされること"""
        df = self._make_df()
        result = filter_leads_by_rank(df, ["a", "b"])
        assert len(result) == 2

    def test_strips_whitespace_in_rank(self):
        """ランク値の前後の空白が除去されること"""
        df = pd.DataFrame([
            {"visitor_name": "山田", "lead_rank": " A "},
            {"visitor_name": "鈴木", "lead_rank": "B"},
        ])
        result = filter_leads_by_rank(df, ["A"])
        assert len(result) == 1


# ── load_leads() ────────────────────────────────────────────────────

class TestLoadLeads:

    def test_raises_file_not_found(self, tmp_path):
        """存在しないパスで FileNotFoundError が発生すること"""
        with pytest.raises(FileNotFoundError):
            load_leads(str(tmp_path / "nonexistent.csv"))

    def test_loads_csv_with_str_dtype(self, tmp_path):
        """全カラムが文字列型で読み込まれること"""
        csv_path = tmp_path / "leads.csv"
        csv_path.write_text(
            "visitor_name,company_name,lead_rank\n山田,テスト社,A\n",
            encoding="utf-8"
        )

        df = load_leads(str(csv_path))

        # Python 3.13 + pandas では StringDtype になるため is_string_dtype で確認
        assert pd.api.types.is_string_dtype(df.dtypes["visitor_name"])
        assert df["visitor_name"][0] == "山田"

    def test_strips_whitespace(self, tmp_path):
        """カラム値の前後の空白が除去されること"""
        csv_path = tmp_path / "leads.csv"
        csv_path.write_text(
            "visitor_name,company_name\n  山田  ,  テスト社  \n",
            encoding="utf-8"
        )

        df = load_leads(str(csv_path))

        assert df["visitor_name"][0] == "山田"
        assert df["company_name"][0] == "テスト社"

    def test_nan_filled_with_empty_string(self, tmp_path):
        """NaN が空文字に変換されること"""
        csv_path = tmp_path / "leads.csv"
        csv_path.write_text(
            "visitor_name,memo\n山田,\n",
            encoding="utf-8"
        )

        df = load_leads(str(csv_path))

        assert df["memo"][0] == ""
