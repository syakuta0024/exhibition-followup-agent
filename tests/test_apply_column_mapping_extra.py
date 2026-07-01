"""tests/test_apply_column_mapping_extra.py — apply_column_mapping() の追加テスト

test_column_mapping.py でカバーされていない以下を検証する:
  - 未マッピング列への extra_ プレフィックス付与
  - 標準フィールドの空文字補完
  - lead_id 列が全て空欄のときの自動採番
"""

import pandas as pd
import pytest

from src.config import Config
from src.utils import apply_column_mapping, auto_map_columns

ALL_FIELDS = {**Config.REQUIRED_FIELDS, **Config.OPTIONAL_FIELDS}


# ── extra_ プレフィックス ────────────────────────────────────────────────────

class TestExtraPrefix:

    def test_unmapped_columns_get_extra_prefix(self):
        """マッピング対象外の列に extra_ プレフィックスが付くこと"""
        df = pd.DataFrame({
            "氏名": ["山田太郎"],
            "会社名": ["ABC社"],
            "メールアドレス": ["a@a.com"],
            "アンケート回答1": ["興味あり"],
            "自由記入": ["詳細コメント"],
        })
        mapping = auto_map_columns(df.columns.tolist(), ALL_FIELDS)
        result = apply_column_mapping(df, mapping)

        assert "extra_アンケート回答1" in result.columns
        assert "extra_自由記入" in result.columns
        assert result.iloc[0]["extra_アンケート回答1"] == "興味あり"
        assert result.iloc[0]["extra_自由記入"] == "詳細コメント"

    def test_mapped_columns_not_prefixed(self):
        """マッピング済みの標準フィールド名が extra_ 付きにならないこと"""
        df = pd.DataFrame({
            "氏名": ["鈴木"],
            "会社名": ["XYZ社"],
            "メールアドレス": ["s@xyz.com"],
        })
        mapping = auto_map_columns(df.columns.tolist(), ALL_FIELDS)
        result = apply_column_mapping(df, mapping)

        assert "extra_氏名" not in result.columns
        assert "extra_会社名" not in result.columns
        assert "extra_メールアドレス" not in result.columns
        assert "visitor_name" in result.columns
        assert "company_name" in result.columns
        assert "email" in result.columns

    def test_all_extra_when_no_columns_match(self):
        """どの標準フィールドにもマッチしない列は全て extra_ になること"""
        df = pd.DataFrame({
            "Q1_解答": ["はい"],
            "Q2_解答": ["製造業"],
        })
        mapping = auto_map_columns(df.columns.tolist(), ALL_FIELDS)
        result = apply_column_mapping(df, mapping)

        assert "extra_Q1_解答" in result.columns
        assert "extra_Q2_解答" in result.columns

    def test_extra_columns_coexist_with_mapped_columns(self):
        """extra_ 列とマッピング済み標準フィールドが共存すること"""
        df = pd.DataFrame({
            "氏名": ["田中"],
            "会社名": ["DEF社"],
            "メールアドレス": ["t@def.com"],
            "独自質問A": ["回答A"],
        })
        mapping = auto_map_columns(df.columns.tolist(), ALL_FIELDS)
        result = apply_column_mapping(df, mapping)

        assert "visitor_name" in result.columns
        assert "extra_独自質問A" in result.columns
        assert result.iloc[0]["visitor_name"] == "田中"
        assert result.iloc[0]["extra_独自質問A"] == "回答A"


# ── 標準フィールド補完 ────────────────────────────────────────────────────────

class TestStandardFieldCompletion:

    def test_missing_optional_fields_added_as_empty(self):
        """CSV にない任意フィールドが空文字で補完されること"""
        df = pd.DataFrame({
            "氏名": ["山田太郎"],
            "会社名": ["ABC社"],
            "メールアドレス": ["a@a.com"],
        })
        mapping = auto_map_columns(df.columns.tolist(), ALL_FIELDS)
        result = apply_column_mapping(df, mapping)

        # lead_id 以外の任意フィールドが空文字で補完されていること
        for field_key in Config.OPTIONAL_FIELDS:
            assert field_key in result.columns, f"{field_key} が補完されていない"
            if field_key != "lead_id":
                assert result.iloc[0][field_key] == "", f"{field_key} が空文字でない"

    def test_required_fields_exist_after_mapping(self):
        """必須フィールドがマッピング後も全て存在すること"""
        df = pd.DataFrame({
            "氏名": ["山田"],
            "会社名": ["ABC社"],
            "メールアドレス": ["a@a.com"],
        })
        mapping = auto_map_columns(df.columns.tolist(), ALL_FIELDS)
        result = apply_column_mapping(df, mapping)

        for field_key in Config.REQUIRED_FIELDS:
            assert field_key in result.columns, f"必須フィールド {field_key} が見つからない"


# ── lead_id 全空欄の自動採番 ──────────────────────────────────────────────────

class TestLeadIdAutoNumber:

    def test_lead_id_all_empty_gets_auto_numbered(self):
        """lead_id 列が存在するが全て空欄のとき L001, L002… に採番されること"""
        df = pd.DataFrame({
            "lead_id": ["", ""],
            "氏名": ["山田", "鈴木"],
            "会社名": ["A社", "B社"],
            "メールアドレス": ["a@a.com", "b@b.com"],
        })
        mapping = auto_map_columns(df.columns.tolist(), ALL_FIELDS)
        result = apply_column_mapping(df, mapping)

        assert list(result["lead_id"]) == ["L001", "L002"]

    def test_lead_id_all_empty_three_rows(self):
        """3行すべて空欄でも正しく L001〜L003 が採番されること"""
        df = pd.DataFrame({
            "lead_id": ["", "", ""],
            "氏名": ["A", "B", "C"],
            "会社名": ["X社", "Y社", "Z社"],
            "メールアドレス": ["a@x.com", "b@y.com", "c@z.com"],
        })
        mapping = auto_map_columns(df.columns.tolist(), ALL_FIELDS)
        result = apply_column_mapping(df, mapping)

        assert list(result["lead_id"]) == ["L001", "L002", "L003"]
