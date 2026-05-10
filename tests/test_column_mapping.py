"""tests/test_column_mapping.py — auto_map_columns / apply_column_mapping の単体テスト"""

import pandas as pd
import pytest

from src.utils import apply_column_mapping, auto_map_columns
from src.config import Config


# ── フィクスチャ ─────────────────────────────────────────────────────────

ALL_FIELD_DEFS = {**Config.REQUIRED_FIELDS, **Config.OPTIONAL_FIELDS}


# ── lead_id 保護テスト ────────────────────────────────────────────────────

class TestLeadIdHandling:

    def test_external_lead_id_preserved(self):
        """外部CSVに独自 lead_id（AT001等）がある場合、元の値が lead_id として保持されること"""
        df = pd.DataFrame({
            "lead_id": ["AT001", "AT002", "AT003"],
            "氏名": ["山田太郎", "鈴木花子", "田中一郎"],
            "会社名": ["A社", "B社", "C社"],
            "メールアドレス": ["a@a.com", "b@b.com", "c@c.com"],
        })
        mapping = auto_map_columns(df.columns.tolist(), ALL_FIELD_DEFS)
        result = apply_column_mapping(df, mapping)

        assert list(result["lead_id"]) == ["AT001", "AT002", "AT003"]
        assert "extra_lead_id" not in result.columns

    def test_l_format_lead_id_unchanged(self):
        """既存の leads.csv（L001 形式）は lead_id がそのまま維持されること"""
        df = pd.DataFrame({
            "lead_id": ["L001", "L002"],
            "氏名": ["山田太郎", "鈴木花子"],
            "会社名": ["A社", "B社"],
            "メールアドレス": ["a@a.com", "b@b.com"],
        })
        mapping = auto_map_columns(df.columns.tolist(), ALL_FIELD_DEFS)
        result = apply_column_mapping(df, mapping)

        assert list(result["lead_id"]) == ["L001", "L002"]

    def test_no_lead_id_column_gets_numbered(self):
        """lead_id カラムが全くない場合は L001, L002... で自動採番されること"""
        df = pd.DataFrame({
            "氏名": ["山田太郎", "鈴木花子", "田中一郎"],
            "会社名": ["A社", "B社", "C社"],
            "メールアドレス": ["a@a.com", "b@b.com", "c@c.com"],
        })
        mapping = auto_map_columns(df.columns.tolist(), ALL_FIELD_DEFS)
        result = apply_column_mapping(df, mapping)

        assert list(result["lead_id"]) == ["L001", "L002", "L003"]


# ── 既存マッピング成功の回帰テスト ──────────────────────────────────────────

class TestMappingRegression:

    def _assert_mapped(self, result_mapping: dict, expected_keys: list):
        """指定フィールドが None でなくマッピングされていることを確認"""
        for key in expected_keys:
            assert result_mapping.get(key) is not None, f"{key} がマッピングされていません"

    def test_required_fields_all_mapped(self):
        """必須3フィールド（visitor_name / company_name / email）が常にマッピングされること"""
        columns = ["氏名", "会社名", "メールアドレス", "評価", "メモ"]
        mapping = auto_map_columns(columns, Config.REQUIRED_FIELDS)
        self._assert_mapped(mapping, ["visitor_name", "company_name", "email"])

    def test_optional_fields_partially_mapped(self):
        """任意フィールドが候補カラム名で正しくマッピングされること"""
        columns = ["評価", "部署", "役職", "来場日", "担当者名", "スキャン時刻"]
        mapping = auto_map_columns(columns, Config.OPTIONAL_FIELDS)
        self._assert_mapped(mapping, ["lead_rank", "department", "job_title",
                                      "visit_date", "rep_name", "scan_time"])

    def test_no_duplicate_assignment(self):
        """1つの CSV カラムが複数フィールドに重複割り当てされないこと"""
        columns = ["氏名", "会社名", "メールアドレス", "評価"]
        mapping = auto_map_columns(columns, ALL_FIELD_DEFS)
        assigned = [v for v in mapping.values() if v is not None]
        assert len(assigned) == len(set(assigned)), "重複割り当てが発生しています"
