"""tests/test_field_labels.py — Config.get_field_label() の単体テスト"""

import pytest

from src.config import Config


class TestGetFieldLabel:

    def test_visitor_name(self):
        assert Config.get_field_label("visitor_name") == "氏名"

    def test_company_name(self):
        assert Config.get_field_label("company_name") == "会社名"

    def test_email(self):
        assert Config.get_field_label("email") == "メールアドレス"

    def test_lead_rank(self):
        assert Config.get_field_label("lead_rank") == "ランク"

    def test_rep_name(self):
        assert Config.get_field_label("rep_name") == "担当営業"

    def test_scan_time(self):
        assert Config.get_field_label("scan_time") == "スキャン時刻"

    def test_fallback_unknown_field(self):
        assert Config.get_field_label("unknown_field") == "unknown_field"

    def test_fallback_extra_column(self):
        assert Config.get_field_label("extra_custom") == "extra_custom"

    def test_all_required_fields_have_label(self):
        """REQUIRED_FIELDS の全フィールドが FIELD_LABELS に登録されていること"""
        for field in Config.REQUIRED_FIELDS:
            assert field in Config.FIELD_LABELS, f"{field} が FIELD_LABELS に未登録"

    def test_all_optional_fields_have_label(self):
        """OPTIONAL_FIELDS の全フィールドが FIELD_LABELS に登録されていること"""
        for field in Config.OPTIONAL_FIELDS:
            assert field in Config.FIELD_LABELS, f"{field} が FIELD_LABELS に未登録"
