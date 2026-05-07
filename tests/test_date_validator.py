import pytest
from freezegun import freeze_time

from src.date_validator import (
    format_for_display,
    format_for_llm_prompt,
    parse_candidate_dates,
    validate_dates,
)

FROZEN = "2026-05-07"  # 木曜日を基準日に固定


class TestParseCandidateDates:
    def test_single_slot(self):
        r = parse_candidate_dates("2026/5/8 10:00-12:00")
        assert r == [{"date": "2026/5/8", "time_slots": ["10:00-12:00"]}]

    def test_multiple_slots(self):
        r = parse_candidate_dates("2026/5/8 10:00-12:00, 14:00-17:00")
        assert r == [{"date": "2026/5/8", "time_slots": ["10:00-12:00", "14:00-17:00"]}]

    def test_multiple_lines(self):
        r = parse_candidate_dates("2026/5/8 10:00-12:00\n2026/5/12 9:00-12:00\n2026/5/13 13:00-17:00")
        assert len(r) == 3
        assert [x["date"] for x in r] == ["2026/5/8", "2026/5/12", "2026/5/13"]

    def test_invalid_date_raises(self):
        with pytest.raises(ValueError):
            parse_candidate_dates("5月8日 10:00-12:00")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parse_candidate_dates("   \n  ")


class TestValidateDates:
    @freeze_time(FROZEN)
    def test_future_single_slot(self):
        r = validate_dates([
            {"date": "2026/5/8",  "time_slots": ["10:00-12:00"]},
            {"date": "2026/5/12", "time_slots": ["9:00-12:00"]},
            {"date": "2026/5/13", "time_slots": ["13:00-17:00"]},
        ])
        assert r.is_valid and r.errors == []

    @freeze_time(FROZEN)
    def test_future_multiple_slots(self):
        r = validate_dates([
            {"date": "2026/5/8",  "time_slots": ["10:00-12:00", "14:00-17:00"]},
            {"date": "2026/5/12", "time_slots": ["9:00-12:00",  "15:00-18:00"]},
        ])
        assert r.is_valid

    @freeze_time(FROZEN)
    def test_today_is_allowed(self):
        r = validate_dates([{"date": "2026/5/7", "time_slots": ["10:00-12:00"]}])
        assert r.is_valid

    @freeze_time(FROZEN)
    def test_past_date_invalid(self):
        r = validate_dates([
            {"date": "2026/5/6",  "time_slots": ["10:00-12:00"]},  # 昨日 → NG
            {"date": "2026/5/10", "time_slots": ["9:00-12:00"]},   # 未来 → OK
        ])
        assert not r.is_valid
        assert any("過去日" in e for e in r.errors)
        assert len([e for e in r.errors if "過去日" in e]) == 1  # 1件だけ

    def test_bad_time_format(self):
        r = validate_dates([{"date": "2026/6/1", "time_slots": ["10時から12時"]}])
        assert not r.is_valid
        assert any("フォーマット" in e for e in r.errors)

    def test_start_after_end(self):
        r = validate_dates([{"date": "2026/6/1", "time_slots": ["12:00-10:00"]}])
        assert not r.is_valid
        assert any("終了時刻" in e for e in r.errors)

    def test_start_equals_end(self):
        r = validate_dates([{"date": "2026/6/1", "time_slots": ["10:00-10:00"]}])
        assert not r.is_valid


class TestFormatForDisplay:
    @pytest.mark.parametrize("date_str,weekday", [
        ("2026/5/4",  "月"),
        ("2026/5/5",  "火"),
        ("2026/5/6",  "水"),
        ("2026/5/7",  "木"),
        ("2026/5/8",  "金"),
        ("2026/5/9",  "土"),
        ("2026/5/10", "日"),
    ])
    def test_all_weekdays(self, date_str, weekday):
        r = format_for_display([{"date": date_str, "time_slots": []}])
        assert f"({weekday})" in r[0]

    def test_friday_single_slot(self):
        r = format_for_display([{"date": "2026/5/8", "time_slots": ["10:00-12:00"]}])
        assert r == ["2026年5月8日(金) 10:00-12:00"]

    def test_multiple_slots_joined(self):
        r = format_for_display([{"date": "2026/5/8", "time_slots": ["10:00-12:00", "14:00-17:00"]}])
        assert r == ["2026年5月8日(金) 10:00-12:00 / 14:00-17:00"]

    def test_month_boundary(self):
        r1 = format_for_display([{"date": "2026/5/31", "time_slots": ["10:00-12:00"]}])
        r2 = format_for_display([{"date": "2026/6/1",  "time_slots": ["10:00-12:00"]}])
        assert "2026年5月31日" in r1[0]
        assert "2026年6月1日" in r2[0]

    def test_year_boundary(self):
        r1 = format_for_display([{"date": "2026/12/31", "time_slots": ["10:00-12:00"]}])
        r2 = format_for_display([{"date": "2027/1/1",  "time_slots": ["10:00-12:00"]}])
        assert "2026年12月31日" in r1[0]
        assert "2027年1月1日" in r2[0]


class TestFormatForLlmPrompt:
    def test_header_and_footer(self):
        r = format_for_llm_prompt([
            {"date": "2026/5/8",  "time_slots": ["10:00-12:00", "14:00-17:00"]},
            {"date": "2026/5/12", "time_slots": ["9:00-12:00"]},
            {"date": "2026/5/13", "time_slots": ["13:00-17:00"]},
        ])
        assert "以下の候補日のいずれかをメール本文に組み込んでください" in r
        assert "これ以外の日付・曜日を本文に書かないでください" in r

    def test_all_dates_present(self):
        r = format_for_llm_prompt([
            {"date": "2026/5/8",  "time_slots": ["10:00-12:00"]},
            {"date": "2026/5/12", "time_slots": ["9:00-12:00"]},
        ])
        assert "5月8日" in r
        assert "5月12日" in r
