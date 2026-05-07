from freezegun import freeze_time

from src.cli_runner import validate_candidate_dates

FROZEN = "2026-05-07"


class TestValidateCandidateDates:
    @freeze_time(FROZEN)
    def test_valid_future_dates(self):
        r = validate_candidate_dates(
            "2026/5/8 10:00-12:00, 14:00-17:00\n"
            "2026/5/12 9:00-12:00\n"
            "2026/5/13 13:00-17:00"
        )
        assert r["is_valid"] is True
        assert len(r["parsed"]) == 3
        assert r["errors"] == []

    @freeze_time(FROZEN)
    def test_past_date_invalid(self):
        r = validate_candidate_dates(
            "2026/5/6 10:00-12:00\n"  # 昨日 → NG
            "2026/5/12 9:00-12:00"
        )
        assert r["is_valid"] is False
        assert r["errors"]
        assert r["parsed"] == []

    def test_bad_format_invalid(self):
        r = validate_candidate_dates("5月8日 10時から12時")
        assert r["is_valid"] is False
        assert r["errors"]
        assert r["parsed"] == []

    def test_empty_input_invalid(self):
        r = validate_candidate_dates("  \n  ")
        assert r["is_valid"] is False
        assert r["errors"]
        assert r["parsed"] == []
