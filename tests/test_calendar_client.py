"""
src/calendar_client.py のテスト。

Google Calendar API はモック化しているため APIコストゼロで実行できる。
基準日: 2026-05-18（月曜日）
  - 2営業日後 = 2026-05-20（水曜日）
"""

from datetime import datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest
from freezegun import freeze_time

from src.calendar_client import fetch_free_slots, format_slots_for_email

_TZ = ZoneInfo("Asia/Tokyo")
FROZEN = "2026-05-18"  # 月曜日


def _iso(dt_str: str) -> str:
    """日本時間の datetime 文字列を ISO 8601 フォーマットに変換する。"""
    return datetime.fromisoformat(dt_str).replace(tzinfo=_TZ).isoformat()


def _make_service(busy_periods: list[dict]) -> MagicMock:
    """モックの Calendar サービスを返す。"""
    service = MagicMock()
    service.freebusy().query().execute.return_value = {
        "calendars": {
            "primary": {
                "busy": busy_periods,
            }
        }
    }
    return service


# ---------------------------------------------------------------------------
# ケース1: 平日の空き枠が正しく返る
# ---------------------------------------------------------------------------
class TestFetchFreeSlots:
    @freeze_time(FROZEN)
    def test_weekday_slots_returned(self):
        """予定なし → 平日の 09:00〜 から順に max_slots 件取得できる。"""
        service = _make_service([])
        slots = fetch_free_slots(service, days_ahead=14, duration_minutes=60, working_hours=(9, 18), max_slots=3)

        assert len(slots) == 3
        # 最初のスロットは 2営業日後 (2026-05-20 水) の 09:00
        assert slots[0]["date"] == "2026-05-20"
        assert slots[0]["weekday"] == "水"
        assert slots[0]["start"] == "09:00"
        assert slots[0]["end"] == "10:00"
        assert "display" in slots[0]
        assert "5月20日（水）09:00〜10:00" == slots[0]["display"]

    @freeze_time(FROZEN)
    def test_slot_fields_complete(self):
        """各スロットに必要なフィールドがすべて含まれる。"""
        service = _make_service([])
        slots = fetch_free_slots(service, days_ahead=14, duration_minutes=60, max_slots=1)

        assert set(slots[0].keys()) == {"date", "weekday", "start", "end", "display"}

    # ---------------------------------------------------------------------------
    # ケース2: 土日が除外される
    # ---------------------------------------------------------------------------
    @freeze_time("2026-05-21")  # 木曜 → 2営業日後は月曜 (2026-05-25)
    def test_weekends_excluded(self):
        """2営業日後が土日を跨ぐ場合でも土日がスキップされる。"""
        service = _make_service([])
        # 基準日: 木曜 2026-05-21 → 2営業日後: 月曜 2026-05-25
        slots = fetch_free_slots(service, days_ahead=10, duration_minutes=60, max_slots=5)

        dates = [s["date"] for s in slots]
        for d in dates:
            weekday = datetime.strptime(d, "%Y-%m-%d").weekday()
            assert weekday < 5, f"{d} は土日（weekday={weekday}）"

    @freeze_time("2026-05-22")  # 金曜 → 2営業日後は火曜 (2026-05-26)
    def test_friday_skips_weekend(self):
        """金曜日基準: 2営業日後は翌週火曜 (土日を除外)。"""
        service = _make_service([])
        slots = fetch_free_slots(service, days_ahead=10, duration_minutes=60, max_slots=1)

        assert len(slots) >= 1
        first = slots[0]["date"]
        weekday = datetime.strptime(first, "%Y-%m-%d").weekday()
        assert weekday < 5

    # ---------------------------------------------------------------------------
    # ケース3: working_hours 外の時間が除外される
    # ---------------------------------------------------------------------------
    @freeze_time(FROZEN)
    def test_working_hours_boundary(self):
        """working_hours=(10, 12) なら 10:00〜11:00, 11:00〜12:00 のみ。"""
        service = _make_service([])
        slots = fetch_free_slots(
            service, days_ahead=14, duration_minutes=60,
            working_hours=(10, 12), max_slots=10
        )
        # 1日に取れるスロットは 2件まで（10:00-11:00 / 11:00-12:00）
        # 5日分 × 2件 < 10件 なので上限に達しない可能性あり
        for slot in slots:
            assert slot["start"] >= "10:00"
            assert slot["end"] <= "12:00"

    @freeze_time(FROZEN)
    def test_no_slot_beyond_working_hours(self):
        """duration_minutes が working_hours をはみ出すスロットは返らない。"""
        service = _make_service([])
        # 9〜10 の 1時間枠のみ: 9:00-10:00 は OK、10:00-11:00 は NG (end > 10)
        slots = fetch_free_slots(
            service, days_ahead=14, duration_minutes=60,
            working_hours=(9, 10), max_slots=10
        )
        for slot in slots:
            end_h = int(slot["end"].split(":")[0])
            end_m = int(slot["end"].split(":")[1])
            assert end_h < 10 or (end_h == 10 and end_m == 0)

    # ---------------------------------------------------------------------------
    # ケース4: 既存予定と重複する時間が除外される
    # ---------------------------------------------------------------------------
    @freeze_time(FROZEN)
    def test_busy_slot_excluded(self):
        """09:00-10:00 に予定がある場合、その枠はスキップされる。"""
        busy = [{"start": _iso("2026-05-20 09:00"), "end": _iso("2026-05-20 10:00")}]
        service = _make_service(busy)
        slots = fetch_free_slots(service, days_ahead=5, duration_minutes=60, max_slots=3)

        starts = [s["start"] for s in slots if s["date"] == "2026-05-20"]
        assert "09:00" not in starts

    @freeze_time(FROZEN)
    def test_partial_overlap_excluded(self):
        """09:30-10:30 の予定は 09:00-10:00 と 10:00-11:00 の両方を除外する。"""
        busy = [{"start": _iso("2026-05-20 09:30"), "end": _iso("2026-05-20 10:30")}]
        service = _make_service(busy)
        slots = fetch_free_slots(service, days_ahead=3, duration_minutes=60, max_slots=10)

        day_slots = [s for s in slots if s["date"] == "2026-05-20"]
        starts = [s["start"] for s in day_slots]
        assert "09:00" not in starts
        assert "10:00" not in starts
        # 11:00 以降は空き
        assert "11:00" in starts

    # ---------------------------------------------------------------------------
    # ケース5: 空き枠が0件のとき空リストを返す（クラッシュしない）
    # ---------------------------------------------------------------------------
    @freeze_time(FROZEN)
    def test_returns_empty_list_when_no_slots(self):
        """全時間が予定で埋まっている場合、空リストを返してクラッシュしない。"""
        # 14日間の全時間を busy にする
        busy = [{"start": _iso("2026-05-20 00:00"), "end": _iso("2026-06-01 23:59")}]
        service = _make_service(busy)
        slots = fetch_free_slots(service, days_ahead=14, duration_minutes=60, max_slots=5)

        assert slots == []

    @freeze_time(FROZEN)
    def test_returns_empty_when_days_ahead_zero(self):
        """days_ahead=0 で min_date > max_date になる場合、空リストを返す。"""
        service = _make_service([])
        # days_ahead=0 → max_date = today < min_date(2営業日後) → 空
        slots = fetch_free_slots(service, days_ahead=0, duration_minutes=60, max_slots=5)
        assert slots == []

    # ---------------------------------------------------------------------------
    # ケース6: format_slots_for_email が正しい文字列を返す
    # ---------------------------------------------------------------------------
    class TestFormatSlotsForEmail:
        def test_single_slot(self):
            slots = [{"date": "2026-05-20", "weekday": "水", "start": "14:00", "end": "15:00", "display": "..."}]
            result = format_slots_for_email(slots)
            assert result == "5月20日（水）14時〜はいかがでしょうか"

        def test_multiple_slots(self):
            slots = [
                {"date": "2026-05-20", "weekday": "水", "start": "14:00", "end": "15:00", "display": "..."},
                {"date": "2026-05-21", "weekday": "木", "start": "10:00", "end": "11:00", "display": "..."},
                {"date": "2026-05-22", "weekday": "金", "start": "13:00", "end": "14:00", "display": "..."},
            ]
            result = format_slots_for_email(slots)
            assert "5月20日（水）14時〜" in result
            assert "5月21日（木）10時〜" in result
            assert "5月22日（金）13時〜" in result
            assert result.endswith("はいかがでしょうか")

        def test_empty_slots(self):
            assert format_slots_for_email([]) == ""

        def test_hour_leading_zero_stripped(self):
            """09:00 は「9時〜」と表示される（先頭の0が除かれる）。"""
            slots = [{"date": "2026-05-20", "weekday": "水", "start": "09:00", "end": "10:00", "display": "..."}]
            result = format_slots_for_email(slots)
            assert "9時〜" in result
            assert "09時〜" not in result

    # ---------------------------------------------------------------------------
    # ケース7: 今日から2営業日以内の枠が除外される
    # ---------------------------------------------------------------------------
    @freeze_time(FROZEN)  # 月曜 2026-05-18
    def test_two_business_days_minimum(self):
        """今日（月）・明日（火）は対象外で、2営業日後（水）以降が最初のスロット。"""
        service = _make_service([])
        slots = fetch_free_slots(service, days_ahead=14, duration_minutes=60, max_slots=10)

        dates = [s["date"] for s in slots]
        # 2026-05-18 (月) と 2026-05-19 (火) は含まれない
        assert "2026-05-18" not in dates
        assert "2026-05-19" not in dates
        # 2026-05-20 (水) 以降が最初
        if dates:
            assert min(dates) >= "2026-05-20"

    @freeze_time("2026-05-21")  # 木曜
    def test_two_business_days_from_thursday(self):
        """木曜基準: 1営業日後=金、2営業日後=月（翌週）。"""
        service = _make_service([])
        slots = fetch_free_slots(service, days_ahead=14, duration_minutes=60, max_slots=5)

        dates = [s["date"] for s in slots]
        # 2026-05-21 (木)、2026-05-22 (金) は除外
        assert "2026-05-21" not in dates
        assert "2026-05-22" not in dates
        # 2026-05-25 (月) 以降が最初
        if dates:
            assert min(dates) >= "2026-05-25"
