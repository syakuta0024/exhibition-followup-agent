"""AudioMatcher の green / red 判定および LeadManager 形式（日付/時刻分離）対応のテスト"""

from datetime import datetime

import pandas as pd
import pytest

from src.audio_matcher import AudioMatcher, _resolve_scan_dt


def _audio_meta(filename: str, start_time):
    return {
        "filename": filename,
        "duration_sec": 60.0,
        "start_time": start_time,
        "file_bytes": b"",
    }


def test_parse_rep_from_filename_with_rep():
    matcher = AudioMatcher()
    assert matcher.parse_rep_from_filename("20260115_営業A_001.m4a") == "営業A"
    assert matcher.parse_rep_from_filename("営業B_002.wav") == "営業B"


def test_parse_rep_from_filename_digit_only_returns_none():
    """ファイル名が日付＋連番のみ（担当者名なし）の場合は None を返す（red 扱いになる）"""
    matcher = AudioMatcher()
    assert matcher.parse_rep_from_filename("20260115_001.m4a") is None
    assert matcher.parse_rep_from_filename("260115_001.mp3") is None  # YYMMDD 6桁
    assert matcher.parse_rep_from_filename("untitled.mp3") is None


def test_match_green_with_split_date_and_time_columns():
    """LeadManager 形式（visit_date と scan_time が別列・scan_time は HH:MM のみ）でも
    タイムスタンプ一致 → green 判定になることを確認する。
    """
    leads_df = pd.DataFrame([
        {"visitor_name": "田中 誠",   "rep_name": "営業A", "visit_date": "2026-01-15", "scan_time": "10:30"},
        {"visitor_name": "佐藤 美咲", "rep_name": "営業A", "visit_date": "2026-01-15", "scan_time": "13:00"},
        {"visitor_name": "渡辺 大輔", "rep_name": "営業A", "visit_date": "2026-01-15", "scan_time": "15:45"},
    ])
    audios = [
        _audio_meta("20260115_営業A_001.m4a", datetime(2026, 1, 15, 10, 30)),
        _audio_meta("20260115_営業A_002.m4a", datetime(2026, 1, 15, 13, 0)),
    ]

    matcher = AudioMatcher()
    results = matcher.match(audios, leads_df, rep_col="rep_name", timestamp_col="scan_time")

    by_fname = {r.audio_filename: r for r in results}

    r1 = by_fname["20260115_営業A_001.m4a"]
    assert r1.confidence == "green"
    assert r1.method == "timestamp"
    assert leads_df.loc[r1.lead_idx, "visitor_name"] == "田中 誠"

    r2 = by_fname["20260115_営業A_002.m4a"]
    assert r2.confidence == "green"
    assert r2.method == "timestamp"
    assert leads_df.loc[r2.lead_idx, "visitor_name"] == "佐藤 美咲"


def test_match_red_when_filename_has_no_rep_name():
    """ファイル名に担当者名が含まれない（日付＋連番のみ）→ red・未紐づけ"""
    leads_df = pd.DataFrame([
        {"visitor_name": "田中 誠",   "rep_name": "営業A", "visit_date": "2026-01-15", "scan_time": "10:30"},
    ])
    audios = [_audio_meta("20260115_001.m4a", datetime(2026, 1, 15, 10, 30))]

    matcher = AudioMatcher()
    results = matcher.match(audios, leads_df, rep_col="rep_name", timestamp_col="scan_time")

    assert len(results) == 1
    r = results[0]
    assert r.confidence == "red"
    assert r.method == "unmatched"
    assert r.lead_idx is None
    assert r.rep_name is None


# ----------------------------------------------------------------------
# _resolve_scan_dt のフォーマット網羅テスト
# ----------------------------------------------------------------------

EXPECTED_DT = datetime(2026, 1, 15, 10, 30)


@pytest.mark.parametrize("time_str", [
    "10:30",       # HH:MM
    "10:30:00",    # HH:MM:SS
    "10時30分",    # 日本語
    "1030",        # 4桁数字
])
def test_resolve_scan_dt_time_formats(time_str):
    """時刻側の4フォーマットすべてが visit_date と結合できる"""
    row = pd.Series({"scan_time": time_str, "visit_date": "2026-01-15"})
    assert _resolve_scan_dt(row, "scan_time", "visit_date") == EXPECTED_DT


@pytest.mark.parametrize("date_str", [
    "2026-01-15",   # ISO ハイフン
    "2026/01/15",   # スラッシュ
    "20260115",     # 8桁数字
])
def test_resolve_scan_dt_date_formats(date_str):
    """日付側のフォーマットが scan_time と結合できる（年あり）"""
    row = pd.Series({"scan_time": "10:30", "visit_date": date_str})
    assert _resolve_scan_dt(row, "scan_time", "visit_date") == EXPECTED_DT


def test_resolve_scan_dt_japanese_date_with_year():
    """2026年1月15日（年あり日本語）"""
    row = pd.Series({"scan_time": "10:30", "visit_date": "2026年1月15日"})
    assert _resolve_scan_dt(row, "scan_time", "visit_date") == EXPECTED_DT


def test_resolve_scan_dt_japanese_date_without_year_uses_current_year():
    """1月15日（年なし日本語）は現在年で補完される"""
    row = pd.Series({"scan_time": "10:30", "visit_date": "1月15日"})
    result = _resolve_scan_dt(row, "scan_time", "visit_date")
    assert result == datetime(datetime.now().year, 1, 15, 10, 30)


@pytest.mark.parametrize("combined_str", [
    "2026-01-15 10:30",
    "2026/01/15 10:30",
])
def test_resolve_scan_dt_combined_single_column(combined_str):
    """visit_date と scan_time が同一列にまとまっている（date_col 不要）"""
    row = pd.Series({"scan_time": combined_str})
    assert _resolve_scan_dt(row, "scan_time", date_col=None) == EXPECTED_DT


def test_resolve_scan_dt_invalid_time_returns_none_or_fallback():
    """ありえない時刻（25:00）は時刻と判定せず、_parse_dt フォールバックで None になる"""
    row = pd.Series({"scan_time": "25:00", "visit_date": "2026-01-15"})
    # 25:00 は時刻として無効 → 結合スキップ → _parse_dt("25:00") も失敗 → None
    assert _resolve_scan_dt(row, "scan_time", "visit_date") is None


def test_match_green_with_each_time_format():
    """match() レベルで各時刻フォーマットが green になることを確認"""
    matcher = AudioMatcher()
    for time_str in ["10:30", "10:30:00", "10時30分", "1030"]:
        leads_df = pd.DataFrame([
            {"visitor_name": "田中 誠", "rep_name": "営業A",
             "visit_date": "2026-01-15", "scan_time": time_str},
        ])
        audios = [_audio_meta("20260115_営業A_001.m4a", EXPECTED_DT)]
        results = matcher.match(audios, leads_df, rep_col="rep_name", timestamp_col="scan_time")
        assert len(results) == 1
        assert results[0].confidence == "green", f"time format {time_str!r} did not match"
        assert results[0].method == "timestamp"
