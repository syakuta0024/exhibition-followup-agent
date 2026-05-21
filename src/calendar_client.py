"""
Google Calendar API クライアント。

Gmail と同じ credentials/token.json を使って空き時間を取得し、
メール候補日スロットとして返す。
"""

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src.gmail_drafter import SCOPES

logger = logging.getLogger(__name__)

_TIMEZONE = ZoneInfo("Asia/Tokyo")
_WEEKDAYS_JP = ["月", "火", "水", "木", "金", "土", "日"]


def _add_business_days(start: date, n: int) -> date:
    """start から n 営業日後の日付を返す（土日はスキップ）。"""
    d = start
    added = 0
    while added < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            added += 1
    return d


def get_calendar_service(
    token_path: str = "credentials/token.json",
    credentials_path: str = "credentials/credentials.json",
):
    """
    Gmail と同じ token.json / credentials.json を使って
    Google Calendar API のサービスオブジェクトを返す。

    token.json が存在しない or スコープ不足の場合はブラウザ認証を起動する。
    """
    _token = Path(token_path)
    _creds_file = Path(credentials_path)

    creds: Optional[Credentials] = None
    if _token.exists():
        creds = Credentials.from_authorized_user_file(str(_token), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not _creds_file.exists():
                raise FileNotFoundError(
                    f"credentials.json が見つかりません: {_creds_file}\n"
                    "Google Cloud Console から OAuth クライアント ID をダウンロードして "
                    "credentials/ フォルダに置いてください。"
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(_creds_file), SCOPES)
            creds = flow.run_local_server(port=0)

        _token.parent.mkdir(parents=True, exist_ok=True)
        _token.write_text(creds.to_json(), encoding="utf-8")

    return build("calendar", "v3", credentials=creds)


def fetch_free_slots(
    service,
    days_ahead: int = 14,
    duration_minutes: int = 60,
    working_hours: tuple = (9, 18),
    max_slots: int = 5,
) -> list[dict]:
    """
    今日から days_ahead 日以内の空き時間を返す。

    ルール:
    - 平日のみ（土日除外）
    - working_hours の範囲内（デフォルト 9:00〜18:00）
    - 既存の予定と重複しない duration_minutes 分の空き枠を探す
    - 最大 max_slots 件を返す
    - 今日から2営業日以降を対象にする（直近すぎる日付は除外）

    戻り値:
    [
        {
            "date": "2026-05-20",
            "weekday": "水",
            "start": "14:00",
            "end": "15:00",
            "display": "5月20日（水）14:00〜15:00"
        },
        ...
    ]
    """
    today = datetime.now(tz=_TIMEZONE).date()
    min_date = _add_business_days(today, 2)
    max_date = today + timedelta(days=days_ahead)

    if min_date > max_date:
        return []

    time_min = datetime(min_date.year, min_date.month, min_date.day, 0, 0, 0, tzinfo=_TIMEZONE)
    time_max = datetime(max_date.year, max_date.month, max_date.day, 23, 59, 59, tzinfo=_TIMEZONE)

    try:
        result = service.freebusy().query(body={
            "timeMin": time_min.isoformat(),
            "timeMax": time_max.isoformat(),
            "items": [{"id": "primary"}],
            "timeZone": "Asia/Tokyo",
        }).execute()
        busy_raw = result.get("calendars", {}).get("primary", {}).get("busy", [])
    except HttpError as e:
        if e.status_code == 403:
            raise RuntimeError(
                "Google Calendar API が有効化されていないか、スコープが不足しています。\n"
                "Google Cloud Console で Calendar API を有効化し、"
                "OAuth 同意画面に calendar.readonly スコープを追加してください。\n"
                "また credentials/token.json を削除して再認証してください。"
            ) from e
        logger.warning("Calendar FreeBusy API エラー: %s", e)
        return []
    except Exception as e:
        logger.warning("Calendar 取得エラー: %s", e)
        return []

    busy: list[tuple[datetime, datetime]] = [
        (
            datetime.fromisoformat(p["start"]).astimezone(_TIMEZONE),
            datetime.fromisoformat(p["end"]).astimezone(_TIMEZONE),
        )
        for p in busy_raw
    ]

    hour_start, hour_end = working_hours
    slots: list[dict] = []
    current = min_date

    while current <= max_date and len(slots) < max_slots:
        if current.weekday() >= 5:  # 土日はスキップ
            current += timedelta(days=1)
            continue

        day_end_dt = datetime(current.year, current.month, current.day, hour_end, 0, tzinfo=_TIMEZONE)
        slot_start = datetime(current.year, current.month, current.day, hour_start, 0, tzinfo=_TIMEZONE)

        while len(slots) < max_slots:
            slot_end = slot_start + timedelta(minutes=duration_minutes)
            if slot_end > day_end_dt:
                break

            is_free = not any(
                slot_start < b_end and slot_end > b_start
                for b_start, b_end in busy
            )

            if is_free:
                wd = _WEEKDAYS_JP[current.weekday()]
                slots.append({
                    "date": current.strftime("%Y-%m-%d"),
                    "weekday": wd,
                    "start": slot_start.strftime("%H:%M"),
                    "end": slot_end.strftime("%H:%M"),
                    "display": (
                        f"{current.month}月{current.day}日（{wd}）"
                        f"{slot_start.strftime('%H:%M')}〜{slot_end.strftime('%H:%M')}"
                    ),
                })

            slot_start = slot_end  # 次のスロットへ

        current += timedelta(days=1)

    return slots


def format_slots_for_email(slots: list[dict]) -> str:
    """
    メール本文に挿入できる形式に整形して返す。

    例:
    「5月20日（水）14時〜、5月21日（木）10時〜、5月22日（金）13時〜はいかがでしょうか」
    """
    if not slots:
        return ""

    parts = []
    for slot in slots:
        d = datetime.strptime(slot["date"], "%Y-%m-%d")
        start_hour = slot["start"].split(":")[0].lstrip("0") or "0"
        wd = slot["weekday"]
        parts.append(f"{d.month}月{d.day}日（{wd}）{start_hour}時〜")

    if len(parts) == 1:
        return f"{parts[0]}はいかがでしょうか"
    return "、".join(parts) + "はいかがでしょうか"
