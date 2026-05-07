from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List

_WEEKDAYS_JA = ["月", "火", "水", "木", "金", "土", "日"]
_TIME_SLOT_RE = re.compile(r"^(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})$")
_DATE_RE = re.compile(r"^(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})$")


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)


def parse_candidate_dates(input_str: str) -> List[Dict]:
    candidates = []
    for line in input_str.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        date_part = parts[0].strip()
        if not _DATE_RE.match(date_part):
            raise ValueError(f"日付フォーマットエラー: '{date_part}'")
        time_slots = []
        if len(parts) == 2:
            for slot in parts[1].strip().split(","):
                slot = slot.strip()
                if slot:
                    time_slots.append(slot)
        candidates.append({"date": date_part, "time_slots": time_slots})
    if not candidates:
        raise ValueError("候補日が1件も見つかりませんでした")
    return candidates


def validate_dates(candidates: List[Dict]) -> ValidationResult:
    errors = []
    today = date.today()
    for item in candidates:
        date_str = item.get("date", "")
        m = _DATE_RE.match(date_str)
        if not m:
            errors.append(f"日付フォーマットエラー: '{date_str}'")
            continue
        try:
            candidate_date = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError as e:
            errors.append(f"無効な日付: '{date_str}' ({e})")
            continue
        if candidate_date < today:
            errors.append(f"過去日は指定できません: '{date_str}'")
        for slot in item.get("time_slots", []):
            sm = _TIME_SLOT_RE.match(slot)
            if not sm:
                errors.append(f"時間帯フォーマットエラー: '{slot}' (例: '10:00-12:00')")
                continue
            sh, smin, eh, emin = int(sm.group(1)), int(sm.group(2)), int(sm.group(3)), int(sm.group(4))
            if not (0 <= sh <= 23 and 0 <= smin <= 59):
                errors.append(f"無効な開始時刻: '{slot}'")
                continue
            if not (0 <= eh <= 23 and 0 <= emin <= 59):
                errors.append(f"無効な終了時刻: '{slot}'")
                continue
            if sh * 60 + smin >= eh * 60 + emin:
                errors.append(f"開始時刻が終了時刻以降です: '{slot}'")
    return ValidationResult(is_valid=len(errors) == 0, errors=errors)


def _parse_date(date_str: str) -> date:
    m = _DATE_RE.match(date_str)
    if not m:
        raise ValueError(f"無効な日付: '{date_str}'")
    return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))


def format_for_display(candidates: List[Dict]) -> List[str]:
    result = []
    for item in candidates:
        d = _parse_date(item["date"])
        weekday = _WEEKDAYS_JA[d.weekday()]
        date_label = f"{d.year}年{d.month}月{d.day}日({weekday})"
        slots = item.get("time_slots", [])
        result.append(f"{date_label} {' / '.join(slots)}" if slots else date_label)
    return result


def format_for_llm_prompt(candidates: List[Dict]) -> str:
    lines = format_for_display(candidates)
    bullets = "\n".join(f"  - {line}" for line in lines)
    return (
        "以下の候補日のいずれかをメール本文に組み込んでください:\n"
        + bullets
        + "\nこれ以外の日付・曜日を本文に書かないでください。"
    )
