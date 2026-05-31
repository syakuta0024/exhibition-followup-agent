"""tests/test_profile.py — 処理プロファイルの保存・読み込みテスト（§3.13）"""

import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

import src.cli_runner as cli_runner
from src.cli_runner import load_last_run_profile, save_last_run_profile


# ── フィクスチャ ────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def tmp_profile_path(tmp_path, monkeypatch):
    """テストごとに一時ディレクトリを使い、LAST_RUN_PROFILE_PATH を書き換える。"""
    profile_file = str(tmp_path / "profiles" / "last_run.yaml")
    monkeypatch.setattr(cli_runner, "LAST_RUN_PROFILE_PATH", profile_file)
    return profile_file


# ── ケース1: save → ファイルが書き込まれる ────────────────────────────

def test_save_creates_yaml_file(tmp_profile_path):
    params = {
        "exhibition_name": "製造業DX展2026",
        "exhibition_date": "2026年4月24日〜26日",
        "exhibition_venue": "東京ビッグサイト",
        "ranks": ["A", "B"],
        "schedule_policy": "ab_only",
        "candidate_dates": [{"date": "2026/5/8", "time_slots": ["10:00-12:00"]}],
    }
    save_last_run_profile(params)
    assert Path(tmp_profile_path).exists()
    with open(tmp_profile_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["exhibition_name"] == "製造業DX展2026"
    assert data["ranks"] == ["A", "B"]


# ── ケース2: load が保存済みファイルを正しく読み込む ─────────────────

def test_load_reads_saved_profile(tmp_profile_path):
    params = {
        "exhibition_name": "IoT/M2M展",
        "exhibition_date": "2026年6月1日",
        "exhibition_venue": "幕張メッセ",
        "ranks": ["A"],
        "schedule_policy": "all",
        "candidate_dates": None,
    }
    save_last_run_profile(params)
    profile = load_last_run_profile()
    assert profile is not None
    assert profile["exhibition_name"] == "IoT/M2M展"
    assert profile["ranks"] == ["A"]
    assert profile["schedule_policy"] == "all"


# ── ケース3: ファイルが存在しないとき None を返す ──────────────────────

def test_load_returns_none_when_file_missing():
    profile = load_last_run_profile()
    assert profile is None


# ── ケース4: saved_at が ISO 形式で保存されている ──────────────────────

def test_saved_at_is_iso_format(tmp_profile_path):
    save_last_run_profile({
        "exhibition_name": "テスト展",
        "ranks": ["A", "B", "C"],
        "schedule_policy": "ab_only",
    })
    with open(tmp_profile_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # ISO形式としてパースできること（例外が出なければ OK）
    datetime.fromisoformat(data["saved_at"])


# ── ケース5: run_generate 失敗時（errors > 0）はプロファイルが保存されない ─

def test_run_generate_does_not_save_profile_on_error(tmp_profile_path):
    """run_load_leads が ok=False を返すとき save_last_run_profile() が呼ばれないことを検証する。"""
    with patch("src.cli_runner.save_last_run_profile") as mock_save:
        with patch("src.config.Config") as mock_config:
            mock_config.validate.return_value = None
            mock_config.CHROMA_DB_DIR = "chroma_db"
            mock_config.CHROMA_COLLECTION_NAME = "test"
            mock_config.OPENAI_API_KEY = "sk-test"
            with patch("src.cli_runner.run_load_leads") as mock_leads:
                mock_leads.return_value = {
                    "ok": False,
                    "message": "CSVが見つかりません",
                    "total": 0,
                    "success": 0,
                    "errors": 0,
                    "output_path": "",
                    "leads_df": None,
                }
                result = cli_runner.run_generate()
    # run_load_leads が ok=False を返すので run_generate も早期リターン
    assert result["ok"] is False
    mock_save.assert_not_called()
