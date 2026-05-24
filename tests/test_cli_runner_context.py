"""
CRM CSV / 音声コンテキストの配線テスト。

- load_crm_csv() の正常系・異常系
- run_generate() が cli_config.yaml の crm_csv_path / audio_context.json から
  読み込んで process_lead に正しく渡すことを確認する。

外部 API（OpenAI / DDGS / ChromaDB）はすべてモック化する。
"""

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ────────────────────────────────────────────────────────────────
# load_crm_csv
# ────────────────────────────────────────────────────────────────

class TestLoadCrmCsv:
    def test_valid_csv_returns_standardized_dataframe(self, tmp_path):
        """ケース1: 正常な CSV を DataFrame に整形して返す。"""
        from src.utils import load_crm_csv

        csv = tmp_path / "crm.csv"
        csv.write_text(
            "Email,Company name,Lifecycle stage\n"
            "alice@example.com,Acme Inc.,Customer\n"
            "bob@example.com,Beta LLC,Lead\n",
            encoding="utf-8",
        )

        df = load_crm_csv(str(csv))
        assert df is not None
        assert len(df) == 2
        # auto_map_columns + apply_column_mapping で標準フィールド名にリネーム済み
        assert "email" in df.columns
        assert "company_name" in df.columns
        assert "lifecycle_stage" in df.columns
        assert df.iloc[0]["email"] == "alice@example.com"
        assert df.iloc[1]["company_name"] == "Beta LLC"

    def test_nonexistent_path_returns_none(self, tmp_path):
        """ケース2: 存在しないパスは None。クラッシュしない。"""
        from src.utils import load_crm_csv

        assert load_crm_csv(str(tmp_path / "does_not_exist.csv")) is None

    def test_empty_path_returns_none(self):
        """空文字パスでも None を返す（cli_config.yaml の crm_csv_path: "" 想定）。"""
        from src.utils import load_crm_csv

        assert load_crm_csv("") is None
        assert load_crm_csv(None) is None  # type: ignore[arg-type]

    def test_empty_file_returns_none(self, tmp_path):
        """ケース3: 空ファイル（ヘッダーのみ）は None を返す。"""
        from src.utils import load_crm_csv

        csv = tmp_path / "empty.csv"
        # ヘッダーのみ → 行0件
        csv.write_text("Email,Company name\n", encoding="utf-8")

        # ヘッダーのみは df.empty=True なので None
        assert load_crm_csv(str(csv)) is None

    def test_malformed_file_returns_none(self, tmp_path):
        """完全に壊れたファイルでも例外を投げずに None を返す。"""
        from src.utils import load_crm_csv

        bin_file = tmp_path / "binary.csv"
        bin_file.write_bytes(b"\x00\x01\x02\x03\x04")

        # pandas が UnicodeDecodeError 等を出しても except で None
        result = load_crm_csv(str(bin_file))
        assert result is None


# ────────────────────────────────────────────────────────────────
# build_lead_key
# ────────────────────────────────────────────────────────────────

class TestBuildLeadKey:
    def test_uses_lead_id_when_present(self):
        from src.cli_runner import build_lead_key
        assert build_lead_key({"lead_id": "L042", "visitor_name": "X", "company_name": "Y"}) == "L042"

    def test_falls_back_to_composite_key(self):
        from src.cli_runner import build_lead_key
        key = build_lead_key({"visitor_name": "田中 太郎", "company_name": "XYZ株式会社"})
        assert key == "田中 太郎_XYZ株式会社"

    def test_empty_lead_id_treated_as_missing(self):
        from src.cli_runner import build_lead_key
        key = build_lead_key({"lead_id": "", "visitor_name": "A", "company_name": "B"})
        assert key == "A_B"


# ────────────────────────────────────────────────────────────────
# load_audio_context
# ────────────────────────────────────────────────────────────────

class TestLoadAudioContext:
    def test_returns_empty_when_path_missing(self, tmp_path):
        from src.cli_runner import load_audio_context
        assert load_audio_context(str(tmp_path / "nope.json")) == {}

    def test_returns_empty_when_path_is_none(self):
        from src.cli_runner import load_audio_context
        assert load_audio_context(None) == {}
        assert load_audio_context("") == {}

    def test_loads_valid_json(self, tmp_path):
        from src.cli_runner import load_audio_context
        path = tmp_path / "audio.json"
        path.write_text(
            json.dumps({"L001": {"transcript": "hello", "needs": {"summary": "x"}}}),
            encoding="utf-8",
        )
        ctx = load_audio_context(str(path))
        assert ctx["L001"]["transcript"] == "hello"
        assert ctx["L001"]["needs"]["summary"] == "x"

    def test_returns_empty_on_malformed_json(self, tmp_path):
        from src.cli_runner import load_audio_context
        path = tmp_path / "broken.json"
        path.write_text("{not valid json", encoding="utf-8")
        assert load_audio_context(str(path)) == {}


# ────────────────────────────────────────────────────────────────
# run_generate の配線確認
#
# 戦略: 重たい依存（VectorDBManager / EmailGenerator / FollowUpAgent）を
# まるごとモックし、agent.process_lead に渡される kwargs を捕捉する。
# ────────────────────────────────────────────────────────────────

@pytest.fixture
def _stub_env(tmp_path, monkeypatch):
    """run_generate の前提を最小構成で揃える共通フィクスチャ。"""
    # 1) cwd を tmp_path に固定して、cli_config.yaml / output/ を隔離
    monkeypatch.chdir(tmp_path)

    # 2) ダミーの cli_config.yaml を書く
    (tmp_path / "cli_config.yaml").write_text(
        "sender_company: テスト株式会社\n"
        "sender_name: テスト太郎\n"
        "default_ranks: [A, B, C]\n"
        "enable_web_search: false\n"
        "enable_rank_estimation: false\n"
        "output_path: output/emails.csv\n"
        "leads_csv_path: data/leads.csv\n"
        "crm_csv_path: \"\"\n"
        "output_naming: overwrite\n",
        encoding="utf-8",
    )

    # 3) ダミーのリード CSV（lead_id 付き 2 件 + lead_id なし 1 件）
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "leads.csv").write_text(
        "lead_id,氏名,会社名,メールアドレス,商談確度\n"
        "L001,山田 太郎,ABC製造,yamada@abc.example,A\n"
        "L002,鈴木 花子,XYZ工業,suzuki@xyz.example,B\n"
        ",佐藤 次郎,DEF商事,sato@def.example,C\n",
        encoding="utf-8",
    )

    # 4) OPENAI_API_KEY を有効化（Config.validate を通すため）
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    # Config はモジュール import 時に env を読むので、属性も直接書き換える
    from src.config import Config
    monkeypatch.setattr(Config, "OPENAI_API_KEY", "sk-test-dummy")

    return tmp_path


def _patch_heavy_deps(captured: List[Dict[str, Any]]):
    """VectorDBManager / EmailGenerator / FollowUpAgent をモック化するヘルパ。

    process_lead が呼ばれるたびに captured に kwargs を追記する。
    """
    fake_agent = MagicMock()

    def fake_process_lead(**kwargs):
        captured.append(kwargs)
        lead = kwargs["lead"]
        return {
            "lead_id": lead.get("lead_id", ""),
            "visitor_name": lead.get("visitor_name", ""),
            "company_name": lead.get("company_name", ""),
            "lead_rank": lead.get("lead_rank", ""),
            "email_to": lead.get("email", ""),
            "subject": "stub subject",
            "body": "stub body",
            "cta": "",
        }

    fake_agent.process_lead.side_effect = fake_process_lead

    return [
        patch("src.cli_runner.VectorDBManager", return_value=MagicMock()),
        patch("src.cli_runner.EmailGenerator", return_value=MagicMock()),
        patch("src.cli_runner.FollowUpAgent", return_value=fake_agent),
    ]


class TestRunGenerateWiring:
    def test_crm_csv_path_loads_crm_df_and_passes_to_process_lead(self, _stub_env):
        """ケース4: crm_csv_path から DataFrame を読み込んで process_lead に渡る。"""
        # CRM CSV を用意して cli_config.yaml を書き換え
        crm_csv = _stub_env / "data" / "crm.csv"
        crm_csv.write_text(
            "Email,Company name,Lifecycle stage\n"
            "yamada@abc.example,ABC製造,Customer\n",
            encoding="utf-8",
        )
        cfg_path = _stub_env / "cli_config.yaml"
        cfg_path.write_text(
            cfg_path.read_text(encoding="utf-8").replace(
                'crm_csv_path: ""',
                f'crm_csv_path: "{crm_csv.as_posix()}"',
            ),
            encoding="utf-8",
        )

        captured: List[Dict[str, Any]] = []
        with patch("src.vectordb.VectorDBManager", return_value=MagicMock()), \
             patch("src.email_generator.EmailGenerator", return_value=MagicMock()), \
             patch("src.agent.FollowUpAgent") as mock_agent_cls:
            fake_agent = MagicMock()

            def fake_process_lead(**kwargs):
                captured.append(kwargs)
                lead = kwargs["lead"]
                return {
                    "lead_id": lead.get("lead_id", ""),
                    "visitor_name": lead.get("visitor_name", ""),
                    "company_name": lead.get("company_name", ""),
                    "lead_rank": lead.get("lead_rank", ""),
                    "email_to": lead.get("email", ""),
                    "subject": "x", "body": "y", "cta": "",
                }

            fake_agent.process_lead.side_effect = fake_process_lead
            mock_agent_cls.return_value = fake_agent

            from src.cli_runner import run_generate
            result = run_generate(audio_context_path=None)

        assert result["ok"] is True
        assert captured, "process_lead が一度も呼ばれていない"
        # crm_df が DataFrame として渡っていること
        for kw in captured:
            crm_df = kw.get("crm_df")
            assert isinstance(crm_df, pd.DataFrame)
            assert "email" in crm_df.columns

    def test_audio_context_is_picked_up_by_lead_id(self, _stub_env):
        """ケース5: audio_context.json の lead_id 紐づけが process_lead に渡る。"""
        audio_path = _stub_env / "output" / "audio_context.json"
        audio_path.parent.mkdir(exist_ok=True)
        audio_path.write_text(
            json.dumps({
                "L001": {
                    "transcript": "音声テキスト L001",
                    "needs": {"summary": "概要", "temperature": "high"},
                },
            }),
            encoding="utf-8",
        )

        captured: List[Dict[str, Any]] = []
        with patch("src.vectordb.VectorDBManager", return_value=MagicMock()), \
             patch("src.email_generator.EmailGenerator", return_value=MagicMock()), \
             patch("src.agent.FollowUpAgent") as mock_agent_cls:
            fake_agent = MagicMock()

            def fake_process_lead(**kwargs):
                captured.append(kwargs)
                lead = kwargs["lead"]
                return {
                    "lead_id": lead.get("lead_id", ""),
                    "visitor_name": lead.get("visitor_name", ""),
                    "company_name": lead.get("company_name", ""),
                    "lead_rank": "C",
                    "email_to": "", "subject": "x", "body": "y", "cta": "",
                }

            fake_agent.process_lead.side_effect = fake_process_lead
            mock_agent_cls.return_value = fake_agent

            from src.cli_runner import run_generate
            run_generate(audio_context_path=str(audio_path))

        # L001 のリードだけ transcript / needs が入る
        by_id = {kw["lead"].get("lead_id"): kw for kw in captured}
        assert "L001" in by_id
        assert by_id["L001"]["transcript"] == "音声テキスト L001"
        assert by_id["L001"]["extracted_needs"] == {"summary": "概要", "temperature": "high"}
        # L002 には音声なし
        assert by_id["L002"]["transcript"] == ""
        assert by_id["L002"]["extracted_needs"] is None

    def test_missing_audio_context_runs_with_empty(self, _stub_env):
        """ケース6: audio_context.json が無くてもクラッシュせず空コンテキストで動く。"""
        captured: List[Dict[str, Any]] = []
        with patch("src.vectordb.VectorDBManager", return_value=MagicMock()), \
             patch("src.email_generator.EmailGenerator", return_value=MagicMock()), \
             patch("src.agent.FollowUpAgent") as mock_agent_cls:
            fake_agent = MagicMock()

            def fake_process_lead(**kwargs):
                captured.append(kwargs)
                lead = kwargs["lead"]
                return {
                    "lead_id": lead.get("lead_id", ""),
                    "visitor_name": lead.get("visitor_name", ""),
                    "company_name": lead.get("company_name", ""),
                    "lead_rank": "C",
                    "email_to": "", "subject": "x", "body": "y", "cta": "",
                }

            fake_agent.process_lead.side_effect = fake_process_lead
            mock_agent_cls.return_value = fake_agent

            from src.cli_runner import run_generate
            result = run_generate(audio_context_path=str(_stub_env / "nope.json"))

        assert result["ok"] is True
        assert captured
        for kw in captured:
            assert kw["transcript"] == ""
            assert kw["extracted_needs"] is None
            assert kw.get("crm_df") is None

    def test_composite_key_matches_for_lead_without_id(self, _stub_env):
        """ケース7: lead_id が無いリードでも複合キーで音声が引ける。"""
        audio_path = _stub_env / "output" / "audio_context.json"
        audio_path.parent.mkdir(exist_ok=True)
        # 3行目の佐藤 次郎 / DEF商事 は lead_id 無し → 複合キーが採番より優先される
        # ※ apply_column_mapping は lead_id 列を「全行空のとき」だけ自動採番する。
        #   今回は 2 行に L001/L002 が入っているので自動採番は走らず、
        #   3 行目の lead_id は空のまま → build_lead_key は複合キーに落ちる
        audio_path.write_text(
            json.dumps({
                "佐藤 次郎_DEF商事": {
                    "transcript": "音声テキスト 佐藤",
                    "needs": {"summary": "佐藤さんの要望"},
                },
            }),
            encoding="utf-8",
        )

        captured: List[Dict[str, Any]] = []
        with patch("src.vectordb.VectorDBManager", return_value=MagicMock()), \
             patch("src.email_generator.EmailGenerator", return_value=MagicMock()), \
             patch("src.agent.FollowUpAgent") as mock_agent_cls:
            fake_agent = MagicMock()

            def fake_process_lead(**kwargs):
                captured.append(kwargs)
                lead = kwargs["lead"]
                return {
                    "lead_id": lead.get("lead_id", ""),
                    "visitor_name": lead.get("visitor_name", ""),
                    "company_name": lead.get("company_name", ""),
                    "lead_rank": "C",
                    "email_to": "", "subject": "x", "body": "y", "cta": "",
                }

            fake_agent.process_lead.side_effect = fake_process_lead
            mock_agent_cls.return_value = fake_agent

            from src.cli_runner import run_generate
            run_generate(audio_context_path=str(audio_path))

        # 佐藤 次郎 のリードを特定
        sato_kw = next(kw for kw in captured if kw["lead"].get("visitor_name") == "佐藤 次郎")
        assert sato_kw["transcript"] == "音声テキスト 佐藤"
        assert sato_kw["extracted_needs"] == {"summary": "佐藤さんの要望"}
