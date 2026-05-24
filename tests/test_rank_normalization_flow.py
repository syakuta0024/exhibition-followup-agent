"""tests/test_rank_normalization_flow.py — ランク値正規化の配線テスト

正規化の優先順位:
  ①/rank-mapping で保存された rank_value_mapping
  ②"X：説明" 形式の自動抽出（フォールバック）
  ③RankEstimator の LLM 推定（未知形式のみ）
"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.utils import filter_leads_by_rank
from src.rank_estimator import RankEstimator


# ── フィクスチャ ──────────────────────────────────────────────────────────────

@pytest.fixture
def rank_estimator():
    """LLM を呼ばない RankEstimator インスタンス"""
    with patch("src.rank_estimator.ChatOpenAI", return_value=MagicMock()):
        return RankEstimator()


def _make_csv(tmp_path, rows):
    """テスト用リードCSVを tmp_path に書き出してパスを返す"""
    path = tmp_path / "leads.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    return str(path)


def _base_config(csv_path, rank_value_mapping=None):
    """run_load_leads のモック用設定辞書"""
    return {
        "leads_csv_path": csv_path,
        "default_ranks": ["A", "B", "C"],
        "rank_value_mapping": rank_value_mapping or {},
    }


# ── ケース1: 保存済みマッピングで正規化 ──────────────────────────────────────

class TestCase1_SavedMappingNormalizesAndFilters:

    def test_rx_japan_format_normalized_and_filtered(self, tmp_path):
        """"A：決裁者商談" → "A" に正規化され、ランクフィルタ A が拾われること"""
        from src.cli_runner import run_load_leads

        csv_path = _make_csv(tmp_path, [
            {"氏名": "田中一郎", "会社名": "株式会社A", "商談確度": "A：決裁者商談", "メールアドレス": "t@a.com"},
            {"氏名": "鈴木花子", "会社名": "株式会社B", "商談確度": "C：情報収集",   "メールアドレス": "s@b.com"},
        ])
        cfg = _base_config(csv_path, {"A：決裁者商談": "A", "C：情報収集": "C"})

        with patch("src.cli_runner.load_cli_config", return_value=cfg):
            result = run_load_leads(csv_path=csv_path, ranks=["A"])

        assert result["ok"]
        assert result["total"] == 1
        assert result["leads_df"].iloc[0]["lead_rank"] == "A"

    def test_all_mapped_ranks_normalized(self, tmp_path):
        """複数のランクが一括で正規化されること"""
        from src.cli_runner import run_load_leads

        csv_path = _make_csv(tmp_path, [
            {"氏名": "田中", "会社名": "A社", "商談確度": "A：決裁者商談",   "メールアドレス": "a@a.com"},
            {"氏名": "鈴木", "会社名": "B社", "商談確度": "B：担当者フォロー", "メールアドレス": "b@b.com"},
            {"氏名": "佐藤", "会社名": "C社", "商談確度": "C：情報収集",     "メールアドレス": "c@c.com"},
        ])
        cfg = _base_config(csv_path, {
            "A：決裁者商談": "A",
            "B：担当者フォロー": "B",
            "C：情報収集": "C",
        })

        with patch("src.cli_runner.load_cli_config", return_value=cfg):
            result = run_load_leads(csv_path=csv_path, ranks=["A", "B", "C"])

        assert result["total"] == 3
        ranks = set(result["leads_df"]["lead_rank"].tolist())
        assert ranks == {"A", "B", "C"}


# ── ケース2: 自動フォールバック（"X：説明" 形式） ────────────────────────────

class TestCase2_AutoFallbackForXColonFormat:

    def test_auto_normalize_x_colon_format(self, tmp_path):
        """rank_value_mapping なしで "B：担当者フォロー" が "B" に自動正規化されること"""
        from src.cli_runner import run_load_leads

        csv_path = _make_csv(tmp_path, [
            {"氏名": "田中", "会社名": "A社", "商談確度": "B：担当者フォロー", "メールアドレス": "t@a.com"},
            {"氏名": "鈴木", "会社名": "B社", "商談確度": "A：決裁者商談",   "メールアドレス": "s@b.com"},
        ])

        with patch("src.cli_runner.load_cli_config", return_value=_base_config(csv_path)):
            result = run_load_leads(csv_path=csv_path, ranks=["A", "B"])

        assert result["ok"]
        assert result["total"] == 2
        ranks = set(result["leads_df"]["lead_rank"].tolist())
        assert ranks == {"A", "B"}

    def test_lowercase_prefix_also_normalized(self, tmp_path):
        """小文字プレフィックス "a：..." も大文字 "A" に正規化されること"""
        from src.cli_runner import run_load_leads

        csv_path = _make_csv(tmp_path, [
            {"氏名": "田中", "会社名": "A社", "商談確度": "a：決裁者商談", "メールアドレス": "t@a.com"},
        ])

        with patch("src.cli_runner.load_cli_config", return_value=_base_config(csv_path)):
            result = run_load_leads(csv_path=csv_path, ranks=["A"])

        assert result["total"] == 1
        assert result["leads_df"].iloc[0]["lead_rank"] == "A"


# ── ケース3: 未知形式は触らない ─────────────────────────────────────────────

class TestCase3_UnknownFormatNotTouched:

    def test_star_format_passes_through_unchanged(self, tmp_path):
        """"★5" は自動正規化されず、RankEstimator に委ねられること"""
        from src.cli_runner import run_load_leads

        csv_path = _make_csv(tmp_path, [
            {"氏名": "田中", "会社名": "A社", "商談確度": "★5", "メールアドレス": "t@a.com"},
        ])

        with patch("src.cli_runner.load_cli_config", return_value=_base_config(csv_path)):
            result = run_load_leads(csv_path=csv_path, ranks=["A", "B", "C"])

        # "★5" は A〜E にも "X：説明" 形式にも当てはまらない → フィルタ通過0件
        assert result["ok"]
        assert result["total"] == 0
        # by_rank には "★5" がそのまま残っていること
        assert "★5" in result["by_rank"]

    def test_english_format_passes_through_unchanged(self, tmp_path):
        """"Hot" のような英語形式も触らないこと"""
        from src.cli_runner import run_load_leads

        csv_path = _make_csv(tmp_path, [
            {"氏名": "田中", "会社名": "A社", "商談確度": "Hot", "メールアドレス": "t@a.com"},
        ])

        with patch("src.cli_runner.load_cli_config", return_value=_base_config(csv_path)):
            result = run_load_leads(csv_path=csv_path, ranks=["A", "B", "C"])

        assert result["total"] == 0
        assert "Hot" in result["by_rank"]


# ── ケース4: filter_leads_by_rank が完全一致で動く ──────────────────────────

class TestCase4_FilterWithFullStringEquality:

    def test_clean_ranks_filtered_correctly(self):
        """正規化済みの A/B は正しくフィルタされること"""
        df = pd.DataFrame({"lead_rank": ["A", "B", "C", "D", "E"]})
        result = filter_leads_by_rank(df, ["A", "B"])
        assert set(result["lead_rank"].tolist()) == {"A", "B"}
        assert len(result) == 2

    def test_x_colon_format_not_matched_after_hack_removal(self):
        """正規化前の "A：決裁者商談" はフィルタ ["A"] に一致しないこと（先頭1文字ハック除去の確認）"""
        df = pd.DataFrame({"lead_rank": ["A：決裁者商談", "B：担当者フォロー"]})
        result = filter_leads_by_rank(df, ["A", "B"])
        # 完全一致のため 0 件
        assert len(result) == 0

    def test_empty_ranks_returns_all(self):
        """ranks が空のときは全件返すこと"""
        df = pd.DataFrame({"lead_rank": ["A", "B", "C"]})
        result = filter_leads_by_rank(df, [])
        assert len(result) == 3

    def test_case_insensitive_filter(self):
        """小文字の ranks 指定でも大文字と一致すること"""
        df = pd.DataFrame({"lead_rank": ["A", "B", "C"]})
        result = filter_leads_by_rank(df, ["a", "b"])
        assert set(result["lead_rank"].tolist()) == {"A", "B"}


# ── ケース5: normalize_rank() が "A：説明" 形式を変換 ───────────────────────

class TestCase5_NormalizeRankHandlesXColonFormat:

    def test_x_colon_format_normalized(self, rank_estimator):
        """RankEstimator.normalize_rank が "A：決裁者商談" を "A" に変換すること"""
        assert rank_estimator.normalize_rank("A：決裁者商談") == "A"
        assert rank_estimator.normalize_rank("B：担当者フォロー") == "B"
        assert rank_estimator.normalize_rank("C：情報収集") == "C"
        assert rank_estimator.normalize_rank("D：継続ウォッチ") == "D"
        assert rank_estimator.normalize_rank("E：見込みなし") == "E"

    def test_colon_variants_normalized(self, rank_estimator):
        """全角コロン・半角コロン・スペース区切りに対応すること"""
        assert rank_estimator.normalize_rank("A：全角コロン") == "A"
        assert rank_estimator.normalize_rank("B:半角コロン") == "B"
        assert rank_estimator.normalize_rank("C 半角スペース") == "C"

    def test_plain_rank_still_works(self, rank_estimator):
        """従来の A/★5/3 形式が引き続き正規化されること"""
        assert rank_estimator.normalize_rank("A") == "A"
        assert rank_estimator.normalize_rank("★5") == "A"
        assert rank_estimator.normalize_rank("3") == "C"
        assert rank_estimator.normalize_rank("1") == "E"

    def test_unknown_format_returns_none(self, rank_estimator):
        """変換できない形式は None を返すこと（LLM 推定に委ねる）"""
        assert rank_estimator.normalize_rank("Hot") is None
        assert rank_estimator.normalize_rank("★★★") is None
        assert rank_estimator.normalize_rank("") is None


# ── ケース6: 保存済みマッピングが自動フォールバックより優先 ──────────────────

class TestCase6_SavedMappingPrioritizedOverAutoFallback:

    def test_saved_mapping_overrides_auto_fallback(self, tmp_path):
        """保存済みマッピングの値が自動抽出より優先されること"""
        from src.cli_runner import run_load_leads

        csv_path = _make_csv(tmp_path, [
            {"氏名": "田中", "会社名": "A社", "商談確度": "A：決裁者商談", "メールアドレス": "t@a.com"},
        ])
        # 意図的に "A：決裁者商談" → "B" という変則マッピング
        cfg = _base_config(csv_path, {"A：決裁者商談": "B"})

        with patch("src.cli_runner.load_cli_config", return_value=cfg):
            result_b = run_load_leads(csv_path=csv_path, ranks=["B"])
            result_a = run_load_leads(csv_path=csv_path, ranks=["A"])

        # 保存済みマッピング "B" が適用され、ランクB フィルタに通ること
        assert result_b["total"] == 1
        assert result_b["leads_df"].iloc[0]["lead_rank"] == "B"
        # ランクA フィルタには通らないこと
        assert result_a["total"] == 0

    def test_empty_mapping_does_not_block_auto_fallback(self, tmp_path):
        """rank_value_mapping が空({}) のとき、自動フォールバックが動作すること"""
        from src.cli_runner import run_load_leads

        csv_path = _make_csv(tmp_path, [
            {"氏名": "田中", "会社名": "A社", "商談確度": "A：決裁者商談", "メールアドレス": "t@a.com"},
        ])
        cfg = _base_config(csv_path, {})  # 空マッピング → フォールバック有効

        with patch("src.cli_runner.load_cli_config", return_value=cfg):
            result = run_load_leads(csv_path=csv_path, ranks=["A"])

        assert result["total"] == 1
        assert result["leads_df"].iloc[0]["lead_rank"] == "A"
