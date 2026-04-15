"""
CRMマッチングモジュール

リードの会社名とCRM CSVの会社名をファジーマッチングで紐付ける。
「株式会社」「(株)」等の表記ゆれを正規化してから比較することで、
異なるツール間のデータ不一致を吸収する。

使用ライブラリ: rapidfuzz (高速なLevenshtein距離ベースのファジーマッチング)
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional

import pandas as pd
from rapidfuzz import fuzz

from src.utils import setup_logger

logger = setup_logger(__name__)

# マッチありと判定する最低スコア（0〜100）
# 80以上 = 十分な類似度。「山田製作所」と「株式会社山田製作所」は正規化後に完全一致するため100になる
MATCH_THRESHOLD = 80


class CRMMatcher:
    """
    会社名のファジーマッチングでリードとCRM情報を紐付けるクラス。

    正規化処理:
    - 「株式会社」「有限会社」「合同会社」「(株)」等の会社形態を除去
    - 全角英数字 → 半角英数字に変換（NFKC正規化）
    - 前後空白の除去・連続空白の統一

    マッチングアルゴリズム:
    - rapidfuzz.fuzz.ratio（レーベンシュタイン距離ベース）
    - スコア80以上をマッチありと判定
    """

    # 正規化時に除去する会社形態のパターン
    _COMPANY_SUFFIX_PATTERNS: List[str] = [
        "株式会社",
        "有限会社",
        "合同会社",
        r"\(株\)",
        r"（株）",
        r"\(有\)",
        r"（有）",
        r"\(合\)",
        r"（合）",
    ]

    def _normalize_company_name(self, name: str) -> str:
        """
        会社名を正規化して比較しやすい形式に変換する。

        Parameters
        ----------
        name : str
            元の会社名（例: 「株式会社山田製作所」「(株)北九州特殊鋼」）

        Returns
        -------
        str
            正規化後の会社名（例: 「山田製作所」「北九州特殊鋼」）
        """
        if not name:
            return ""

        # NFKC正規化: 全角英数字 → 半角、全角スペース → 半角スペース
        name = unicodedata.normalize("NFKC", name)

        # 会社形態サフィックス・プレフィックスを除去
        for pattern in self._COMPANY_SUFFIX_PATTERNS:
            name = re.sub(pattern, "", name)

        # 連続空白を1つに統一し、前後の空白を除去
        name = re.sub(r"\s+", " ", name).strip()

        return name

    def match(
        self,
        lead_company: str,
        crm_df: pd.DataFrame,
        company_col: str = "company_name",
    ) -> Optional[Dict[str, Any]]:
        """
        リードの会社名に最も近いCRMレコードを返す。

        正規化後の会社名でfuzz.ratioスコアを計算し、
        スコアがMATCH_THRESHOLD以上の最高スコアのレコードを返す。

        Parameters
        ----------
        lead_company : str
            リードの会社名
        crm_df : pd.DataFrame
            CRM商談データのDataFrame
        company_col : str
            CRM DataFrameの会社名カラム名（デフォルト: "company_name"）

        Returns
        -------
        Optional[Dict[str, Any]]
            マッチしたCRMレコードの辞書（_crm_match_score キーを含む）。
            マッチなしの場合は None。
        """
        if not lead_company or crm_df.empty:
            return None

        if company_col not in crm_df.columns:
            logger.warning(f"CRM DataFrameに '{company_col}' カラムが存在しません")
            return None

        norm_lead = self._normalize_company_name(lead_company)
        if not norm_lead:
            return None

        best_score = 0
        best_idx = None

        for idx, row in crm_df.iterrows():
            norm_crm = self._normalize_company_name(str(row.get(company_col, "")))
            if not norm_crm:
                continue

            score = fuzz.ratio(norm_lead, norm_crm)

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_score >= MATCH_THRESHOLD and best_idx is not None:
            result = crm_df.loc[best_idx].to_dict()
            result["_crm_match_score"] = best_score
            logger.debug(
                f"  マッチ: '{lead_company}' → '{crm_df.loc[best_idx, company_col]}' "
                f"(スコア: {best_score})"
            )
            return result

        logger.debug(f"  マッチなし: '{lead_company}' (最高スコア: {best_score})")
        return None

    def match_all(
        self,
        leads_df: pd.DataFrame,
        crm_df: pd.DataFrame,
        lead_company_col: str = "company_name",
        crm_company_col: str = "company_name",
    ) -> pd.DataFrame:
        """
        全リードにCRM情報を紐付けしたDataFrameを返す。

        マッチしたCRMレコードのカラムが「_crm_元カラム名」として追加される。
        マッチスコアは「_crm_match_score」カラムに格納される。

        Parameters
        ----------
        leads_df : pd.DataFrame
            リードデータのDataFrame
        crm_df : pd.DataFrame
            CRM商談データのDataFrame
        lead_company_col : str
            リードの会社名カラム名
        crm_company_col : str
            CRMの会社名カラム名

        Returns
        -------
        pd.DataFrame
            CRM情報が_crm_プレフィックス付きで追加されたリードDataFrame
        """
        result_df = leads_df.copy()

        # CRM側の全カラム（会社名カラムを除く）に _crm_ プレフィックスを付けて初期化
        crm_extra_cols = [c for c in crm_df.columns if c != crm_company_col]
        for col in crm_extra_cols:
            result_df[f"_crm_{col}"] = ""
        result_df["_crm_match_score"] = 0
        result_df["_crm_matched_company"] = ""  # マッチしたCRM側の会社名

        total = len(leads_df)
        match_count = 0

        for idx, row in leads_df.iterrows():
            lead_company = str(row.get(lead_company_col, ""))
            matched = self.match(lead_company, crm_df, crm_company_col)

            if matched:
                match_count += 1
                # CRMカラムを _crm_ プレフィックス付きで格納
                for col in crm_extra_cols:
                    result_df.at[idx, f"_crm_{col}"] = matched.get(col, "")
                result_df.at[idx, "_crm_match_score"] = matched.get("_crm_match_score", 0)
                result_df.at[idx, "_crm_matched_company"] = matched.get(crm_company_col, "")

        logger.info(f"CRMマッチング完了: {total}件中 {match_count}件がマッチ")
        return result_df


# -------------------------
# 動作確認用スクリプト
# -------------------------
if __name__ == "__main__":
    import sys
    from src.utils import load_leads
    from src.config import Config

    print("=" * 60)
    print("CRMMatcher 動作確認")
    print("=" * 60)

    matcher = CRMMatcher()

    # ── 正規化テスト ────────────────────────────────────────────
    print("\n[1] 正規化テスト")
    test_names = [
        "株式会社山田製作所",
        "中部鉄鋼工業株式会社",
        "(株)北九州特殊鋼",
        "東洋精工(株)",
        "三河ゴム製造株式会社",
        "有限会社東海製缶",
        "ＡＢＣＤ株式会社",  # 全角英字
    ]
    for name in test_names:
        normalized = matcher._normalize_company_name(name)
        print(f"  '{name}' → '{normalized}'")

    # ── CSVを使ったマッチングテスト ──────────────────────────────
    print("\n[2] leads.csv と crm_demo.csv のマッチングテスト")

    crm_path = "data/crm_demo.csv"
    leads_path = Config.LEADS_CSV_PATH

    if not __import__("os").path.exists(crm_path):
        print(f"  ※ {crm_path} が見つかりません。先に作成してください。")
        sys.exit(0)

    leads_df = load_leads(leads_path)
    crm_df = __import__("pandas").read_csv(crm_path, dtype=str, encoding="utf-8-sig").fillna("")

    print(f"  リード数: {len(leads_df)}件  / CRM数: {len(crm_df)}件")
    print()

    # 表記ゆれのあるケースを個別に確認
    test_cases = [
        ("株式会社山田製作所",   "山田製作所",          "✅ 期待値: マッチ"),
        ("中部鉄鋼工業株式会社", "中部鉄鋼工業",        "✅ 期待値: マッチ"),
        ("株式会社北九州特殊鋼", "(株)北九州特殊鋼",    "✅ 期待値: マッチ"),
        ("東洋精工株式会社",     "東洋精工(株)",        "✅ 期待値: マッチ"),
        ("株式会社三河ゴム製造", "三河ゴム製造株式会社", "✅ 期待値: マッチ"),
        ("架空株式会社テスト",   "（存在しない会社）",   "❌ 期待値: マッチなし"),
    ]

    print("-" * 50)
    for lead_name, crm_note, expected in test_cases:
        result = matcher.match(lead_name, crm_df, company_col="顧客企業名")
        if result:
            matched_name = result.get("顧客企業名", "?")
            score = result.get("_crm_match_score", 0)
            print(f"  {expected}")
            print(f"    '{lead_name}' → '{matched_name}' (スコア: {score})")
        else:
            print(f"  {expected}")
            print(f"    '{lead_name}' → マッチなし")
        print()

    # 全件マッチングテスト（match_allは標準カラム名が必要なので個別テストで代替）
    print("-" * 50)
    print("[3] 全リードのマッチング結果サマリー")
    match_count = 0
    for _, row in leads_df.iterrows():
        company = row.get("company_name", "")
        result = matcher.match(company, crm_df, company_col="顧客企業名")
        status = f"✅ スコア{result['_crm_match_score']}" if result else "－ なし"
        print(f"  {company:25s}  {status}")
        if result:
            match_count += 1

    print()
    print(f"  合計マッチ: {match_count}/{len(leads_df)}件")
    print("=" * 60)
    print("動作確認完了")
    print("=" * 60)
