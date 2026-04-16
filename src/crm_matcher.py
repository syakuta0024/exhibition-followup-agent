"""
CRMマッチングモジュール

リードの情報とHubSpot CRM CSVを2段階ロジックで紐付ける。

紐付け優先度:
  1. メールアドレス完全一致（大文字小文字を無視）→ _crm_match_method="email"
  2. 会社名ファジーマッチング（rapidfuzz、スコア80以上）→ _crm_match_method="company_fuzzy"

会社名の正規化処理:
  「株式会社」「(株)」等の表記ゆれを除去してから比較することで、
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
    リードとCRM情報を紐付けるクラス。

    紐付けロジック:
    1. メールアドレス完全一致（最優先）
    2. 会社名のファジーマッチング（NFKC正規化 + 法人格除去 + fuzz.ratio）
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
        lead: Dict[str, Any],
        crm_df: pd.DataFrame,
    ) -> Optional[Dict[str, Any]]:
        """
        リード情報に最も近いCRMレコードを2段階ロジックで返す。

        紐付けロジック（優先度順）:
        1. Email完全一致（大文字小文字を無視）
           → 見つかればそれを返す（スコア100 / _crm_match_method="email"）
        2. 会社名ファジーマッチング（rapidfuzz.fuzz.ratio >= MATCH_THRESHOLD）
           → スコア最大のレコードを返す（_crm_match_method="company_fuzzy"）
        3. どちらも失敗 → None

        Parameters
        ----------
        lead : Dict[str, Any]
            リードデータ辞書（email, company_name キーを参照）
        crm_df : pd.DataFrame
            CRM商談データのDataFrame（email / company_name カラムを想定）

        Returns
        -------
        Optional[Dict[str, Any]]
            マッチしたCRMレコードの辞書。
            _crm_match_score（0〜100）と _crm_match_method を含む。
            マッチなしの場合は None。
        """
        if crm_df.empty:
            return None

        lead_email = str(lead.get("email", "")).strip().lower()
        lead_company = str(lead.get("company_name", "")).strip()

        # ── 1. Email完全一致 ──────────────────────────────────────
        if lead_email and "email" in crm_df.columns:
            for idx, row in crm_df.iterrows():
                crm_email = str(row.get("email", "")).strip().lower()
                if crm_email and crm_email == lead_email:
                    result = crm_df.loc[idx].to_dict()
                    result["_crm_match_score"] = 100
                    result["_crm_match_method"] = "email"
                    logger.debug(f"  Emailマッチ: '{lead_email}'")
                    return result

        # ── 2. 会社名ファジーマッチング ───────────────────────────
        if lead_company and "company_name" in crm_df.columns:
            norm_lead = self._normalize_company_name(lead_company)
            if not norm_lead:
                return None

            best_score = 0
            best_idx = None

            for idx, row in crm_df.iterrows():
                norm_crm = self._normalize_company_name(str(row.get("company_name", "")))
                if not norm_crm:
                    continue
                score = fuzz.ratio(norm_lead, norm_crm)
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_score >= MATCH_THRESHOLD and best_idx is not None:
                result = crm_df.loc[best_idx].to_dict()
                result["_crm_match_score"] = best_score
                result["_crm_match_method"] = "company_fuzzy"
                logger.debug(
                    f"  会社名マッチ: '{lead_company}' → "
                    f"'{crm_df.loc[best_idx, 'company_name']}' (スコア: {best_score})"
                )
                return result

        logger.debug(
            f"  マッチなし: email='{lead_email}', company='{lead_company}'"
        )
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
        マッチスコアは「_crm_match_score」、マッチ方法は「_crm_match_method」に格納。

        Parameters
        ----------
        leads_df : pd.DataFrame
            リードデータのDataFrame
        crm_df : pd.DataFrame
            CRM商談データのDataFrame
        lead_company_col : str
            リードの会社名カラム名（内部参照用）
        crm_company_col : str
            CRMの会社名カラム名（内部参照用）

        Returns
        -------
        pd.DataFrame
            CRM情報が_crm_プレフィックス付きで追加されたリードDataFrame
        """
        result_df = leads_df.copy()

        # CRM側のカラムに _crm_ プレフィックスを付けて初期化
        crm_extra_cols = [c for c in crm_df.columns if c not in ("email", crm_company_col)]
        for col in crm_extra_cols:
            result_df[f"_crm_{col}"] = ""
        result_df["_crm_match_score"] = 0
        result_df["_crm_match_method"] = ""
        result_df["_crm_matched_company"] = ""

        total = len(leads_df)
        match_count = 0

        for idx, row in leads_df.iterrows():
            lead = row.to_dict()
            matched = self.match(lead, crm_df)

            if matched:
                match_count += 1
                for col in crm_extra_cols:
                    result_df.at[idx, f"_crm_{col}"] = matched.get(col, "")
                result_df.at[idx, "_crm_match_score"] = matched.get("_crm_match_score", 0)
                result_df.at[idx, "_crm_match_method"] = matched.get("_crm_match_method", "")
                result_df.at[idx, "_crm_matched_company"] = matched.get(crm_company_col, "")

        logger.info(f"CRMマッチング完了: {total}件中 {match_count}件がマッチ")
        return result_df
