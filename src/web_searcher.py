"""
顧客企業のWeb検索モジュール

DuckDuckGo Search APIを使って顧客企業の最新情報を検索する。
APIキー不要・プライバシー保護・無料で利用可能。
"""

import re
from typing import Dict, List

from src.utils import setup_logger

logger = setup_logger(__name__)


class WebSearcher:
    """DuckDuckGoで顧客企業の最新情報を検索するクラス"""

    def search_company(
        self,
        company_name: str,
        max_results: int = 3,
        timelimit: str = "m",
    ) -> Dict:
        """
        顧客企業名で検索し、最新情報を返す。

        Parameters
        ----------
        company_name : str
            検索対象の会社名
        max_results : int
            取得する検索結果数（デフォルト3件）
        timelimit : str
            検索期間（d=1日, w=1週間, m=1ヶ月, y=1年）

        Returns
        -------
        dict
            success, company_name, results (list), summary (str), error
        """
        if not company_name or len(company_name.strip()) < 2:
            return self._empty_result(company_name, "会社名が短すぎます")

        short_name = self._normalize_for_search(company_name)
        query = f"{short_name} ニュース プレスリリース"
        logger.info(f"  Web検索: '{query}'")

        try:
            from duckduckgo_search import DDGS
            from duckduckgo_search.exceptions import DuckDuckGoSearchException
        except ImportError:
            return self._empty_result(company_name, "duckduckgo-search 未インストール")

        try:
            with DDGS() as ddgs:
                raw_results = list(
                    ddgs.text(
                        keywords=query,
                        region="jp-ja",
                        safesearch="moderate",
                        timelimit=timelimit,
                        max_results=max_results,
                    )
                )
        except Exception as e:
            logger.warning(f"  Web検索エラー: {e}")
            return self._empty_result(company_name, str(e))

        if not raw_results:
            logger.info(f"  Web検索: 結果なし（{company_name}）")
            return self._empty_result(company_name, "検索結果なし")

        results: List[Dict] = [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", "")[:200],
            }
            for r in raw_results
        ]

        summary_lines = [f"【{company_name}の最新情報（直近1ヶ月）】"]
        for i, r in enumerate(results, 1):
            summary_lines.append(f"{i}. {r['title']}")
            if r["snippet"]:
                summary_lines.append(f"   {r['snippet']}")
        summary = "\n".join(summary_lines)

        logger.info(f"  Web検索完了: {len(results)}件ヒット")
        return {
            "success": True,
            "company_name": company_name,
            "results": results,
            "summary": summary,
            "error": None,
        }

    def _normalize_for_search(self, company_name: str) -> str:
        """検索用に会社名を正規化（会社形態を除去）"""
        name = company_name
        for pattern in [r"株式会社", r"有限会社", r"\(株\)", r"（株）"]:
            name = re.sub(pattern, "", name)
        return name.strip()

    def _empty_result(self, company_name: str, error: str) -> Dict:
        return {
            "success": False,
            "company_name": company_name,
            "results": [],
            "summary": "",
            "error": error,
        }
