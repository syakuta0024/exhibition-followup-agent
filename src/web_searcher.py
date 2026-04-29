"""
顧客企業のWeb検索モジュール

DuckDuckGo Search APIを使って顧客企業の情報を2段階で検索する。
  クエリA: 事業内容・製品・サービス（期間制限なし）
  クエリB: 最新ニュース・プレスリリース（直近1ヶ月）
APIキー不要・プライバシー保護・無料で利用可能。
"""

import re
from typing import Dict, List, Optional

from src.utils import setup_logger

logger = setup_logger(__name__)


class WebSearcher:
    """DuckDuckGoで顧客企業の事業内容・製品・最新動向を検索するクラス"""

    def search_company(
        self,
        company_name: str,
        max_results_per_query: int = 3,
    ) -> Dict:
        """
        顧客企業を2クエリで検索し、事業概要と最新動向を返す。

        Parameters
        ----------
        company_name : str
            検索対象の会社名
        max_results_per_query : int
            1クエリあたりの取得件数（デフォルト3件）

        Returns
        -------
        dict
            success, company_name, results (list), summary (str), error
            results の各要素: title, url, snippet, section("profile"|"news")
        """
        if not company_name or len(company_name.strip()) < 2:
            return self._empty_result(company_name, "会社名が短すぎます")

        short_name = self._normalize_for_search(company_name)

        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS
            except ImportError:
                return self._empty_result(company_name, "ddgs 未インストール（pip install ddgs）")

        profile_results = self._run_query(
            ddgs_cls=DDGS,
            query=f"{short_name} 事業内容 製品 サービス",
            section="profile",
            max_results=max_results_per_query,
            timelimit=None,
        )

        news_results = self._run_query(
            ddgs_cls=DDGS,
            query=f"{short_name} ニュース プレスリリース 新製品",
            section="news",
            max_results=max_results_per_query,
            timelimit="m",
        )

        # URL重複を除去（profileを優先、newsで同じURLは除く）
        seen_urls = {r["url"] for r in profile_results}
        deduped_news = [r for r in news_results if r["url"] not in seen_urls]

        all_results = profile_results + deduped_news

        if not all_results:
            logger.info(f"  Web検索: 結果なし（{company_name}）")
            return self._empty_result(company_name, "検索結果なし")

        summary = self._build_summary(company_name, profile_results, deduped_news)

        logger.info(
            f"  Web検索完了: 事業情報{len(profile_results)}件 / "
            f"最新動向{len(deduped_news)}件"
        )
        return {
            "success": True,
            "company_name": company_name,
            "results": all_results,
            "summary": summary,
            "error": None,
        }

    def _run_query(
        self,
        ddgs_cls,
        query: str,
        section: str,
        max_results: int,
        timelimit: Optional[str],
    ) -> List[Dict]:
        """1クエリを実行して結果リストを返す。失敗時は空リスト。"""
        logger.info(f"  Web検索クエリ: '{query}'")
        try:
            kwargs = {
                "region": "jp-ja",
                "safesearch": "moderate",
                "max_results": max_results,
            }
            if timelimit is not None:
                kwargs["timelimit"] = timelimit
            with ddgs_cls() as ddgs:
                raw = ddgs.text(query, **kwargs)
        except Exception as e:
            logger.warning(f"  Web検索エラー ({section}): {e}")
            return []

        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", "")[:250],
                "section": section,
            }
            for r in raw
        ]

    def _build_summary(
        self,
        company_name: str,
        profile_results: List[Dict],
        news_results: List[Dict],
    ) -> str:
        lines = []

        if profile_results:
            lines.append(f"【{company_name}の事業内容・製品・サービス】")
            for i, r in enumerate(profile_results, 1):
                lines.append(f"{i}. {r['title']}")
                if r["snippet"]:
                    lines.append(f"   {r['snippet']}")

        if news_results:
            if lines:
                lines.append("")
            lines.append(f"【{company_name}の最新動向（直近1ヶ月）】")
            for i, r in enumerate(news_results, 1):
                lines.append(f"{i}. {r['title']}")
                if r["snippet"]:
                    lines.append(f"   {r['snippet']}")

        return "\n".join(lines)

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
