"""tests/test_web_searcher.py — WebSearcher の単体テスト

外部ネットワーク呼び出し（DDGS）はすべてモック。
search_company() の正常系・異常系・URL 重複除去などを検証する。
"""

from unittest.mock import MagicMock, patch

import pytest

from src.web_searcher import WebSearcher


# ── ヘルパー ─────────────────────────────────────────────────────────

def _make_ddgs_result(title: str, url: str, body: str):
    return {"title": title, "href": url, "body": body}


# ── search_company() 正常系 ────────────────────────────────────────

class TestSearchCompanySuccess:

    def test_returns_success_with_results(self):
        """正常時に success=True と results が返ること"""
        searcher = WebSearcher()

        profile_hit = _make_ddgs_result("ABC社公式", "https://abc.co.jp", "ABCの事業内容")
        news_hit    = _make_ddgs_result("ABCニュース", "https://news.co.jp/abc", "新製品を発表")

        fake_ddgs = MagicMock()
        fake_ddgs.__enter__ = MagicMock(return_value=fake_ddgs)
        fake_ddgs.__exit__ = MagicMock(return_value=False)
        fake_ddgs.text = MagicMock(side_effect=[[profile_hit], [news_hit]])

        with patch("src.web_searcher.DDGS", return_value=fake_ddgs, create=True):
            with patch.dict("sys.modules", {"ddgs": MagicMock(DDGS=MagicMock(return_value=fake_ddgs))}):
                # _run_query を直接モックして呼び出す
                searcher._run_query = MagicMock(side_effect=[
                    [{"title": "ABC社公式", "url": "https://abc.co.jp", "snippet": "事業内容", "section": "profile"}],
                    [{"title": "ABCニュース", "url": "https://news.co.jp/abc", "snippet": "新製品", "section": "news"}],
                ])
                result = searcher.search_company("ABC株式会社")

        assert result["success"] is True
        assert len(result["results"]) == 2
        assert result["summary"] != ""
        assert result["error"] is None

    def test_summary_contains_company_name(self):
        """summary に会社名が含まれること"""
        searcher = WebSearcher()
        searcher._run_query = MagicMock(side_effect=[
            [{"title": "テスト社概要", "url": "https://test.co.jp", "snippet": "概要", "section": "profile"}],
            [],
        ])

        result = searcher.search_company("テスト株式会社")

        assert "テスト" in result["summary"]

    def test_result_sections_are_labeled(self):
        """results の各要素に section が含まれること"""
        searcher = WebSearcher()
        searcher._run_query = MagicMock(side_effect=[
            [{"title": "プロフィール", "url": "https://a.jp", "snippet": "aaa", "section": "profile"}],
            [{"title": "ニュース", "url": "https://b.jp", "snippet": "bbb", "section": "news"}],
        ])

        result = searcher.search_company("サンプル社")

        sections = {r["section"] for r in result["results"]}
        assert "profile" in sections
        assert "news" in sections


# ── URL 重複除去 ────────────────────────────────────────────────────

class TestUrlDeduplication:

    def test_duplicate_urls_in_news_are_removed(self):
        """profile と news で同じ URL があれば news 側が除去されること"""
        searcher = WebSearcher()
        shared_url = "https://shared.co.jp"

        searcher._run_query = MagicMock(side_effect=[
            [{"title": "プロフィール", "url": shared_url, "snippet": "profile", "section": "profile"}],
            [{"title": "ニュース", "url": shared_url, "snippet": "news", "section": "news"},
             {"title": "別ニュース", "url": "https://other.co.jp", "snippet": "other", "section": "news"}],
        ])

        result = searcher.search_company("重複社")

        urls = [r["url"] for r in result["results"]]
        assert urls.count(shared_url) == 1  # 重複は1件だけ
        assert len(urls) == 2  # profile 1件 + 重複でない news 1件


# ── 空・短い会社名の異常系 ────────────────────────────────────────

class TestSearchCompanyEdgeCases:

    def test_empty_company_name_returns_error(self):
        """空文字の会社名は success=False を返すこと"""
        searcher = WebSearcher()
        result = searcher.search_company("")
        assert result["success"] is False
        assert result["error"] is not None

    def test_single_char_company_name_returns_error(self):
        """1文字の会社名は success=False を返すこと"""
        searcher = WebSearcher()
        result = searcher.search_company("A")
        assert result["success"] is False

    def test_no_results_returns_error(self):
        """検索結果が0件のとき success=False になること"""
        searcher = WebSearcher()
        searcher._run_query = MagicMock(return_value=[])

        result = searcher.search_company("存在しない架空の会社名XYZ")

        assert result["success"] is False
        assert result["results"] == []
        assert result["summary"] == ""


# ── _run_query() のエラーハンドリング ─────────────────────────────

class TestRunQueryErrorHandling:

    def test_run_query_returns_empty_on_exception(self):
        """_run_query() が例外を投げたとき空リストが返ること"""
        searcher = WebSearcher()

        fake_ddgs_cls = MagicMock()
        fake_instance = MagicMock()
        fake_instance.__enter__ = MagicMock(return_value=fake_instance)
        fake_instance.__exit__ = MagicMock(return_value=False)
        fake_instance.text.side_effect = RuntimeError("接続エラー")
        fake_ddgs_cls.return_value = fake_instance

        result = searcher._run_query(
            ddgs_cls=fake_ddgs_cls,
            query="テスト",
            section="profile",
            max_results=3,
            timelimit=None,
        )

        assert result == []

    def test_run_query_timelimit_passed_correctly(self):
        """timelimit='m' が DDGS.text() に渡されること"""
        searcher = WebSearcher()

        fake_ddgs_cls = MagicMock()
        fake_instance = MagicMock()
        fake_instance.__enter__ = MagicMock(return_value=fake_instance)
        fake_instance.__exit__ = MagicMock(return_value=False)
        fake_instance.text.return_value = []
        fake_ddgs_cls.return_value = fake_instance

        searcher._run_query(
            ddgs_cls=fake_ddgs_cls,
            query="ニュース",
            section="news",
            max_results=3,
            timelimit="m",
        )

        call_kwargs = fake_instance.text.call_args[1]
        assert call_kwargs.get("timelimit") == "m"

    def test_run_query_no_timelimit_when_none(self):
        """timelimit=None のとき timelimit キーが渡されないこと"""
        searcher = WebSearcher()

        fake_ddgs_cls = MagicMock()
        fake_instance = MagicMock()
        fake_instance.__enter__ = MagicMock(return_value=fake_instance)
        fake_instance.__exit__ = MagicMock(return_value=False)
        fake_instance.text.return_value = []
        fake_ddgs_cls.return_value = fake_instance

        searcher._run_query(
            ddgs_cls=fake_ddgs_cls,
            query="事業内容",
            section="profile",
            max_results=3,
            timelimit=None,
        )

        call_kwargs = fake_instance.text.call_args[1]
        assert "timelimit" not in call_kwargs


# ── _normalize_for_search() ────────────────────────────────────────

class TestNormalizeForSearch:

    @pytest.mark.parametrize("input_name,expected_contains", [
        ("株式会社テクノ", "テクノ"),
        ("テクノ株式会社", "テクノ"),
        ("有限会社サンプル", "サンプル"),
        ("(株)ABC", "ABC"),
        ("（株）テスト", "テスト"),
        ("普通会社名", "普通会社名"),  # パターンなし → そのまま
    ])
    def test_removes_corporate_suffix(self, input_name, expected_contains):
        searcher = WebSearcher()
        result = searcher._normalize_for_search(input_name)
        assert expected_contains in result
        # 除去した後に会社形態文字列が残っていないこと
        assert "株式会社" not in result or input_name == "普通会社名"


# ── _build_summary() ──────────────────────────────────────────────

class TestBuildSummary:

    def test_includes_profile_section(self):
        """profile_results がある場合、事業内容セクションが含まれること"""
        searcher = WebSearcher()
        profile = [{"title": "会社概要", "url": "https://a.jp", "snippet": "製品の説明"}]
        summary = searcher._build_summary("テスト社", profile, [])
        assert "事業内容" in summary
        assert "会社概要" in summary

    def test_includes_news_section(self):
        """news_results がある場合、最新動向セクションが含まれること"""
        searcher = WebSearcher()
        news = [{"title": "新製品発表", "url": "https://b.jp", "snippet": "新製品リリース"}]
        summary = searcher._build_summary("テスト社", [], news)
        assert "最新動向" in summary
        assert "新製品発表" in summary

    def test_empty_results_returns_empty_string(self):
        """どちらも空のとき空文字になること"""
        searcher = WebSearcher()
        summary = searcher._build_summary("テスト社", [], [])
        assert summary == ""
