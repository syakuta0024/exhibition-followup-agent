"""tests/test_crm_matcher.py — CRMMatcher の単体テスト"""

import pandas as pd
import pytest

from src.crm_matcher import CRMMatcher, MATCH_THRESHOLD


@pytest.fixture
def matcher():
    return CRMMatcher()


@pytest.fixture
def crm_df():
    return pd.DataFrame([
        {"email": "tanaka@example.com", "company_name": "山田製作所", "notes": "初回来場"},
        {"email": "sato@example.com",   "company_name": "北九州特殊鋼", "notes": "提案中"},
        {"email": "suzuki@corp.jp",     "company_name": "東海自動車部品", "notes": "PoC"},
    ])


# ── ケース1: メールアドレス完全一致 ─────────────────────────────────

def test_match_by_email_exact(matcher, crm_df):
    lead = {"email": "tanaka@example.com", "company_name": "株式会社山田製作所"}
    result = matcher.match(lead, crm_df)
    assert result is not None
    assert result["_crm_match_method"] == "email"
    assert result["_crm_match_score"] == 100
    assert result["notes"] == "初回来場"


def test_match_by_email_case_insensitive(matcher, crm_df):
    lead = {"email": "TANAKA@EXAMPLE.COM", "company_name": ""}
    result = matcher.match(lead, crm_df)
    assert result is not None
    assert result["_crm_match_method"] == "email"


# ── ケース2: 会社名ファジーマッチ（閾値以上）────────────────────────

def test_match_by_company_fuzzy_above_threshold(matcher, crm_df):
    # 「株式会社北九州特殊鋼」は正規化後「北九州特殊鋼」で完全一致 → スコア100
    lead = {"email": "", "company_name": "株式会社北九州特殊鋼"}
    result = matcher.match(lead, crm_df)
    assert result is not None
    assert result["_crm_match_method"] == "company_fuzzy"
    assert result["_crm_match_score"] >= MATCH_THRESHOLD
    assert result["notes"] == "提案中"


def test_match_by_company_fuzzy_partial(matcher):
    crm = pd.DataFrame([
        {"email": "", "company_name": "関東精密工業株式会社"}
    ])
    # 正規化後「関東精密工業」vs「関東精密」→ 类似度は高くないかもしれないが閾値テスト
    lead = {"email": "", "company_name": "株式会社関東精密工業"}
    result = matcher.match(lead, crm)
    assert result is not None  # 正規化後完全一致になるはず


# ── ケース3: 閾値以下の社名は無視される ─────────────────────────────

def test_match_company_below_threshold(matcher):
    crm = pd.DataFrame([
        {"email": "", "company_name": "全然違う会社名株式会社"}
    ])
    lead = {"email": "", "company_name": "株式会社山田製作所"}
    result = matcher.match(lead, crm)
    assert result is None


# ── ケース4: マッチなし → None を返す ────────────────────────────────

def test_no_match_returns_none(matcher, crm_df):
    lead = {"email": "nobody@nowhere.com", "company_name": "存在しない会社XYZ"}
    result = matcher.match(lead, crm_df)
    assert result is None


# ── ケース5: CRM DataFrame が空 → None を返す ──────────────────────

def test_empty_crm_df_returns_none(matcher):
    empty_df = pd.DataFrame(columns=["email", "company_name", "notes"])
    lead = {"email": "tanaka@example.com", "company_name": "山田製作所"}
    result = matcher.match(lead, empty_df)
    assert result is None


# ── 会社名正規化 ─────────────────────────────────────────────────────

def test_normalize_removes_kabushiki(matcher):
    assert matcher._normalize_company_name("株式会社山田製作所") == "山田製作所"
    assert matcher._normalize_company_name("山田製作所株式会社") == "山田製作所"
    assert matcher._normalize_company_name("(株)北九州特殊鋼") == "北九州特殊鋼"
    assert matcher._normalize_company_name("有限会社東海製缶") == "東海製缶"


def test_normalize_empty_string(matcher):
    assert matcher._normalize_company_name("") == ""


# ── match_all ────────────────────────────────────────────────────────

def test_match_all_adds_crm_columns(matcher, crm_df):
    leads_df = pd.DataFrame([
        {"email": "tanaka@example.com", "company_name": "山田製作所"},
        {"email": "nobody@x.com",       "company_name": "存在しない会社"},
    ])
    result_df = matcher.match_all(leads_df, crm_df)
    assert "_crm_match_score" in result_df.columns
    assert "_crm_match_method" in result_df.columns
    # 1行目はマッチ
    assert result_df.loc[0, "_crm_match_score"] == 100
    # 2行目はマッチなし
    assert result_df.loc[1, "_crm_match_score"] == 0
