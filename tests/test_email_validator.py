"""tests/test_email_validator.py — email_validator の単体テスト"""

import pytest
from src.email_validator import validate_email, _load_known_products, _check_too_many_products


LEAD_DIGIMA = {"interested_products": "DigiMA"}
LEAD_EDGE = {"interested_products": "EdgeGuard"}
LEAD_MULTI = {"interested_products": "DigiMA, EdgeGuard"}
LEAD_EMPTY = {"interested_products": ""}

# テスト用の既知製品セット（cli_config.yaml の known_products に相当）
_DEMO_PRODUCTS = {"DigiMA", "EdgeGuard", "FactoryBrain", "NTX-OCR", "SmartVision", "Sorani"}


# ── Rule 1: 内部 ID 混入 ──────────────────────────────────────────────

def test_internal_id_in_body():
    result = validate_email(
        subject="件名",
        body="リード L007 様にご連絡します",
        lead=LEAD_DIGIMA,
    )
    assert not result.passed
    assert any("L007" in e for e in result.errors)


def test_internal_id_in_subject():
    result = validate_email(
        subject="L012 様へのご提案",
        body="本文です",
        lead=LEAD_DIGIMA,
    )
    assert not result.passed
    assert any("L012" in e for e in result.errors)


def test_internal_id_short_not_flagged():
    # L12 は3桁未満なので対象外
    result = validate_email(
        subject="件名",
        body="L12 という記述",
        lead=LEAD_DIGIMA,
    )
    assert result.passed


# ── Rule 2: ダミー URL 検出 ─────────────────────────────────────────

def test_example_com_url():
    result = validate_email(
        subject="件名",
        body="詳細は https://example.com/product をご覧ください",
        lead=LEAD_DIGIMA,
    )
    assert not result.passed
    assert any("example.com" in e for e in result.errors)


def test_example_org_url():
    result = validate_email(
        subject="件名",
        body="https://www.example.org/page",
        lead=LEAD_DIGIMA,
    )
    assert not result.passed
    assert any("example.org" in e for e in result.errors)


# ── Rule 3: product_urls 未設定時の URL 混入 ────────────────────────

def test_url_in_body_no_product_urls():
    result = validate_email(
        subject="件名",
        body="詳細は https://company.co.jp/digima をご覧ください",
        lead=LEAD_DIGIMA,
        product_urls=None,
    )
    # warning（errors ではない）
    assert result.passed
    assert any("product_urls未設定" in w for w in result.warnings)


def test_url_in_body_with_empty_product_urls():
    result = validate_email(
        subject="件名",
        body="https://company.co.jp/digima",
        lead=LEAD_DIGIMA,
        product_urls={"DigiMA": "", "EdgeGuard": ""},
    )
    assert result.passed
    assert any("product_urls未設定" in w for w in result.warnings)


def test_url_in_body_with_valid_product_urls():
    # product_urls に値があれば Rule 3 は発動しない
    result = validate_email(
        subject="件名",
        body="https://company.co.jp/digima",
        lead=LEAD_DIGIMA,
        product_urls={"DigiMA": "https://company.co.jp/digima"},
    )
    assert result.passed
    assert not result.warnings


# ── Rule 4: 関心外製品への言及 ────────────────────────────────────────

def test_off_topic_product():
    # DigiMA のみ興味 → EdgeGuard が本文に登場 → warning
    result = validate_email(
        subject="件名",
        body="弊社の EdgeGuard はいかがでしょうか",
        lead=LEAD_DIGIMA,
        known_products=_DEMO_PRODUCTS,
    )
    assert result.passed  # warning のみ、errors なし
    assert any("EdgeGuard" in w for w in result.warnings)


def test_off_topic_product_not_flagged_if_interested():
    # EdgeGuard が interested に含まれる → warning なし
    result = validate_email(
        subject="件名",
        body="弊社の EdgeGuard はいかがでしょうか",
        lead=LEAD_EDGE,
        known_products=_DEMO_PRODUCTS,
    )
    assert result.passed
    assert not result.warnings


def test_multiple_interested_products():
    # DigiMA と EdgeGuard の両方 interested → Rule 4 は FactoryBrain のみ警告
    result = validate_email(
        subject="件名",
        body="DigiMA と EdgeGuard と FactoryBrain のご紹介",
        lead=LEAD_MULTI,
        known_products=_DEMO_PRODUCTS,
    )
    assert result.passed
    rule4_warnings = [w for w in result.warnings if "関心外製品" in w]
    assert any("FactoryBrain" in w for w in rule4_warnings)
    assert not any("DigiMA" in w for w in rule4_warnings)
    assert not any("EdgeGuard" in w for w in rule4_warnings)


def test_rule4_disabled_when_no_known_products():
    # known_products=None → Rule 4 は無効、関心外製品でも warning なし
    result = validate_email(
        subject="件名",
        body="弊社の EdgeGuard はいかがでしょうか",
        lead=LEAD_DIGIMA,
        known_products=None,
    )
    assert result.passed
    assert not result.warnings


def test_load_known_products_from_cfg():
    assert _load_known_products({"known_products": ["A", "B"]}) == {"A", "B"}
    assert _load_known_products({}) == set()


# ── 正常系 ────────────────────────────────────────────────────────────

def test_clean_email_passes():
    result = validate_email(
        subject="ご来場ありがとうございました",
        body="先日はお越しいただきありがとうございます。DigiMA のご説明をさせてください。",
        lead=LEAD_DIGIMA,
        product_urls={"DigiMA": "https://company.co.jp/digima"},
    )
    assert result.passed
    assert result.errors == []
    assert result.warnings == []


def test_clean_email_no_product_urls_no_urls_in_body():
    # product_urls 未設定でも本文に URL がなければ warning なし
    result = validate_email(
        subject="ご来場御礼",
        body="先日はご来場いただきありがとうございました。",
        lead=LEAD_DIGIMA,
        product_urls=None,
    )
    assert result.passed
    assert result.errors == []
    assert result.warnings == []


# ── Rule 5: 多製品言及チェック ────────────────────────────────────────

def test_too_many_products_warns_when_3_or_more():
    # 3件の製品が本文に登場 → warnings に追加される
    result = validate_email(
        subject="件名",
        body="DigiMA と EdgeGuard と FactoryBrain のご紹介です",
        lead={"interested_products": "DigiMA, EdgeGuard, FactoryBrain"},
        known_products=_DEMO_PRODUCTS,
    )
    assert result.passed
    assert any("製品言及が3件" in w for w in result.warnings)


def test_too_many_products_no_warn_when_2_or_less():
    # 2件以下なら warnings に追加されない
    result = validate_email(
        subject="件名",
        body="DigiMA と EdgeGuard のご紹介です",
        lead={"interested_products": "DigiMA, EdgeGuard"},
        known_products=_DEMO_PRODUCTS,
    )
    assert result.passed
    assert not any("製品言及" in w for w in result.warnings)


def test_too_many_products_skipped_when_known_products_empty():
    # known_products が空(None)なら Rule 5 はスキップされる
    result = validate_email(
        subject="件名",
        body="DigiMA と EdgeGuard と FactoryBrain のご紹介です",
        lead={"interested_products": ""},
        known_products=None,
    )
    assert result.passed
    assert not any("製品言及" in w for w in result.warnings)


# ── 複数ルール同時違反 ────────────────────────────────────────────────

def test_multiple_violations():
    result = validate_email(
        subject="L007 様",
        body="https://example.com/ をご覧ください。EdgeGuard もおすすめです。",
        lead=LEAD_DIGIMA,
        known_products=_DEMO_PRODUCTS,
    )
    assert not result.passed
    assert len(result.errors) >= 2  # 内部ID + ダミーURL
    assert any("EdgeGuard" in w for w in result.warnings)
