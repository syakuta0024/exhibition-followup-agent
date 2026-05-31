"""
メール生成後の Python 機械チェック層（LLM なし・API コストゼロ）

§3.3 の Layer 1 実装。内部 ID 混入・ダミー URL・URL 混入・関心外製品の4ルールを
正規表現で検査し、ValidationResult を返す。
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class ValidationResult:
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _load_known_products(cfg: dict) -> Set[str]:
    """cli_config.yaml の known_products から製品名セットを返す。未設定は空セット（チェック無効）。"""
    return set(cfg.get("known_products", []))


def _check_too_many_products(body: str, known_products: Set[str]) -> List[str]:
    """
    Rule 5: メール本文中の製品言及が3件以上の場合に警告を返す。
    known_products が空の場合はスキップ。
    """
    if not known_products:
        return []
    mentioned = [p for p in known_products if p in body]
    if len(mentioned) >= 3:
        return [f"製品言及が{len(mentioned)}件あります（{', '.join(mentioned)}）。主要1〜2件に絞ることを推奨します。"]
    return []


def validate_email(
    subject: str,
    body: str,
    lead: Dict,
    product_urls: Optional[Dict] = None,
    known_products: Optional[Set[str]] = None,
) -> ValidationResult:
    """
    生成済みメールを静的パターン検査する。

    Parameters
    ----------
    subject : str
    body : str
    lead : dict
        リード情報（interested_products を参照）
    product_urls : dict, optional
        製品URL辞書。値がすべて空なら「product_urls 未設定」とみなす。
    known_products : set, optional
        関心外製品チェックに使う製品名セット。None または空セットの場合は Rule 4 を無効化。
        通常は cli_config.yaml の known_products を _load_known_products() で読み込んで渡す。

    Returns
    -------
    ValidationResult
        passed=False かつ errors に内容あり → LLM 再生成 or フラグ付き出力を推奨
    """
    errors: List[str] = []
    warnings: List[str] = []
    full_text = subject + " " + body

    # Rule 1: 内部 ID 混入 (L\d{3,})
    found = re.findall(r"L\d{3,}", full_text)
    if found:
        errors.append(f"内部ID混入: {found}")

    # Rule 2: ダミー URL 検出
    found = re.findall(
        r"https?://(?:www\.)?example\.(?:com|org|net)\S*", body
    )
    if found:
        errors.append(f"ダミーURL: {found}")

    # Rule 3: product_urls 未設定時の URL 混入
    if not any((product_urls or {}).values()):
        found = re.findall(r"https?://\S+", body)
        if found:
            warnings.append(f"URL混入(product_urls未設定): {found}")

    # Rule 4: 関心外製品への言及（known_products が空なら無効）
    if known_products:
        interested = [
            p.strip()
            for p in str(lead.get("interested_products", "")).split(",")
            if p.strip()
        ]
        off = [p for p in known_products if p not in interested and p in body]
        if off:
            warnings.append(f"関心外製品の言及候補: {off}")

    # Rule 5: 製品言及数が3件以上（known_products が空なら無効）
    warnings.extend(_check_too_many_products(body, known_products or set()))

    return ValidationResult(
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )
