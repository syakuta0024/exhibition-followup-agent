"""
LLM-as-a-Judge による品質評価層（§3.2 / Layer 2 実装）

email_validator.py（Layer 1）を通過したメールを対象に、
LLM でトーン・内容・製品情報の正確性を審査する。

呼び出しパターンは email_generator.py に準拠:
  ChatOpenAI + langchain_core.messages.SystemMessage / HumanMessage
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class JudgeResult:
    passed: bool
    score: int          # 0–100
    issues: List[str] = field(default_factory=list)
    recommendation: str = ""


_SYSTEM_PROMPT = """あなたはビジネスメールの品質審査員です。以下の観点で評価してください:
1. ビジネスマナーとして適切か
2. 顧客の関心・状況に沿った内容か
3. 内部管理情報(IDコード等)が漏れていないか
4. 製品情報が正確か(架空URLがないか)
5. ランクに合ったトーンか(A=積極的提案、B=丁寧な提案、C=情報提供型)

以下の JSON のみ出力(前置き・後置きなし、マークダウン不可):
{"passed": true/false, "score": 0-100, "issues": ["問題点1", ...], "recommendation": "改善提案"}"""


def judge_email(
    subject: str,
    body: str,
    lead: Dict,
    lead_rank: str,
    llm_model: Optional[str] = None,
) -> JudgeResult:
    """
    生成済みメールを LLM で品質審査する。

    Parameters
    ----------
    subject : str
    body : str
    lead : dict
        リード情報（interested_products / memo を参照）
    lead_rank : str
        商談確度ランク（A–E）。トーン評価の基準に使う。
    llm_model : str, optional
        使用モデル。None 時は Config.LLM_MODEL を使用。

    Returns
    -------
    JudgeResult
        passed=False → agent.py でフラグ付き出力またはリトライを推奨
    """
    from src.config import Config
    model = llm_model or Config.LLM_MODEL

    user = (
        f"顧客ランク: {lead_rank}\n"
        f"関心製品: {lead.get('interested_products', '')}\n"
        f"メモ: {lead.get('memo', '')}\n\n"
        f"件名: {subject}\n"
        f"本文:\n{body}"
    )

    response = _call_llm(model=model, system=_SYSTEM_PROMPT, user=user)
    return _parse_judge_response(response)


def _call_llm(model: str, system: str, user: str) -> str:
    """LLM を呼び出してテキストレスポンスを返す。temperature=0 で deterministic に。"""
    from src.config import Config
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatOpenAI(
        model=model,
        temperature=0,
        openai_api_key=Config.OPENAI_API_KEY,
    )
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    response = llm.invoke(messages)
    return response.content


def _parse_judge_response(text: str) -> JudgeResult:
    """
    LLM レスポンスを JudgeResult にパースする。
    JSON コードブロック（```json ... ```）にも対応。
    パース失敗時はクラッシュせず failed の JudgeResult を返す。
    """
    try:
        clean = text.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
            clean = clean.strip()
        data = json.loads(clean)
        return JudgeResult(
            passed=bool(data.get("passed", False)),
            score=int(data.get("score", 0)),
            issues=list(data.get("issues", [])),
            recommendation=str(data.get("recommendation", "")),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return JudgeResult(
            passed=False,
            score=0,
            issues=["審査レスポンスのパース失敗"],
            recommendation=text[:200],
        )
