"""
商談ランク自動推定モジュール

リードのメモ・役職・部署情報からLLMを使って
商談確度（A〜E）を推定する。
CSVにランク情報がない場合や、★1〜5等の異なる形式の場合に使用。
"""

from typing import Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import Config
from src.utils import setup_logger

logger = setup_logger(__name__)

# ★形式・数値形式からA〜Eへの変換マップ（高い数値 = 高確度）
STAR_TO_RANK: Dict[str, str] = {"5": "A", "4": "B", "3": "C", "2": "D", "1": "E"}

SYSTEM_PROMPT = """あなたは展示会営業の専門家です。
展示会来場者の情報をもとに、フォローアップの優先度（商談確度）を
A〜Eの5段階で判定してください。

判定基準:
A: 即商談。「デモ希望」「来月導入したい」「予算確保済み」「決裁者が来場」等
B: 積極的フォロー。「詳細資料希望」「検討中」「上司に報告する」等
C: 通常フォロー。「興味あり」「将来的に検討」「情報収集」等
D: 軽めのフォロー。「とりあえず見に来た」「別担当に繋ぐ」等
E: 最小限の接触。「同伴」「通りすがり」「全く関心なし」等

必ずA/B/C/D/Eの1文字のみで回答してください。"""


class RankEstimator:
    """LLMを使って商談ランクを推定するクラス"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=0,
            openai_api_key=Config.OPENAI_API_KEY,
        )

    def normalize_rank(self, raw_rank: str) -> Optional[str]:
        """
        既存のランク値をA〜Eに正規化する。
        変換できない場合はNoneを返す（LLM推定にフォールバック）。

        Examples:
            "A" → "A"
            "★5" / "5" / "5.0" → "A"
            "★3" / "3" → "C"
            "high" / "Hot" → None
        """
        if not raw_rank:
            return None

        cleaned = str(raw_rank).strip()

        if cleaned.upper() in ("A", "B", "C", "D", "E"):
            return cleaned.upper()

        # ★☆を除去して数値に変換
        cleaned = cleaned.replace("★", "").replace("☆", "").strip()
        try:
            num_str = str(int(float(cleaned)))
            if num_str in STAR_TO_RANK:
                return STAR_TO_RANK[num_str]
        except (ValueError, TypeError):
            pass

        return None

    def estimate_from_lead(self, lead: Dict, enable_llm: bool = True) -> Dict:
        """
        リード情報からランクを推定する。

        既存 lead_rank の正規化を試み、できない場合のみ LLM で推定する。

        Parameters
        ----------
        lead : Dict
            リードデータ辞書
        enable_llm : bool
            True の場合、正規化失敗時に LLM 推定を行う

        Returns
        -------
        dict
            rank (A-E), method (existing/normalized/llm_estimated/default),
            original (元の値), confidence (high/medium/low)
        """
        raw_rank = str(lead.get("lead_rank", "")).strip()
        normalized = self.normalize_rank(raw_rank)

        if normalized:
            method = "existing" if raw_rank.upper() == normalized else "normalized"
            return {
                "rank": normalized,
                "method": method,
                "original": raw_rank,
                "confidence": "high",
            }

        if enable_llm:
            estimated = self._llm_estimate(lead)
            if estimated:
                return {
                    "rank": estimated,
                    "method": "llm_estimated",
                    "original": raw_rank,
                    "confidence": "medium",
                }

        logger.warning(f"  ランク推定失敗: {lead.get('visitor_name')} → デフォルトC")
        return {
            "rank": "C",
            "method": "default",
            "original": raw_rank,
            "confidence": "low",
        }

    def _llm_estimate(self, lead: Dict) -> Optional[str]:
        """LLMでランクを推定する"""
        try:
            prompt = f"""以下の展示会来場者情報から商談確度を判定してください。

来場者情報:
- 氏名: {lead.get('visitor_name', '不明')}
- 会社名: {lead.get('company_name', '不明')}
- 部署: {lead.get('department', '不明')}
- 役職: {lead.get('job_title', '不明')}
- 関心製品: {lead.get('interested_products', '不明')}
- 展示会でのメモ: {lead.get('memo', '（なし）')}
- 今後のご要望: {lead.get('future_requests', '（なし）')}

A〜Eの1文字のみで回答してください。"""

            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
            response = self.llm.invoke(messages)
            result = response.content.strip().upper()

            if result in ("A", "B", "C", "D", "E"):
                logger.info(f"  LLMランク推定: {lead.get('visitor_name')} → {result}")
                return result
        except Exception as e:
            logger.warning(f"  LLMランク推定エラー: {e}")

        return None
