"""
フォローアップエージェントモジュール

LangChain エージェントは使わず、シンプルなオーケストレーター設計を採用。
VectorDBManager で関連情報を検索し、EmailGenerator でメールを生成する
パイプラインを提供する。

CRM情報の取得優先順位:
  1. CRM CSV がある場合 → CRMMatcher でファジーマッチング（CSV連携モード）
  2. CRM CSV がない場合 → vectordb.search_crm() にフォールバック（従来モード）
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils import setup_logger, parse_interested_products, check_lead_quality
from src.web_searcher import WebSearcher
from src.rank_estimator import RankEstimator

logger = setup_logger(__name__)


class FollowUpAgent:
    """
    展示会フォローアップエージェントクラス。

    VectorDBManagerで関連情報を検索し、EmailGeneratorでメールを生成する
    シンプルなオーケストレーター。LangChain エージェントは使用しない。
    """

    def __init__(self, vectordb_manager, email_generator):
        """
        Parameters
        ----------
        vectordb_manager : VectorDBManager
            ベクトルDB検索インスタンス
        email_generator : EmailGenerator
            メール生成インスタンス
        """
        self.vectordb = vectordb_manager
        self.email_gen = email_generator
        self.web_searcher = WebSearcher()
        self.rank_estimator = RankEstimator()
        logger.info("FollowUpAgent 初期化完了")

    def process_lead(
        self,
        lead: Dict[str, Any],
        crm_df: Optional[pd.DataFrame] = None,
        exhibition_info: Optional[Dict[str, Any]] = None,
        on_step=None,
        enable_web_search: bool = True,
        enable_rank_estimation: bool = True,
        sender_company: str = "",
        sender_name: str = "",
        transcript: str = "",
        extracted_needs: Optional[Dict[str, Any]] = None,
        product_urls: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        1件のリードを処理してフォローアップメールを生成する。

        処理フロー:
        1. interested_products でベクトルDB検索（技術資料）
        2. CRM情報の取得:
           - crm_df あり → CRMMatcher でファジーマッチング（CSV連携）
           - crm_df なし → vectordb.search_crm() にフォールバック
        3. EmailGenerator でメール生成
        4. 結果を辞書で返す

        Parameters
        ----------
        lead : Dict[str, Any]
            リードデータ（1行分の辞書）
        crm_df : pd.DataFrame, optional
            CRM商談データのDataFrame。None の場合はvectordb検索を使用。

        Returns
        -------
        Dict[str, Any]
            生成結果の辞書。以下のキーを含む:
            - lead_id, visitor_name, company_name, lead_rank, email_to
            - subject, body, cta
            - ref_tech_docs: 参照した技術資料ファイル名リスト
            - ref_crm: 参照したCRMファイル名リスト
            - crm_match_score: CRMマッチスコア（0〜100、マッチなしは0）
            - crm_deal_stage: 商談ステージ（マッチなしは空文字）
            - crm_source: "csv" / "vectordb" / "none"（CRM情報の取得元）
        """
        visitor = lead.get("visitor_name", "")
        company = lead.get("company_name", "")
        logger.info(f"処理開始: {visitor} さん ({company}) [ランク {lead.get('lead_rank')}]")

        def _step(num: int, name: str, status: str, detail: str = "") -> None:
            if on_step:
                on_step(num, name, status, detail)

        # ── Step 1: データ確認 + ランク正規化・推定 ───────────────
        rank_result = self.rank_estimator.estimate_from_lead(
            lead, enable_llm=enable_rank_estimation
        )
        lead = lead.copy()
        lead["lead_rank"] = rank_result["rank"]
        lead["_rank_method"] = rank_result["method"]
        lead["_rank_confidence"] = rank_result["confidence"]
        lead["_rank_original"] = rank_result["original"]

        rank_detail = (
            f"ランク: {rank_result['rank']}"
            + (f" ← {rank_result['original']}" if rank_result["original"] != rank_result["rank"] else "")
            + f" ({rank_result['method']})"
        )
        _step(1, "データ確認", "done",
              f"{visitor} / {company} / {rank_detail}")

        quality = check_lead_quality(lead)

        # ── Step 2: 関心製品で技術資料を検索 ──────────────────────
        _step(2, "ベクトルDB検索", "running", "関心製品で技術資料を検索中...")
        products = parse_interested_products(str(lead.get("interested_products", "")))
        tech_query = " ".join(products) if products else company
        tech_results = []
        ref_tech_docs = []

        if tech_query and self.vectordb.is_index_built():
            tech_results = self.vectordb.search_tech_docs(tech_query, top_k=3)
            ref_tech_docs = list({r["metadata"].get("source_file", "") for r in tech_results})
            logger.info(f"  技術資料検索: {len(tech_results)}件ヒット ({', '.join(ref_tech_docs)})")
            has_parent_count = sum(1 for r in tech_results if r.get("has_parent"))
            _step(2, "ベクトルDB検索", "done",
                  f"{len(tech_results)}件ヒット"
                  + (f" (親チャンクで拡張: {has_parent_count}件)" if has_parent_count > 0 else "")
                  + (f": {', '.join(ref_tech_docs[:3])}" if ref_tech_docs else ""))
        else:
            logger.warning("  技術資料: インデックス未構築またはクエリなし")
            _step(2, "ベクトルDB検索", "warning", "インデックス未構築またはクエリなし")

        tech_context = "\n\n".join(r["text"][:500] for r in tech_results)

        # ── Step 3: CRM情報の取得（優先順位あり）────────────────────
        crm_context = ""
        crm_structured: Optional[Dict[str, Any]] = None
        ref_crm: List[str] = []
        crm_match_score = 0
        crm_deal_stage = ""
        crm_source = "none"
        crm_vdb_results: List[Dict] = []

        _step(3, "CRM照合", "running", "過去商談を照合中...")

        if crm_df is not None and not crm_df.empty:
            # ── 優先度1: CRM CSV マッチング（Email優先→会社名ファジー）──
            crm_structured, crm_match_score = self._match_crm_from_csv(lead, crm_df)

            if crm_structured:
                crm_deal_stage = crm_structured.get("lifecycle_stage", "")
                crm_source = "csv"
                method = crm_structured.get("match_method", "")
                method_label = "メール一致" if method == "email" else "会社名マッチ"
                logger.info(
                    f"  CRM CSV マッチ: スコア={crm_match_score}, "
                    f"方法={method}, ステージ={crm_deal_stage}"
                )
                _step(3, "CRM照合", "done",
                      f"紐付け: {method_label} / スコア: {crm_match_score}")
            else:
                logger.info("  CRM CSV: マッチなし（vectordbにフォールバック）")
                crm_context, ref_crm, crm_source, crm_vdb_results = self._search_crm_from_vectordb(company)
                if crm_vdb_results:
                    _step(3, "CRM照合", "done", f"vectordb: {len(crm_vdb_results)}件ヒット")
                else:
                    _step(3, "CRM照合", "warning", "マッチする過去商談なし（新規顧客）")

        else:
            # ── 優先度2: vectordb 検索（従来通り）──────────────────
            crm_context, ref_crm, crm_source, crm_vdb_results = self._search_crm_from_vectordb(company)
            if crm_vdb_results:
                _step(3, "CRM照合", "done", f"vectordb: {len(crm_vdb_results)}件ヒット")
            else:
                _step(3, "CRM照合", "warning", "マッチする過去商談なし（新規顧客）")

        # ── Step 4: Web検索 ──────────────────────────────────────
        web_info: Dict[str, Any] = {"success": False, "summary": "", "results": []}
        if enable_web_search and company:
            _step(4, "Web検索", "running", f"{company}の事業情報・最新動向を検索中...")
            try:
                web_info = self.web_searcher.search_company(company)
                if web_info["success"]:
                    profile_cnt = sum(1 for r in web_info["results"] if r.get("section") == "profile")
                    news_cnt = sum(1 for r in web_info["results"] if r.get("section") == "news")
                    _step(4, "Web検索", "done",
                          f"事業情報 {profile_cnt}件 / 最新動向 {news_cnt}件")
                else:
                    error_msg = web_info.get("error", "不明なエラー")
                    logger.warning(f"  Web検索: 結果なし — {error_msg}")
                    _step(4, "Web検索", "warning",
                          f"結果なし（{error_msg}）")
            except Exception as e:
                logger.error(f"  Web検索 例外: {type(e).__name__}: {e}")
                _step(4, "Web検索", "warning", f"検索エラー: {type(e).__name__}: {e}")
                web_info = {"success": False, "summary": "", "results": []}
        else:
            _step(4, "Web検索", "skip", "スキップ（無効）")

        # ── Step 5: 情報充足確認 ─────────────────────────────────
        if quality["errors"]:
            _step(5, "情報充足確認", "warning",
                  f"スコア: {quality['score']}% / エラー: {' / '.join(quality['errors'])}")
        elif quality["warnings"]:
            _step(5, "情報充足確認", "warning",
                  f"スコア: {quality['score']}% / 不足: {' / '.join(quality['warnings'])}")
        else:
            _step(5, "情報充足確認", "done", f"充足度: {quality['score']}%")

        # ── Step 6: メール生成 ──────────────────────────────────
        _step(6, "メール生成中", "running", "LLMがメール文を生成中...")
        audio_context = _build_audio_context(transcript, extracted_needs)
        email = self.email_gen.generate(
            lead=lead,
            tech_context=tech_context,
            crm_context=crm_context,
            crm_structured=crm_structured,
            exhibition_info=exhibition_info,
            web_context=web_info.get("summary", ""),
            sender_company=sender_company,
            sender_name=sender_name,
            audio_context=audio_context,
            product_urls=product_urls,
        )
        _step(6, "メール生成中", "done", f"件名: {email.get('subject', '')[:40]}")
        _step(7, "完了", "done", "メール生成が完了しました")

        # ── 参照チャンクの構築 ──────────────────────────────────
        retrieved_tech_chunks: List[Dict] = []
        if tech_results:
            max_s = max(r.get("score", 0) for r in tech_results) or 1.0
            for r in tech_results:
                raw = r.get("score", 0)
                norm = raw / max_s
                label = "高" if norm >= 0.7 else ("中" if norm >= 0.4 else "低")
                retrieved_tech_chunks.append({
                    "source_file": r["metadata"].get("source_file", ""),
                    "source_type": r["metadata"].get("source_type", "tech_doc"),
                    "score": raw,
                    "score_label": label,
                    "has_parent": r.get("has_parent", False),
                    "text_preview": r.get("child_text", r["text"])[:300],
                })

        retrieved_crm_chunks: List[Dict] = []
        if crm_source == "csv" and crm_structured:
            retrieved_crm_chunks = [{
                "source_file": "",
                "source_type": "crm_csv",
                "company_name": crm_structured.get("matched_company", ""),
                "deal_stage": crm_structured.get("lifecycle_stage", ""),
                "lead_status": crm_structured.get("lead_status", ""),
                "match_method": crm_structured.get("match_method", ""),
                "match_score": crm_match_score,
                "text_preview": "",
            }]
        elif crm_vdb_results:
            max_s = max(r.get("score", 0) for r in crm_vdb_results) or 1.0
            for r in crm_vdb_results:
                raw = r.get("score", 0)
                norm = raw / max_s
                label = "高" if norm >= 0.7 else ("中" if norm >= 0.4 else "低")
                retrieved_crm_chunks.append({
                    "source_file": r["metadata"].get("source_file", ""),
                    "source_type": "crm_record",
                    "company_name": r["metadata"].get("company_name", ""),
                    "deal_stage": r["metadata"].get("deal_stage", ""),
                    "lead_status": "",
                    "match_method": "vectordb",
                    "match_score": 0,
                    "score": raw,
                    "score_label": label,
                    "text_preview": r["text"][:300],
                })

        # ── Step 7: 結果を返す ──────────────────────────────────
        return {
            "lead_id": lead.get("lead_id", ""),
            "visitor_name": visitor,
            "company_name": company,
            "lead_rank": lead.get("lead_rank", ""),
            "email_to": lead.get("email", ""),
            "subject": email.get("subject", ""),
            "body": email.get("body", ""),
            "cta": email.get("cta", ""),
            "ref_tech_docs": ref_tech_docs,
            "ref_crm": ref_crm,
            "crm_match_score": crm_match_score,
            "crm_deal_stage": crm_deal_stage,
            "crm_source": crm_source,
            "quality_score": quality["score"],
            "retrieved_tech_chunks": retrieved_tech_chunks,
            "retrieved_crm_chunks": retrieved_crm_chunks,
            "crm_structured": crm_structured,
            "web_search_results": web_info.get("results", []),
            "rank_info": rank_result,
        }

    def _match_crm_from_csv(
        self,
        lead: Dict[str, Any],
        crm_df: pd.DataFrame,
    ) -> tuple:
        """
        CRM CSV からリード情報で2段階マッチングを行い、構造化データを返す。

        Parameters
        ----------
        lead : Dict[str, Any]
            リードデータ辞書（email / company_name を参照）
        crm_df : pd.DataFrame
            CRM商談データ（標準フィールド名に変換済み）

        Returns
        -------
        tuple[Optional[Dict], int]
            (crm_structured辞書, マッチスコア)
            マッチなしの場合は (None, 0)
        """
        # CRMMatcher を遅延インポート（循環インポート防止）
        from src.crm_matcher import CRMMatcher

        matcher = CRMMatcher()
        matched = matcher.match(lead, crm_df)

        if not matched:
            return None, 0

        score = matched.get("_crm_match_score", 0)
        match_method = matched.get("_crm_match_method", "")

        # HubSpot標準フィールドをcrm_structured辞書に整理
        crm_structured = {
            "last_activity_date": str(matched.get("last_activity_date", "")),
            "lifecycle_stage":    str(matched.get("lifecycle_stage", "")),
            "lead_status":        str(matched.get("lead_status", "")),
            "contact_owner":      str(matched.get("contact_owner", "")),
            "original_source":    str(matched.get("original_source", "")),
            "create_date":        str(matched.get("create_date", "")),
            "record_id":          str(matched.get("record_id", "")),
            "first_name":         str(matched.get("first_name", "")),
            "last_name":          str(matched.get("last_name", "")),
            "phone":              str(matched.get("phone", "")),
            "job_title":          str(matched.get("job_title", "")),
            "matched_company":    str(matched.get("company_name", "")),
            "match_method":       match_method,
        }

        return crm_structured, score

    def _search_crm_from_vectordb(self, company: str) -> tuple:
        """
        vectordb から会社名でCRM記録を検索する（従来のフォールバック処理）。

        Parameters
        ----------
        company : str
            リードの会社名

        Returns
        -------
        tuple[str, List[str], str, List[Dict]]
            (crm_context テキスト, 参照ファイルリスト, "vectordb" or "none", 生検索結果)
        """
        if not company or not self.vectordb.is_index_built():
            return "", [], "none", []

        crm_results = self.vectordb.search_crm(company, top_k=2)
        ref_crm = list({r["metadata"].get("source_file", "") for r in crm_results})

        if crm_results:
            logger.info(f"  vectordb CRM検索: {len(crm_results)}件ヒット ({', '.join(ref_crm)})")
        else:
            logger.info("  vectordb CRM検索: 関連記録なし")

        crm_context = "\n\n".join(r["text"][:500] for r in crm_results)
        source = "vectordb" if crm_results else "none"
        return crm_context, ref_crm, source, crm_results

    def process_all_leads(
        self,
        leads_df: pd.DataFrame,
        crm_df: Optional[pd.DataFrame] = None,
        exhibition_info: Optional[Dict[str, Any]] = None,
        enable_web_search: bool = True,
        enable_rank_estimation: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        DataFrameの全リードを処理してメール生成結果のリストを返す。

        Parameters
        ----------
        leads_df : pd.DataFrame
            全リードデータ
        crm_df : pd.DataFrame, optional
            CRM商談データ。None の場合は vectordb 検索を使用。

        Returns
        -------
        List[Dict[str, Any]]
            各リードの処理結果リスト
        """
        total = len(leads_df)
        logger.info(f"全リード処理開始: {total}件 (CRM: {'CSV' if crm_df is not None else 'vectordb'})")

        results = []
        for idx, row in leads_df.iterrows():
            lead = row.to_dict()
            logger.info(f"[{idx + 1}/{total}] ─────────────────────────")
            try:
                result = self.process_lead(
                    lead, crm_df=crm_df, exhibition_info=exhibition_info,
                    enable_web_search=enable_web_search,
                    enable_rank_estimation=enable_rank_estimation,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"  エラー ({lead.get('visitor_name')}): {e}")
                results.append({
                    "lead_id": lead.get("lead_id", ""),
                    "visitor_name": lead.get("visitor_name", ""),
                    "company_name": lead.get("company_name", ""),
                    "lead_rank": lead.get("lead_rank", ""),
                    "email_to": lead.get("email", ""),
                    "subject": "ERROR",
                    "body": str(e),
                    "cta": "",
                    "ref_tech_docs": [],
                    "ref_crm": [],
                    "crm_match_score": 0,
                    "crm_deal_stage": "",
                    "crm_source": "none",
                    "quality_score": check_lead_quality(lead).get("score", 0),
                    "retrieved_tech_chunks": [],
                    "retrieved_crm_chunks": [],
                    "crm_structured": None,
                })

        logger.info(f"全リード処理完了: {len(results)}件")
        return results


def _build_audio_context(transcript: str, needs: Optional[Dict[str, Any]]) -> str:
    """音声コンテキスト文字列を組み立てる（メール生成の最優先情報として使用）"""
    if not transcript and not needs:
        return ""
    parts = []
    if needs and any(needs.get(k) for k in ("issues", "needs", "budget", "decision_maker", "temperature")):
        parts.append("【音声から抽出したニーズ・課題】")
        for key, label in [
            ("issues", "課題"),
            ("needs", "ニーズ"),
            ("budget", "予算感"),
            ("decision_maker", "決裁者"),
            ("temperature", "温度感"),
        ]:
            if needs.get(key):
                parts.append(f"- {label}: {needs[key]}")
        if needs.get("summary"):
            parts.append(f"- 要約: {needs['summary']}")
    if transcript:
        parts.append(f"\n【会話録音の文字起こし（参考）】\n{transcript[:2000]}")
    return "\n".join(parts)


# -------------------------
# 動作確認用スクリプト
# -------------------------
if __name__ == "__main__":
    import sys
    from src.config import Config
    from src.vectordb import VectorDBManager
    from src.email_generator import EmailGenerator
    from src.utils import load_leads
    import pandas as pd

    print("=" * 60)
    print("FollowUpAgent 動作確認")
    print("=" * 60)

    try:
        Config.validate()
    except EnvironmentError as e:
        print(f"設定エラー: {e}")
        sys.exit(1)

    print("\n[1] VectorDB 初期化")
    db = VectorDBManager()
    if not db.is_index_built():
        print("  インデックスを構築します...")
        db.build_index(Config.TECH_DOCS_DIR, Config.CRM_RECORDS_DIR)
    else:
        print("  既存インデックスを使用します")

    print("\n[2] EmailGenerator 初期化")
    email_gen = EmailGenerator()

    agent = FollowUpAgent(vectordb_manager=db, email_generator=email_gen)

    print("\n[3] リードデータ読み込み")
    leads_df = load_leads(Config.LEADS_CSV_PATH)
    lead = leads_df.iloc[0].to_dict()

    # CRM CSVの読み込みテスト
    crm_df = None
    crm_path = "data/crm_demo.csv"
    if __import__("os").path.exists(crm_path):
        from src.utils import auto_map_columns, apply_column_mapping
        raw_crm = pd.read_csv(crm_path, dtype=str, encoding="utf-8-sig").fillna("")
        mapping = auto_map_columns(list(raw_crm.columns), {**Config.CRM_REQUIRED_FIELDS, **Config.CRM_OPTIONAL_FIELDS})
        crm_df = apply_column_mapping(raw_crm, mapping)
        print(f"  CRM CSV読み込み: {len(crm_df)}件")

    print(f"  対象: {lead['visitor_name']} ({lead['company_name']}) / ランク {lead['lead_rank']}")

    print("\n[4] メール生成（CRM CSV連携）")
    result = agent.process_lead(lead, crm_df=crm_df)

    print("\n" + "=" * 60)
    print("【生成結果】")
    print("=" * 60)
    print(f"\n■ 件名\n{result['subject']}")
    print(f"\n■ CRM情報")
    print(f"  取得元: {result['crm_source']}")
    print(f"  マッチスコア: {result['crm_match_score']}")
    print(f"  商談ステージ: {result['crm_deal_stage']}")
    print(f"\n■ 参照技術資料: {', '.join(result['ref_tech_docs']) or 'なし'}")
    print(f"■ 参照CRM記録: {', '.join(result['ref_crm']) or 'なし'}")
    print("\n" + "=" * 60)
    print("動作確認完了")
    print("=" * 60)
