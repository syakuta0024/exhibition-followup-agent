"""
フォローアップエージェントモジュール

LangChain エージェントは使わず、シンプルなオーケストレーター設計を採用。
VectorDBManager で関連情報を検索し、EmailGenerator でメールを生成する
パイプラインを提供する。
OpenAI GPT をLLMとして使用し、リードデータ・技術資料・CRM記録をもとに
各リードへの最適なフォローアップ戦略を立案し、メール生成へ橋渡しする。
"""

from typing import Any, Dict, List

import pandas as pd

from src.utils import setup_logger, parse_interested_products

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
        logger.info("FollowUpAgent 初期化完了")

    def process_lead(self, lead: Dict[str, Any]) -> Dict[str, Any]:
        """
        1件のリードを処理してフォローアップメールを生成する。

        処理フロー:
        1. interested_products でベクトルDB検索（技術資料）
        2. company_name でベクトルDB検索（CRM記録）
        3. EmailGenerator でメール生成
        4. 結果を辞書で返す

        Parameters
        ----------
        lead : Dict[str, Any]
            leads.csvの1行分のデータ（辞書形式）

        Returns
        -------
        Dict[str, Any]
            {
              'lead_id', 'visitor_name', 'company_name', 'lead_rank',
              'email_to', 'subject', 'body', 'cta',
              'ref_tech_docs': 参照した技術資料ファイル名リスト,
              'ref_crm': 参照したCRMファイル名リスト,
            }
        """
        visitor = lead.get("visitor_name", "")
        company = lead.get("company_name", "")
        logger.info(f"処理開始: {visitor} さん ({company}) [ランク {lead.get('lead_rank')}]")

        # ── Step 1: 関心製品で技術資料を検索 ──────────────────────
        products = parse_interested_products(str(lead.get("interested_products", "")))
        tech_query = " ".join(products) if products else company
        tech_results = []
        ref_tech_docs = []

        if tech_query and self.vectordb.is_index_built():
            tech_results = self.vectordb.search_tech_docs(tech_query, top_k=3)
            ref_tech_docs = list({r["metadata"].get("source_file", "") for r in tech_results})
            logger.info(f"  技術資料検索: {len(tech_results)}件ヒット ({', '.join(ref_tech_docs)})")
        else:
            logger.warning("  技術資料: インデックス未構築またはクエリなし")

        # ── Step 2: 会社名でCRM記録を検索 ────────────────────────
        crm_results = []
        ref_crm = []

        if company and self.vectordb.is_index_built():
            crm_results = self.vectordb.search_crm(company, top_k=2)
            ref_crm = list({r["metadata"].get("source_file", "") for r in crm_results})
            if crm_results:
                logger.info(f"  CRM検索: {len(crm_results)}件ヒット ({', '.join(ref_crm)})")
            else:
                logger.info("  CRM検索: 関連記録なし")

        # 検索結果をテキストに変換（LLMへのコンテキストとして渡す）
        tech_context = "\n\n".join(r["text"][:500] for r in tech_results)
        crm_context = "\n\n".join(r["text"][:500] for r in crm_results)

        # ── Step 3: メール生成 ──────────────────────────────────
        email = self.email_gen.generate(
            lead=lead,
            tech_context=tech_context,
            crm_context=crm_context,
        )

        # ── Step 4: 結果を返す ──────────────────────────────────
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
        }

    def process_all_leads(self, leads_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        DataFrameの全リードを処理してメール生成結果のリストを返す。

        Parameters
        ----------
        leads_df : pd.DataFrame
            全リードデータ

        Returns
        -------
        List[Dict[str, Any]]
            各リードの処理結果リスト
        """
        total = len(leads_df)
        logger.info(f"全リード処理開始: {total}件")

        results = []
        for idx, row in leads_df.iterrows():
            lead = row.to_dict()
            logger.info(f"[{idx + 1}/{total}] ─────────────────────────")
            try:
                result = self.process_lead(lead)
                results.append(result)
            except Exception as e:
                logger.error(f"  エラー ({lead.get('visitor_name')}): {e}")
                # エラーが起きても次のリードへ継続
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
                })

        logger.info(f"全リード処理完了: {len(results)}件")
        return results


# -------------------------
# 動作確認用スクリプト
# -------------------------
if __name__ == "__main__":
    import sys
    from src.config import Config
    from src.vectordb import VectorDBManager
    from src.email_generator import EmailGenerator
    from src.utils import load_leads

    print("=" * 60)
    print("FollowUpAgent 動作確認")
    print("=" * 60)

    # 設定の検証
    try:
        Config.validate()
    except EnvironmentError as e:
        print(f"設定エラー: {e}")
        sys.exit(1)

    # ── VectorDBManager 初期化・インデックス確認 ──
    print("\n[1] VectorDB 初期化")
    db = VectorDBManager()
    if not db.is_index_built():
        print("  インデックスを構築します...")
        db.build_index(Config.TECH_DOCS_DIR, Config.CRM_RECORDS_DIR)
    else:
        print("  既存インデックスを使用します")

    # ── EmailGenerator 初期化 ──
    print("\n[2] EmailGenerator 初期化")
    email_gen = EmailGenerator()

    # ── FollowUpAgent 初期化 ──
    agent = FollowUpAgent(vectordb_manager=db, email_generator=email_gen)

    # ── leads.csv から1件目を読み込み ──
    print("\n[3] リードデータ読み込み")
    leads_df = load_leads(Config.LEADS_CSV_PATH)
    lead = leads_df.iloc[0].to_dict()

    print(f"  対象: {lead['visitor_name']} ({lead['company_name']}) / ランク {lead['lead_rank']}")
    print(f"  関心製品: {lead['interested_products']}")

    # ── メール生成 ──
    print("\n[4] メール生成")
    result = agent.process_lead(lead)

    # ── 結果表示 ──
    print("\n" + "=" * 60)
    print("【生成結果】")
    print("=" * 60)
    print(f"\n■ 件名\n{result['subject']}")
    print(f"\n■ 本文\n{result['body']}")
    print(f"\n■ CTA\n{result['cta']}")
    print(f"\n■ 参照技術資料: {', '.join(result['ref_tech_docs']) or 'なし'}")
    print(f"■ 参照CRM記録: {', '.join(result['ref_crm']) or 'なし'}")
    print("\n" + "=" * 60)
    print("動作確認完了")
    print("=" * 60)
