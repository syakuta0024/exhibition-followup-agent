"""
設定管理モジュール

環境変数の読み込みとアプリケーション全体の設定値を管理する。
"""

import os
from dotenv import load_dotenv

# .envファイルを読み込む
load_dotenv()


class Config:
    """アプリケーション設定クラス"""

    # APIキー（OpenAI APIのみ使用）
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # データパス
    DATA_DIR: str = "data"
    LEADS_CSV_PATH: str = os.path.join(DATA_DIR, "leads.csv")
    TECH_DOCS_DIR: str = os.path.join(DATA_DIR, "tech_documents")
    CRM_RECORDS_DIR: str = os.path.join(DATA_DIR, "crm_records")

    # ChromaDBの保存先
    CHROMA_DB_DIR: str = "chroma_db"
    CHROMA_COLLECTION_NAME: str = "ntx_knowledge_base"

    # LLMモデル設定（LLM・Embedding ともに OpenAI を使用）
    LLM_MODEL: str = "gpt-5.4-nano"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # RAG設定
    RETRIEVER_TOP_K: int = 5       # ベクトル検索で取得する上位件数
    CHUNK_SIZE: int = 600          # テキスト分割のチャンクサイズ（日本語最適化）
    CHUNK_OVERLAP: int = 150       # チャンク間のオーバーラップ文字数

    # ---------------------------------------------------------------
    # リードCSVカラムマッピング定義
    # RX Japan Lead Manager / Q-PASS / Sansan 等の異なるカラム名に対応。
    # 未マッピングのカスタム質問列は extra_ プレフィックスで自動保持され、
    # メール生成のコンテキストに自動活用されます（Lead Managerのカスタム質問に対応）。
    # ---------------------------------------------------------------

    # 必須フィールド: メール生成に最低限必要な項目
    REQUIRED_FIELDS: dict = {
        "visitor_name": {
            "label": "氏名",
            "description": "来場者の氏名（姓名は1列・非分離）",
            "候補カラム名": ["氏名", "名前", "お名前", "担当者名", "来場者名", "Full name", "Name", "visitor_name"],
        },
        "company_name": {
            "label": "会社名",
            "description": "来場者の所属企業名",
            "候補カラム名": ["会社名", "所属会社", "企業名", "組織名", "Company name", "Company", "company_name"],
        },
        "email": {
            "label": "メールアドレス",
            "description": "連絡先メールアドレス",
            "候補カラム名": ["メールアドレス", "メール", "メアド", "E-mail", "Email", "email", "mail"],
        },
    }

    # 任意フィールド: あれば活用する項目（Lead Manager固定列 + カスタム質問の共通候補）
    OPTIONAL_FIELDS: dict = {
        "department": {
            "label": "所属部署",
            "候補カラム名": ["所属部署", "部署", "部署名", "Department", "department"],
        },
        "job_title": {
            "label": "役職",
            "候補カラム名": ["役職", "職位", "Job title", "Title", "Position", "job_title", "肩書"],
        },
        "phone": {
            "label": "電話番号",
            "候補カラム名": ["電話番号", "電話", "TEL", "Phone", "Tel", "phone"],
        },
        "address": {
            "label": "住所",
            "候補カラム名": ["住所", "会社所在地", "所在地", "Address", "address"],
        },
        "lead_rank": {
            "label": "評価",
            "候補カラム名": ["評価", "ランク", "商談確度", "Rating", "Rank", "lead_rank", "リード評価"],
        },
        "memo": {
            "label": "メモ",
            "候補カラム名": ["メモ", "フリーコメント", "備考", "Notes", "Memo", "memo", "コメント"],
        },
        "visit_date": {
            "label": "来場日",
            "候補カラム名": ["来場日", "訪問日", "登録日", "Visit date", "Date", "visit_date"],
        },
        "interested_products": {
            "label": "関心製品",
            "候補カラム名": ["関心製品", "Products of interest", "興味", "関心", "Products", "interested_products", "ご興味"],
        },
        "future_requests": {
            "label": "今後のご要望",
            "候補カラム名": ["要望", "ご要望", "今後の希望", "Requests", "future_requests"],
        },
    }

    # ---------------------------------------------------------------
    # CRM CSVカラムマッピング定義
    # HubSpot Contacts エクスポート形式に準拠。
    # email（完全一致）→ company_name（ファジーマッチ）の2段階で紐付けを行う。
    # ---------------------------------------------------------------

    # CRM必須フィールド: リードとの紐付けキー（いずれか1つ以上をマッピング）
    CRM_REQUIRED_FIELDS: dict = {
        "email": {
            "label": "メールアドレス",
            "description": "第1紐付けキー（完全一致でリードと照合）",
            "候補カラム名": ["Email", "メールアドレス", "メール", "E-mail", "email", "メアド"],
        },
        "company_name": {
            "label": "会社名",
            "description": "第2紐付けキー（ファジーマッチングでリードと照合）",
            "候補カラム名": ["Company name", "会社名", "企業名", "Company", "取引先名", "company_name"],
        },
    }

    # CRM任意フィールド: HubSpot Contacts標準エクスポートフィールド
    CRM_OPTIONAL_FIELDS: dict = {
        "record_id": {
            "label": "Record ID",
            "候補カラム名": ["Record ID", "record_id", "コンタクトID", "ID"],
        },
        "first_name": {
            "label": "First name",
            "候補カラム名": ["First name", "first_name", "名", "ファーストネーム"],
        },
        "last_name": {
            "label": "Last name",
            "候補カラム名": ["Last name", "last_name", "姓", "ラストネーム"],
        },
        "phone": {
            "label": "Phone number",
            "候補カラム名": ["Phone number", "phone", "Phone", "電話番号", "TEL"],
        },
        "job_title": {
            "label": "Job title",
            "候補カラム名": ["Job title", "job_title", "役職", "Title", "職位"],
        },
        "lifecycle_stage": {
            "label": "Lifecycle stage",
            "候補カラム名": ["Lifecycle stage", "lifecycle_stage", "ライフサイクルステージ", "フェーズ", "Stage"],
        },
        "lead_status": {
            "label": "Lead status",
            "候補カラム名": ["Lead status", "lead_status", "リードステータス", "ステータス"],
        },
        "original_source": {
            "label": "Original source",
            "候補カラム名": ["Original source", "original_source", "ソース", "流入元", "獲得経路"],
        },
        "contact_owner": {
            "label": "Contact owner",
            "候補カラム名": ["Contact owner", "contact_owner", "担当者", "Owner", "担当営業"],
        },
        "create_date": {
            "label": "Create date",
            "候補カラム名": ["Create date", "create_date", "登録日", "作成日"],
        },
        "last_activity_date": {
            "label": "Last activity date",
            "候補カラム名": ["Last activity date", "last_activity_date", "最終活動日", "最終接触日"],
        },
    }

    @classmethod
    def validate(cls) -> bool:
        """
        必須設定値が揃っているか検証する。

        Returns
        -------
        bool
            全ての必須キーが設定されていれば True
        """
        missing = []

        if not cls.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")

        if missing:
            raise EnvironmentError(
                f"以下の環境変数が設定されていません: {', '.join(missing)}\n"
                ".env ファイルに設定してください。"
            )

        return True

    @classmethod
    def get_llm_config(cls) -> dict:
        """
        LLM初期化用の設定辞書を返す。

        Returns
        -------
        dict
            モデル名・APIキーを含む設定辞書
        """
        return {
            "model": cls.LLM_MODEL,
            "openai_api_key": cls.OPENAI_API_KEY,
        }
