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
    # CSVカラムマッピング定義
    # 様々な展示会管理ツール（Lead Manager, Q-PASS, Sansan等）の
    # 異なるカラム名に対応するための候補リスト
    # ---------------------------------------------------------------

    # 必須フィールド: メール生成に最低限必要な項目
    REQUIRED_FIELDS: dict = {
        "visitor_name": {
            "label": "氏名",
            "description": "来場者の氏名",
            "候補カラム名": ["氏名", "名前", "お名前", "Name", "visitor_name", "担当者名", "来場者名"],
        },
        "company_name": {
            "label": "会社名",
            "description": "来場者の所属企業名",
            "候補カラム名": ["会社名", "企業名", "Company", "company_name", "組織名", "所属"],
        },
        "email": {
            "label": "メールアドレス",
            "description": "連絡先メールアドレス",
            "候補カラム名": ["メール", "Email", "email", "E-mail", "メールアドレス", "mail"],
        },
    }

    # 任意フィールド: あれば活用する項目
    OPTIONAL_FIELDS: dict = {
        "department": {
            "label": "部署",
            "候補カラム名": ["部署", "部署名", "Department", "department", "所属部署"],
        },
        "job_title": {
            "label": "役職",
            "候補カラム名": ["役職", "職位", "Title", "job_title", "肩書"],
        },
        "lead_rank": {
            "label": "商談確度",
            "候補カラム名": ["確度", "ランク", "Rank", "lead_rank", "評価", "商談確度", "リード評価"],
        },
        "interested_products": {
            "label": "関心製品",
            "候補カラム名": ["関心製品", "興味", "関心", "Products", "interested_products", "ご興味", "展示物"],
        },
        "memo": {
            "label": "営業メモ",
            "候補カラム名": ["メモ", "備考", "Notes", "memo", "コメント", "会話メモ", "商談メモ", "ノート"],
        },
        "future_requests": {
            "label": "今後のご要望",
            "候補カラム名": ["要望", "ご要望", "Requests", "future_requests", "今後の希望"],
        },
        "visit_date": {
            "label": "来場日",
            "候補カラム名": ["来場日", "訪問日", "Date", "visit_date", "日付"],
        },
        "phone": {
            "label": "電話番号",
            "候補カラム名": ["電話", "TEL", "Phone", "phone", "電話番号"],
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
