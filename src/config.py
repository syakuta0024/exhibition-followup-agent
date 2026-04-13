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
