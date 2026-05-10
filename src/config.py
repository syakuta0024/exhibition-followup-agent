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
    # APIコスト見積もり設定（一括生成前の確認ダイアログ用）
    # gpt-5.4-nano 料金 (2026年4月時点)
    # ---------------------------------------------------------------
    LLM_PRICE_INPUT_PER_1M: float = 0.20    # $/1M 入力トークン
    LLM_PRICE_OUTPUT_PER_1M: float = 1.25   # $/1M 出力トークン
    # 1リードあたりの推定トークン数（プロンプト構造から算出）
    # 入力: システムプロンプト(~400) + リードデータ(~200) + RAG3チャンク(~800)
    #       + CRM(~200) + Web検索(~300) + 展示会情報+指示(~300) = ~2200
    EST_INPUT_TOKENS_PER_LEAD: int = 2200
    # 出力: メール本文・件名・CTA（ランク平均 ~350字 → ~350トークン）
    EST_OUTPUT_TOKENS_PER_LEAD: int = 400
    # ランク推定LLM（有効時の追加入力トークン、出力は1トークンなので無視）
    EST_RANK_EXTRA_INPUT_TOKENS: int = 300
    # 1リードあたりの推定処理時間（秒）
    EST_SECONDS_PER_LEAD_BASE: float = 8.0   # メール生成のみ
    EST_SECONDS_PER_LEAD_WEB: float = 3.0    # Web検索追加分

    # ---------------------------------------------------------------
    # 音声文字起こし（Whisper API）
    # ---------------------------------------------------------------
    WHISPER_MODEL: str = "whisper-1"
    WHISPER_PRICE_PER_MIN: float = 0.006          # $/分
    WHISPER_MAX_FILE_MB: int = 25                 # Whisper APIの上限（MB）
    AUDIO_TIMESTAMP_TOLERANCE_MINUTES: int = 5   # タイムスタンプ許容誤差（分）。展示会は5〜10分で来場者が入れ替わるため5分に設定。
    AUDIO_RED_FLAG_WARNING_THRESHOLD: float = 0.30  # 赤フラグ率の警告閾値

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
            "候補カラム名": [
                "評価", "ランク", "商談確度", "Rating", "Rank", "lead_rank", "リード評価",
                "リードランク", "Lead Rank", "Grade", "グレード",
            ],
        },
        "memo": {
            "label": "メモ",
            "候補カラム名": [
                "メモ", "フリーコメント", "備考", "Notes", "Memo", "memo", "コメント",
                "Comment", "Remarks", "Note", "対応状況",
            ],
        },
        "visit_date": {
            "label": "来場日",
            "候補カラム名": [
                "来場日", "訪問日", "登録日", "Visit date", "Date", "visit_date",
                "来場時刻", "来場日時", "VisitDate", "Visit Date", "訪問日時",
            ],
        },
        "interested_products": {
            "label": "関心製品",
            "候補カラム名": [
                "関心製品", "Products of interest", "興味", "関心", "Products", "interested_products", "ご興味",
                "興味製品", "製品", "Product", "興味のある商材", "興味のある製品", "商材",
            ],
        },
        "future_requests": {
            "label": "今後のご要望",
            "候補カラム名": [
                "要望", "ご要望", "今後の希望", "Requests", "future_requests",
                "Request", "Needs", "ニーズ", "課題",
            ],
        },
        "rep_name": {
            "label": "担当者名（ブース担当営業）",
            "候補カラム名": [
                "担当者名", "担当者", "営業担当", "担当営業",
                "ブース担当者", "対応者",
                "Sales_Rep", "sales_rep", "Rep", "rep",
                "Salesperson", "Staff", "Assigned_To",
                "Sales Rep", "フォロー担当", "スキャン担当者",
            ],
        },
        "scan_time": {
            "label": "スキャン時刻",
            "候補カラム名": [
                "スキャン時刻", "スキャン日時", "来場時刻", "受付時刻",
                "スキャン時間", "QRスキャン時刻",
                "Scan_Time", "scan_time", "Visit_Time",
                "Timestamp", "timestamp", "Scanned_At",
                "Scan Time", "Time", "時間",
            ],
        },
        "lead_id": {
            "label": "リードID",
            "候補カラム名": ["ID", "LeadID", "Lead ID", "lead_id", "No.", "No", "番号"],
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

    # ---------------------------------------------------------------
    # 展示会情報フィールド定義
    # CSVアップロード時にユーザーが入力し、全リードのメール生成に共通で使用される。
    # 空欄の場合はフォールバック値（「展示会」等）が使われる。
    # ---------------------------------------------------------------
    EXHIBITION_INFO_FIELDS: dict = {
        "exhibition_name": {
            "label": "展示会名",
            "placeholder": "例: 第35回 日本ものづくりワールド 2026",
            "required": False,
        },
        "exhibition_date": {
            "label": "開催日",
            "placeholder": "例: 2026年4月10日〜12日",
            "required": False,
        },
        "exhibition_venue": {
            "label": "会場",
            "placeholder": "例: 東京ビッグサイト",
            "required": False,
        },
    }

    # ---------------------------------------------------------------
    # フィールド日本語ラベル
    # ---------------------------------------------------------------
    FIELD_LABELS: dict = {
        "visitor_name":        "氏名",
        "company_name":        "会社名",
        "email":               "メールアドレス",
        "lead_rank":           "ランク",
        "visit_date":          "来場日",
        "interested_products": "関心製品",
        "memo":                "メモ",
        "future_requests":     "今後のご要望",
        "rep_name":            "担当営業",
        "scan_time":           "スキャン時刻",
        "department":          "部署",
        "job_title":           "役職",
        "lead_id":             "リードID",
        "phone":               "電話番号",
        "address":             "住所",
    }

    @classmethod
    def get_field_label(cls, field: str) -> str:
        """標準フィールド名を日本語ラベルに変換。未登録の場合はそのまま返す。"""
        return cls.FIELD_LABELS.get(field, field)

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
