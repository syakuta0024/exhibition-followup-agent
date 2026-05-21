"""
Skills / CLI 共通ビジネスロジック層。

.claude/commands/ の各 Skill から呼ばれる。
戻り値はすべて dict で統一し、呼び出し側でフォーマットする。
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

CONFIG_PATH = Path("cli_config.yaml")
DEFAULT_CONFIG: Dict[str, Any] = {
    "sender_company": "",
    "sender_name": "",
    "default_ranks": ["A", "B", "C"],
    "enable_web_search": True,
    "enable_rank_estimation": True,
    "output_path": "output/emails.csv",
    "leads_csv_path": "data/leads.csv",
    "product_urls": {
        "EdgeGuard": "",
        "DigiMA": "",
        "Sorani": "",
        "FactoryBrain": "",
        "NTX-OCR": "",
        "SmartVision": "",
    },
}


def load_cli_config() -> Dict[str, Any]:
    """cli_config.yaml を読み込む。存在しなければデフォルト値を返す。"""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        return {**DEFAULT_CONFIG, **loaded}
    return DEFAULT_CONFIG.copy()


def save_cli_config(config: Dict[str, Any]) -> None:
    """cli_config.yaml に書き込む。"""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)


def mask_api_key(key: str) -> str:
    """APIキーをマスキングして表示用文字列を返す。末尾4文字のみ表示。"""
    if not key:
        return "未設定 → .env に OPENAI_API_KEY を設定してください"
    if len(key) < 4:
        return "設定済み（短い値）"
    return f"設定済み（***...{key[-4:]}）"


def run_check() -> Dict[str, Any]:
    """
    環境・設定の健全性チェック。

    VectorDBManager をインスタンス化せず（API key なしでも動く）、
    ファイルシステムと環境変数のみを確認する。

    Returns
    -------
    dict
        ok (bool): エラーがなければ True
        items (list): 各チェック項目の {label, status, detail}
    """
    from src.config import Config

    items = []

    # 1. OPENAI_API_KEY
    api_key = Config.OPENAI_API_KEY
    if api_key:
        items.append({"label": "OPENAI_API_KEY", "status": "ok", "detail": "設定済み"})
    else:
        items.append({
            "label": "OPENAI_API_KEY",
            "status": "error",
            "detail": ".env ファイルに OPENAI_API_KEY=sk-... を設定してください",
        })

    # 2. cli_config.yaml
    config = load_cli_config()
    if CONFIG_PATH.exists():
        items.append({"label": "cli_config.yaml", "status": "ok", "detail": "存在"})
    else:
        items.append({"label": "cli_config.yaml", "status": "warning", "detail": "未作成（デフォルト値を使用）"})

    # 3. 送信元会社名
    sender = config.get("sender_company", "")
    if sender:
        items.append({"label": "送信元会社名", "status": "ok", "detail": sender})
    else:
        items.append({
            "label": "送信元会社名",
            "status": "warning",
            "detail": "未設定（generate 実行時に入力が必要）",
        })

    # 4. リードCSV
    leads_path = Path(config.get("leads_csv_path", "data/leads.csv"))
    if leads_path.exists():
        try:
            row_count = sum(1 for _ in open(leads_path, encoding="utf-8", errors="replace")) - 1
            items.append({"label": "リードCSV", "status": "ok", "detail": f"{leads_path} ({max(row_count, 0)}件)"})
        except Exception:
            items.append({"label": "リードCSV", "status": "ok", "detail": str(leads_path)})
    else:
        items.append({
            "label": "リードCSV",
            "status": "warning",
            "detail": f"{leads_path} が見つかりません。data/ フォルダに配置してください",
        })

    # 5. 技術資料ディレクトリ
    tech_dir = Path(Config.TECH_DOCS_DIR)
    if tech_dir.exists():
        md_files = list(tech_dir.glob("*.md"))
        pdf_files = list(tech_dir.glob("*.pdf"))
        if md_files or pdf_files:
            detail_parts = []
            if md_files:
                detail_parts.append(f"{len(md_files)} 件の Markdown")
            if pdf_files:
                detail_parts.append(f"{len(pdf_files)} 件の PDF（VLM でテキスト化）")
            items.append({"label": "技術資料", "status": "ok", "detail": "、".join(detail_parts)})
        else:
            items.append({
                "label": "技術資料",
                "status": "warning",
                "detail": (
                    f"{tech_dir}/ が空です。"
                    "自社の製品技術資料を .md 形式で配置してください。メール生成の品質が向上します"
                ),
            })
    else:
        items.append({
            "label": "技術資料",
            "status": "warning",
            "detail": f"{tech_dir}/ ディレクトリが見つかりません",
        })

    # 6. CRM 記録ディレクトリ
    crm_dir = Path(Config.CRM_RECORDS_DIR)
    if crm_dir.exists():
        items.append({"label": "CRM 記録ディレクトリ", "status": "ok", "detail": str(crm_dir)})
    else:
        items.append({
            "label": "CRM 記録ディレクトリ",
            "status": "warning",
            "detail": f"{crm_dir}/ が見つかりません",
        })

    # 7. ナレッジベース（ChromaDB）— SQLite3 存在確認 + 実チャンク数確認
    chroma_sqlite = Path(Config.CHROMA_DB_DIR) / "chroma.sqlite3"
    if not chroma_sqlite.exists() or chroma_sqlite.stat().st_size <= 1000:
        items.append({
            "label": "ナレッジベース (KB)",
            "status": "warning",
            "detail": "未構築 → /email-workflow の Step 3 でナレッジベース構築を実行してください",
        })
    else:
        try:
            import chromadb as _chromadb
            _client = _chromadb.PersistentClient(path=str(Config.CHROMA_DB_DIR))
            try:
                _col = _client.get_collection(Config.CHROMA_COLLECTION_NAME)
                chunk_count = _col.count()
            except Exception:
                chunk_count = 0
            if chunk_count > 0:
                items.append({
                    "label": "ナレッジベース (KB)",
                    "status": "ok",
                    "detail": f"構築済み（{chunk_count}件のチャンク）",
                })
            else:
                items.append({
                    "label": "ナレッジベース (KB)",
                    "status": "warning",
                    "detail": "KBファイルはあるが空です。/email-workflow Step 3 で再構築してください",
                })
        except Exception:
            items.append({
                "label": "ナレッジベース (KB)",
                "status": "ok",
                "detail": "構築済み（チャンク数確認不可）",
            })

    # 8. Gmail credentials
    creds_path = Path("credentials/credentials.json")
    if creds_path.exists():
        items.append({
            "label": "Gmail credentials",
            "status": "ok",
            "detail": "credentials.json 配置済み（Gmail下書き機能が利用可能）",
        })
    else:
        items.append({
            "label": "Gmail credentials",
            "status": "warning",
            "detail": "credentials/credentials.json を Google Cloud Console からダウンロードして配置してください。詳細は system_overview.md の「6-5. Gmail API」セクションを参照",
        })

    any_error = any(i["status"] == "error" for i in items)
    return {"ok": not any_error, "items": items}


def run_build_kb(
    tech_docs_dir: Optional[str] = None,
    crm_records_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    VectorDB インデックスを構築する。

    Returns
    -------
    dict
        ok (bool), message (str), chunk_count (int)
    """
    from src.config import Config
    from src.vectordb import VectorDBManager

    tech_dir = tech_docs_dir or Config.TECH_DOCS_DIR
    crm_dir = crm_records_dir or Config.CRM_RECORDS_DIR

    try:
        Config.validate()
    except EnvironmentError as e:
        return {"ok": False, "message": str(e), "chunk_count": 0}

    try:
        vectordb = VectorDBManager(
            persist_dir=Config.CHROMA_DB_DIR,
            collection_name=Config.CHROMA_COLLECTION_NAME,
        )
        vectordb.build_index(tech_dir, crm_dir)
        chunk_count = vectordb.vectorstore._collection.count()
    except Exception as e:
        return {"ok": False, "message": f"構築エラー: {e}", "chunk_count": 0}

    return {
        "ok": True,
        "message": f"ナレッジベース構築完了（総チャンク数: {chunk_count}）",
        "chunk_count": chunk_count,
    }


def run_load_leads(
    csv_path: Optional[str] = None,
    ranks: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    リードCSVを読み込み、カラムマッピングを実行して統計情報を返す。

    Returns
    -------
    dict
        ok (bool), total (int), total_all (int), by_rank (dict),
        columns_mapped (dict), leads_df (DataFrame), message (str)
    """
    from src.config import Config
    from src.utils import load_leads, auto_map_columns, apply_column_mapping, filter_leads_by_rank

    config = load_cli_config()
    path = csv_path or config.get("leads_csv_path", "data/leads.csv")

    try:
        df = load_leads(path)
    except FileNotFoundError as e:
        return {
            "ok": False,
            "message": str(e),
            "total": 0,
            "total_all": 0,
            "by_rank": {},
            "columns_mapped": {},
            "leads_df": None,
        }

    all_fields = {**Config.REQUIRED_FIELDS, **Config.OPTIONAL_FIELDS}
    mapping = auto_map_columns(df.columns.tolist(), all_fields)
    df = apply_column_mapping(df, mapping)

    target_ranks = ranks or config.get("default_ranks", ["A", "B", "C"])
    filtered = filter_leads_by_rank(df, target_ranks)

    by_rank = df["lead_rank"].value_counts().to_dict() if "lead_rank" in df.columns else {}
    columns_mapped = {k: v for k, v in mapping.items() if v is not None}

    return {
        "ok": True,
        "total": len(filtered),
        "total_all": len(df),
        "by_rank": by_rank,
        "columns_mapped": columns_mapped,
        "leads_df": filtered,
        "message": f"{len(filtered)}件のリードを読み込みました（全{len(df)}件中、ランク{target_ranks}）",
    }


def _manage_output_path(output_naming: str, base_path: str) -> str:
    """
    output_naming に従い出力パスを決定する。

    "timestamp" 時は既存の base_path を legacy/ に退避し、
    タイムスタンプ付きの新パスを返す。"overwrite" 時は base_path をそのまま返す。
    既存ファイルが存在しない場合は退避しない。
    """
    from src.utils import setup_logger
    _log = setup_logger(__name__)

    if output_naming != "timestamp":
        return base_path

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.dirname(base_path) or "output"
    stem, ext = os.path.splitext(os.path.basename(base_path))
    new_path = os.path.join(base_dir, f"{stem}_{ts}{ext}")

    if os.path.exists(base_path):
        legacy_dir = os.path.join(base_dir, "legacy")
        os.makedirs(legacy_dir, exist_ok=True)
        legacy_dst = os.path.join(legacy_dir, f"{stem}_{ts}{ext}")
        shutil.move(base_path, legacy_dst)
        _log.info(f"旧出力を退避: {legacy_dst}")

    return new_path


def run_generate(
    csv_path: Optional[str] = None,
    ranks: Optional[List[str]] = None,
    sender_company: Optional[str] = None,
    sender_name: Optional[str] = None,
    enable_web_search: Optional[bool] = None,
    enable_rank_estimation: Optional[bool] = None,
    output_path: Optional[str] = None,
    exhibition_name: Optional[str] = None,
    exhibition_date: Optional[str] = None,
    exhibition_venue: Optional[str] = None,
    on_progress: Optional[Callable] = None,
    candidate_dates: Optional[List[Dict]] = None,
    schedule_policy: str = "ab_only",
    enable_llm_judge: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    展示会リードのフォローアップメールを一括生成して CSV に保存する。

    Parameters
    ----------
    on_progress : callable(current, total, result), optional
        進捗コールバック。各リード処理後に呼ばれる。

    Returns
    -------
    dict
        ok (bool), total (int), success (int), errors (int),
        output_path (str), message (str)
    """
    from src.config import Config
    from src.vectordb import VectorDBManager
    from src.email_generator import EmailGenerator
    from src.agent import FollowUpAgent
    from src.utils import save_results_to_csv

    try:
        Config.validate()
    except EnvironmentError as e:
        return {"ok": False, "message": str(e), "total": 0, "success": 0, "errors": 0, "output_path": ""}

    config = load_cli_config()

    # CLI 引数を優先し、未指定は cli_config.yaml で補完
    _sender = sender_company if sender_company is not None else config.get("sender_company", "")
    _sender_name = sender_name if sender_name is not None else config.get("sender_name", "")
    _web = enable_web_search if enable_web_search is not None else config.get("enable_web_search", True)
    _rank_est = enable_rank_estimation if enable_rank_estimation is not None else config.get("enable_rank_estimation", True)
    _judge    = enable_llm_judge       if enable_llm_judge       is not None else config.get("enable_llm_judge", False)
    _output_naming = config.get("output_naming", "timestamp")
    _canonical     = config.get("output_path", "output/emails.csv")
    if output_path is not None:
        _output = output_path  # 引数明示時は常に優先（後方互換）
    else:
        _output = _manage_output_path(_output_naming, _canonical)
    _ranks = ranks or config.get("default_ranks", ["A", "B", "C"])
    _product_urls = config.get("product_urls", {})

    exhibition_info = {
        "exhibition_name": exhibition_name or "",
        "exhibition_date": exhibition_date or "",
        "exhibition_venue": exhibition_venue or "",
    }

    # リード読み込み
    lead_result = run_load_leads(csv_path=csv_path, ranks=_ranks)
    if not lead_result["ok"]:
        return {"ok": False, "message": lead_result["message"], "total": 0, "success": 0, "errors": 0, "output_path": ""}

    leads_df = lead_result["leads_df"]
    total = len(leads_df)
    if total == 0:
        return {
            "ok": False,
            "message": f"処理対象リードが0件です（ランクフィルタ {_ranks} を確認してください）",
            "total": 0,
            "success": 0,
            "errors": 0,
            "output_path": "",
        }

    # コンポーネント初期化
    try:
        vectordb = VectorDBManager(
            persist_dir=Config.CHROMA_DB_DIR,
            collection_name=Config.CHROMA_COLLECTION_NAME,
        )
        email_gen = EmailGenerator()
        agent = FollowUpAgent(vectordb_manager=vectordb, email_generator=email_gen)
    except Exception as e:
        return {"ok": False, "message": f"初期化エラー: {e}", "total": total, "success": 0, "errors": total, "output_path": ""}

    results = []
    error_count = 0

    for i, (idx, row) in enumerate(leads_df.iterrows()):
        lead = row.to_dict()
        try:
            result = agent.process_lead(
                lead=lead,
                sender_company=_sender,
                sender_name=_sender_name,
                enable_web_search=_web,
                enable_rank_estimation=_rank_est,
                exhibition_info=exhibition_info,
                product_urls=_product_urls,
                candidate_dates=candidate_dates,
                schedule_policy=schedule_policy,
                enable_llm_judge=_judge,
            )
            results.append(result)
        except Exception as e:
            error_count += 1
            results.append({
                "lead_id": lead.get("lead_id", f"L{i+1:03d}"),
                "visitor_name": lead.get("visitor_name", ""),
                "company_name": lead.get("company_name", ""),
                "email_to": lead.get("email", ""),
                "subject": "ERROR",
                "body": str(e),
                "cta": "",
            })

        if on_progress:
            on_progress(i + 1, total, results[-1])

    save_results_to_csv(results, _output)

    return {
        "ok": True,
        "total": total,
        "success": total - error_count,
        "errors": error_count,
        "output_path": _output,
        "results": results,
        "message": f"メール生成完了: {total - error_count}件成功 / {error_count}件エラー → {_output}",
    }


def validate_candidate_dates(input_str: str) -> Dict[str, Any]:
    """
    候補日入力文字列を検証・パースして単一エントリで返す（Skill 用）。

    Returns
    -------
    dict
        is_valid (bool), errors (list[str]), parsed (list[dict])
    """
    from src.date_validator import parse_candidate_dates, validate_dates

    try:
        parsed = parse_candidate_dates(input_str)
    except ValueError as e:
        return {"is_valid": False, "errors": [str(e)], "parsed": []}

    result = validate_dates(parsed)
    return {
        "is_valid": result.is_valid,
        "errors": result.errors,
        "parsed": parsed if result.is_valid else [],
    }


def run_setup_status() -> Dict[str, Any]:
    """現在のセットアップ状態を一覧で返す（/setup 用補助関数）。"""
    config = load_cli_config()
    check = run_check()

    def _find(label: str) -> dict:
        for item in check["items"]:
            if item["label"] == label:
                return item
        return {"status": "unknown", "detail": "不明"}

    api = _find("OPENAI_API_KEY")
    kb  = _find("ナレッジベース (KB)")
    gm  = _find("Gmail credentials")
    td  = _find("技術資料")

    return {
        "api_key_status":        api["status"],
        "api_key_detail":        api["detail"],
        "leads_csv":             config.get("leads_csv_path", "未設定"),
        "sender_company":        config.get("sender_company") or "未設定",
        "sender_name":           config.get("sender_name")    or "未設定",
        "exhibition_name":       config.get("exhibition_name") or "未設定",
        "exhibition_date":       config.get("exhibition_date") or "未設定",
        "exhibition_venue":      config.get("exhibition_venue") or "未設定",
        "tech_documents_status": td["status"],
        "tech_documents_detail": td["detail"],
        "kb_status":             kb["status"],
        "kb_detail":             kb["detail"],
        "gmail_status":          gm["status"],
        "gmail_detail":          gm["detail"],
        "default_ranks":         config.get("default_ranks", []),
    }


def run_kb_status() -> Dict[str, Any]:
    """
    ChromaDB のナレッジベース状態を返す。

    Returns
    -------
    dict
        total_chunks (int): 総チャンク数
        documents (list): ドキュメントごとの情報リスト
            - source (str): ファイル名
            - source_type (str): "markdown" / "pdf" / "unknown"
            - chunk_count (int): チャンク数
        last_updated (str or None): 最終更新日時 "YYYY-MM-DD HH:MM"
        is_empty (bool): KB が空なら True
    """
    from src.config import Config
    from collections import defaultdict

    chroma_sqlite = Path(Config.CHROMA_DB_DIR) / "chroma.sqlite3"
    if not chroma_sqlite.exists() or chroma_sqlite.stat().st_size <= 1000:
        return {
            "total_chunks": 0,
            "documents": [],
            "last_updated": None,
            "is_empty": True,
        }

    try:
        import chromadb as _chromadb
        _client = _chromadb.PersistentClient(path=str(Config.CHROMA_DB_DIR))
        try:
            _col = _client.get_collection(Config.CHROMA_COLLECTION_NAME)
        except Exception:
            return {
                "total_chunks": 0,
                "documents": [],
                "last_updated": None,
                "is_empty": True,
            }

        results = _col.get(include=["metadatas"])
        metadatas: list = results.get("metadatas") or []

        if not metadatas:
            return {
                "total_chunks": 0,
                "documents": [],
                "last_updated": None,
                "is_empty": True,
            }

        # ソースファイルごとにチャンク数・フォーマットを集計
        doc_chunks: Dict[str, int] = defaultdict(int)
        doc_formats: Dict[str, str] = {}

        for meta in metadatas:
            source_file = meta.get("source_file") or meta.get("source") or "unknown"
            doc_chunks[source_file] += 1
            if source_file not in doc_formats:
                # doc_format フィールドがあれば優先、なければ拡張子から判定
                doc_format = meta.get("doc_format")
                if doc_format is None:
                    if source_file.lower().endswith(".pdf"):
                        doc_format = "pdf"
                    elif source_file.lower().endswith(".md"):
                        doc_format = "markdown"
                    else:
                        doc_format = "unknown"
                doc_formats[source_file] = doc_format

        documents = [
            {
                "source": source_file,
                "source_type": doc_formats.get(source_file, "unknown"),
                "chunk_count": count,
            }
            for source_file, count in sorted(doc_chunks.items())
        ]

        # 最終更新日時: chroma_db/ ディレクトリの mtime で代替
        try:
            mtime = os.path.getmtime(str(Config.CHROMA_DB_DIR))
            last_updated = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        except Exception:
            last_updated = None

        return {
            "total_chunks": len(metadatas),
            "documents": documents,
            "last_updated": last_updated,
            "is_empty": False,
        }

    except Exception:
        return {
            "total_chunks": 0,
            "documents": [],
            "last_updated": None,
            "is_empty": True,
        }


def run_rank_mapping(leads_csv_path: str, rank_field: str, client) -> Dict[str, Any]:
    """
    CSVを読み込み → ユニーク値抽出 → LLM 推定 → 結果を返す。

    Returns
    -------
    dict
        unique_values (list[str]): ランクフィールドのユニーク値
        proposed_mapping (dict[str, str]): LLM が推定したマッピング
        already_clean (bool): すべての値がすでに A〜E の場合 True
        error (str, optional): エラーメッセージ（異常時のみ）
    """
    import re as _re
    from src.utils import load_leads, extract_unique_rank_values
    from src.rank_estimator import infer_rank_mapping_with_llm

    try:
        df = load_leads(leads_csv_path)
    except FileNotFoundError as e:
        return {"unique_values": [], "proposed_mapping": {}, "already_clean": False, "error": str(e)}

    if rank_field not in df.columns:
        return {
            "unique_values": [],
            "proposed_mapping": {},
            "already_clean": False,
            "error": f"フィールド '{rank_field}' が CSV に存在しません",
        }

    leads = df.to_dict(orient="records")
    unique_values = extract_unique_rank_values(leads, rank_field)

    clean_pattern = _re.compile(r"^[A-Ea-e]$")
    already_clean = bool(unique_values) and all(clean_pattern.match(v) for v in unique_values)

    if already_clean:
        return {"unique_values": unique_values, "proposed_mapping": {}, "already_clean": True}

    proposed_mapping = infer_rank_mapping_with_llm(unique_values, client)

    return {"unique_values": unique_values, "proposed_mapping": proposed_mapping, "already_clean": False}


def run_fetch_calendar_slots(
    days_ahead: int = 14,
    duration_minutes: int = 60,
) -> dict:
    """
    Google Calendar から空き時間スロットを取得する。

    cli_config.yaml の calendar: セクションの値を優先し、
    引数はデフォルト値として使用する。

    Returns
    -------
    dict
        slots (list[dict]): 空き枠リスト
        formatted (str): メール本文挿入用の整形文字列
        error (str or None): エラーメッセージ（正常時は None）
    """
    from src.calendar_client import get_calendar_service, fetch_free_slots, format_slots_for_email

    config = load_cli_config()
    cal_cfg = config.get("calendar", {})

    _days_ahead = cal_cfg.get("days_ahead", days_ahead)
    _duration = cal_cfg.get("duration_minutes", duration_minutes)
    _working_hours = (
        cal_cfg.get("working_hours_start", 9),
        cal_cfg.get("working_hours_end", 18),
    )
    _max_slots = cal_cfg.get("max_slots", 5)

    try:
        service = get_calendar_service()
        slots = fetch_free_slots(
            service,
            days_ahead=_days_ahead,
            duration_minutes=_duration,
            working_hours=_working_hours,
            max_slots=_max_slots,
        )
        formatted = format_slots_for_email(slots)
        return {"slots": slots, "formatted": formatted, "error": None}
    except FileNotFoundError as e:
        return {"slots": [], "formatted": "", "error": str(e)}
    except RuntimeError as e:
        return {"slots": [], "formatted": "", "error": str(e)}
    except Exception as e:
        return {"slots": [], "formatted": "", "error": str(e)}


def run_draft_to_gmail(
    results: Optional[List[Dict]] = None,
    output_csv_path: Optional[str] = None,
    credentials_path: str = "credentials/credentials.json",
    token_path: str = "credentials/token.json",
) -> Dict[str, Any]:
    """
    生成済みメール結果を Gmail の下書きフォルダに一括追加する。

    Parameters
    ----------
    results : run_generate() の戻り値に含まれる results リスト。
              指定しない場合は output_csv_path から CSV を読み込む。
    output_csv_path : results が None の場合に使う CSV パス。
    credentials_path : OAuth クライアント情報 JSON のパス。
    token_path : OAuth トークンの保存/読み込みパス。

    Returns
    -------
    dict
        ok (bool), success (int), errors (int),
        draft_ids (list), error_details (list), message (str)
    """
    from src.gmail_drafter import GmailDrafter

    if results is None and output_csv_path is None:
        config = load_cli_config()
        output_csv_path = config.get("output_path", "output/emails.csv")

    if results is None:
        import pandas as pd
        path = Path(output_csv_path)
        if not path.exists():
            return {
                "ok": False,
                "message": f"CSV が見つかりません: {path}",
                "success": 0,
                "errors": 0,
                "draft_ids": [],
                "error_details": [],
            }
        df = pd.read_csv(path, encoding="utf-8-sig")
        results = df.to_dict(orient="records")
        for r in results:
            r.setdefault("email_to", "")
            r.setdefault("subject", "")
            r.setdefault("body", "")

    try:
        drafter = GmailDrafter(
            credentials_path=credentials_path,
            token_path=token_path,
        )
        result = drafter.create_drafts_from_results(results)
    except FileNotFoundError as e:
        return {
            "ok": False,
            "message": str(e),
            "success": 0,
            "errors": 0,
            "draft_ids": [],
            "error_details": [],
        }
    except Exception as e:
        return {
            "ok": False,
            "message": f"Gmail API エラー: {e}",
            "success": 0,
            "errors": 0,
            "draft_ids": [],
            "error_details": [],
        }

    ok = result["errors"] == 0
    message = (
        f"Gmail 下書き作成完了: {result['success']}件成功 / {result['errors']}件エラー"
    )
    return {
        "ok": ok,
        "success": result["success"],
        "errors": result["errors"],
        "draft_ids": result["draft_ids"],
        "error_details": result["error_details"],
        "message": message,
    }
