"""
Streamlitアプリケーション エントリーポイント

NTX株式会社 展示会フォローアップエージェントのWebUI。
CSVアップロード → カラムマッピング → CRM連携（任意）→ メール生成 → ダウンロード のフローを提供する。

対応CSVフォーマット:
  - Lead Manager, Q-PASS, Sansan, 自社タブレット等の各種エクスポート形式
  - エンコーディング: UTF-8 / UTF-8 BOM / Shift_JIS / CP932

CRM連携:
  - Salesforce, kintone, HubSpot 等のエクスポートCSVに対応
  - 会社名のファジーマッチング（rapidfuzz）で自動紐付け
  - CRM CSV がない場合は従来のvectordb検索にフォールバック
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from src.config import Config
from src.agent import FollowUpAgent
from src.email_generator import EmailGenerator
from src.utils import (
    auto_map_columns,
    apply_column_mapping,
    check_lead_quality,
    filter_leads_by_rank,
    format_lead_summary,
    load_csv_with_encoding,
    load_leads,
)
from src.vectordb import VectorDBManager

# pypdf の利用可否を起動時に1回だけ確認
try:
    import pypdf  # noqa: F401
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

# ---------------------------------------------------------------
# 商談確度バッジの色定義（HTML インラインスタイル）
# ---------------------------------------------------------------
RANK_BADGE_STYLE: Dict[str, str] = {
    "A": "background-color:#ff4b4b;color:white;padding:2px 8px;border-radius:4px;font-weight:bold;",
    "B": "background-color:#ff9f00;color:white;padding:2px 8px;border-radius:4px;font-weight:bold;",
    "C": "background-color:#f0c000;color:#333;padding:2px 8px;border-radius:4px;font-weight:bold;",
    "D": "background-color:#4b8bff;color:white;padding:2px 8px;border-radius:4px;font-weight:bold;",
    "E": "background-color:#aaaaaa;color:white;padding:2px 8px;border-radius:4px;font-weight:bold;",
}

# マッピングしない場合のプレースホルダー
NO_MAPPING_LABEL = "（マッピングしない）"


# ---------------------------------------------------------------
# session_state の初期化
# ---------------------------------------------------------------
def _init_session_state() -> None:
    """アプリ起動時に session_state のキーを初期化する"""
    defaults = {
        # コンポーネントインスタンス
        "vectordb": None,           # VectorDBManager インスタンス
        "agent": None,              # FollowUpAgent インスタンス
        "email_gen": None,          # EmailGenerator インスタンス
        "db_built": False,          # ナレッジベース構築済みフラグ
        # リードデータ管理
        "leads_df": None,               # マッピング確定後の標準化DataFrame
        "raw_uploaded_df": None,        # アップロード直後の生DataFrame
        "auto_mapping": {},             # 自動推定マッピング辞書
        "mapping_confirmed": False,     # カラムマッピング確定フラグ
        "data_source": None,            # "upload" or "demo"
        # CRMデータ管理
        "crm_df": None,                 # マッピング確定後のCRM DataFrame（標準カラム名）
        "raw_crm_df": None,             # アップロード直後の生CRM DataFrame
        "crm_auto_mapping": {},         # CRMの自動推定マッピング辞書
        "crm_mapping_confirmed": False, # CRMカラムマッピング確定フラグ
        # 生成結果
        "results": [],                  # 一括生成結果リスト
        "single_result": None,          # 単一メール生成結果
        # 展示会情報
        "exhibition_info": {},
        # 品質チェック状態機械
        "gen_state": None,              # None / "error" / "confirm_warning" / "supplement"
        "quality_result": {},
        "pending_lead": {},
        # 機能トグル
        "enable_web_search": True,
        "enable_rank_estimation": True,
        # 送信元情報
        "sender_company": "",
        # ステップ進捗ガイド
        "show_next_step_kb": False,     # マッピング確定後にKB構築バナーを表示するフラグ
        # 音声管理
        "audio_files_meta": [],         # [{filename, duration_sec, start_time, file_bytes, rep_name}]
        "audio_associations": {},       # {lead_idx: filename} 確定した紐づけ
        "audio_transcripts": {},        # {lead_idx: transcript_str}
        "audio_needs": {},              # {lead_idx: needs_dict}
        "audio_match_results": [],      # List[MatchResult] 最新の紐づけ結果（ギャップ検出用）
        "audio_mapping_csvs": [],       # アップロードされた紐づけCSVのDataFrameリスト
        "audio_mapping_mode": "auto",   # "csv" | "auto"
        # 一括生成コスト確認ダイアログ
        "batch_confirm_pending": False,
        # プライバシー通知
        "privacy_notice_acknowledged": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ---------------------------------------------------------------
# アプリ起動時の共通初期化（VectorDB・EmailGenerator）
# ---------------------------------------------------------------
def _initialize_components() -> bool:
    """
    VectorDBManager と EmailGenerator を初期化して session_state に格納する。
    APIキーが未設定の場合は False を返す。
    """
    if not Config.OPENAI_API_KEY:
        st.error(
            "⚠️ OPENAI_API_KEY が設定されていません。\n\n"
            "`.env` ファイルに `OPENAI_API_KEY=your-key` を設定してから再起動してください。"
        )
        return False

    if st.session_state["vectordb"] is None:
        st.session_state["vectordb"] = VectorDBManager()

    if st.session_state["email_gen"] is None:
        st.session_state["email_gen"] = EmailGenerator()

    # FollowUpAgent は毎回再生成してモジュール変更を即反映する
    # (VectorDBManager と EmailGenerator はセッション内でキャッシュ済み)
    st.session_state["agent"] = FollowUpAgent(
        vectordb_manager=st.session_state["vectordb"],
        email_generator=st.session_state["email_gen"],
    )

    if st.session_state["vectordb"] is not None:
        st.session_state["db_built"] = st.session_state["vectordb"].is_index_built()

    return True


# ---------------------------------------------------------------
# サイドバー
# ---------------------------------------------------------------
def _render_sidebar() -> List[str]:
    """
    サイドバーを描画する。

    上から順に:
    1. CSVアップロード / デモデータ読み込みセクション
    2. ナレッジベース構築セクション
    3. 商談確度フィルター（データ読み込み後のみ表示）

    Returns
    -------
    List[str]
        選択された商談確度のリスト
    """
    selected_ranks: List[str] = ["A", "B", "C", "D", "E"]

    with st.sidebar:
        st.markdown("## 🏭 展示会フォローアップ")
        st.markdown("**メール自動生成システム**")
        st.divider()

        # ── 送信元会社名 ─────────────────────────────────────────
        st.session_state["sender_company"] = st.text_input(
            "🏢 送信元会社名",
            value=st.session_state.get("sender_company", ""),
            placeholder="例: 株式会社○○",
            help="生成するメールの署名・挨拶文に使用されます。未入力の場合は「弊社」になります。",
        )
        st.divider()

        # ── ステップ進捗インジケーター ────────────────────────────
        _mapping_done = st.session_state.get("mapping_confirmed", False)
        _kb_done = st.session_state.get("db_built", False)

        def _step_label(done: bool, active: bool, num: str, label: str) -> str:
            if done:
                return f"✅ {num} {label}"
            if active:
                return f"**🔷 {num} {label}**"
            return f"⬜ {num} {label}"

        st.markdown(
            _step_label(_mapping_done, not _mapping_done, "①", "データ読込") + "  \n" +
            _step_label(_kb_done, _mapping_done and not _kb_done, "②", "KB構築") + "  \n" +
            _step_label(False, _kb_done, "③", "メール生成")
        )
        st.divider()

        # ── セクション1: データ読み込み ───────────────────────────
        st.markdown("#### 📁 データ読み込み")

        # CSVアップロード
        uploaded_file = st.file_uploader(
            "CSVファイルをアップロード",
            type=["csv"],
            help="Lead Manager, Q-PASS, Sansan等のエクスポートCSVに対応。\nUTF-8 / Shift_JIS / BOM付き UTF-8 をサポートします。",
        )
        st.caption("🔒 アップロードされたデータはメール生成のためOpenAI APIに送信されます")

        # ファイルがアップロードされた場合の処理
        if uploaded_file is not None:
            # 前回とファイルが変わった場合はマッピングをリセット
            if uploaded_file.name != st.session_state.get("_last_uploaded_filename"):
                st.session_state["_last_uploaded_filename"] = uploaded_file.name
                st.session_state["mapping_confirmed"] = False
                st.session_state["leads_df"] = None
                st.session_state["results"] = []
                st.session_state["single_result"] = None
                try:
                    raw_df = load_csv_with_encoding(uploaded_file)
                    st.session_state["raw_uploaded_df"] = raw_df
                    # 必須・任意フィールドのマッピングを自動推定
                    all_fields = {**Config.REQUIRED_FIELDS, **Config.OPTIONAL_FIELDS}
                    st.session_state["auto_mapping"] = auto_map_columns(
                        list(raw_df.columns), all_fields
                    )
                    st.session_state["data_source"] = "upload"
                except ValueError as e:
                    st.error(str(e))
                    st.session_state["raw_uploaded_df"] = None

        # マッピング状態の表示
        if st.session_state["mapping_confirmed"]:
            st.success(f"✅ データ読み込み済み（{len(st.session_state['leads_df'])}件）")
            if st.session_state.get("raw_uploaded_df") is not None:
                if st.button("↩️ マッピングをやり直す", use_container_width=True, key="redo_mapping_sidebar"):
                    st.session_state["mapping_confirmed"] = False
                    st.rerun()
        elif st.session_state["raw_uploaded_df"] is not None:
            st.info("⚙️ カラムマッピングを確認してください（下部）")

        st.divider()

        # デモデータ読み込みボタン（Lead Manager形式を優先）
        if st.button("🗂️ デモデータを使う", use_container_width=True,
                     help="data/leads_rx_demo.csv（Lead Manager形式）を読み込みます"):
            import os as _os
            try:
                rx_path = "data/leads_rx_demo.csv"
                if _os.path.exists(rx_path):
                    import pandas as _pd2
                    raw_demo = _pd2.read_csv(rx_path, dtype=str, encoding="utf-8-sig").fillna("")
                    all_fields = {**Config.REQUIRED_FIELDS, **Config.OPTIONAL_FIELDS}
                    mapping = auto_map_columns(list(raw_demo.columns), all_fields)
                    demo_df = apply_column_mapping(raw_demo, mapping)
                    st.toast("✅ デモデータを読み込みました（Lead Manager形式）", icon="🗂️")
                else:
                    demo_df = load_leads(Config.LEADS_CSV_PATH)
                    st.toast("✅ デモデータを読み込みました", icon="🗂️")
                st.session_state["leads_df"] = demo_df
                st.session_state["raw_uploaded_df"] = None
                st.session_state["mapping_confirmed"] = True
                st.session_state["data_source"] = "demo"
                st.session_state["results"] = []
                st.session_state["single_result"] = None
                st.session_state["exhibition_info"] = {
                    "exhibition_name": "第35回 日本ものづくりワールド 2026",
                    "exhibition_date": "2026年4月10日〜12日",
                    "exhibition_venue": "東京ビッグサイト",
                }
                st.rerun()
            except Exception as e:
                st.error(f"デモデータの読み込みエラー: {e}")

        st.divider()

        # ── セクション2: ナレッジベース ──────────────────────────
        st.markdown("#### 🗄️ ナレッジベース")

        if st.session_state["db_built"]:
            st.success("✅ 構築済み")
        else:
            st.warning("⚠️ 未構築")

        _kb_is_next = (
            st.session_state.get("mapping_confirmed", False)
            and not st.session_state.get("db_built", False)
        )
        if st.button(
            "🔨 ① ナレッジベース構築",
            use_container_width=True,
            type="primary" if _kb_is_next else "secondary",
            help="Markdownの技術資料とCRM記録を再構築します。\nアップロード済みPDFは保持されます。",
        ):
            db: VectorDBManager = st.session_state["vectordb"]
            with st.spinner("ベクトルDBを構築中..."):
                try:
                    db.build_index(Config.TECH_DOCS_DIR, Config.CRM_RECORDS_DIR)
                    st.session_state["db_built"] = True
                    st.session_state["show_next_step_kb"] = False
                    st.toast("✅ ナレッジベースを構築しました", icon="✅")
                    st.rerun()
                except Exception as e:
                    st.error(f"構築エラー: {e}")

        # ── PDF アップロード ──────────────────────────────────────
        st.markdown("#### 📄 製品資料PDFアップロード")
        if not PYPDF_AVAILABLE:
            st.info(
                "📄 PDF取り込み機能は現在ご利用いただけません。\n\n"
                "ご利用を希望される場合は **IT担当者にご連絡ください**。\n\n"
                "（必要設定: `pypdf` パッケージのインストール）"
            )
        else:
            st.caption("PDFをアップロードしてナレッジベースに追加できます")
            pdf_files = st.file_uploader(
                "PDFをアップロード",
                type=["pdf"],
                accept_multiple_files=True,
                key="pdf_uploader",
                help="複数ファイルを同時にアップロード可能。追加後はメール生成時の参照資料として活用されます。",
            )
            if pdf_files and st.button("🔨 PDFをベクトルDBに追加", use_container_width=True, key="pdf_add_btn"):
                db: VectorDBManager = st.session_state["vectordb"]
                total_chunks = 0
                added_count = 0
                with st.spinner(f"{len(pdf_files)}件のPDFを取り込み中..."):
                    for pdf_file in pdf_files:
                        try:
                            chunks = db.add_pdf(pdf_file, source_name=pdf_file.name)
                            total_chunks += chunks
                            if chunks > 0:
                                added_count += 1
                            else:
                                st.warning(
                                    f"⚠️ '{pdf_file.name}' からテキストを抽出できませんでした。"
                                    "スキャンPDF（画像のみ）は現在非対応です。"
                                )
                        except Exception as e:
                            st.warning(f"⚠️ '{pdf_file.name}' の取り込みに失敗: {e}")
                if added_count > 0:
                    st.session_state["db_built"] = True
                    st.success(f"✅ {added_count}件のPDFを取り込みました（合計{total_chunks}チャンク）")

        st.divider()

        # ── セクション3: CRM情報の連携（任意）──────────────────────
        _render_crm_sidebar_section()

        st.divider()

        # ── セクション4: 商談確度フィルター（データ読み込み後のみ）──
        if st.session_state["mapping_confirmed"] and st.session_state["leads_df"] is not None:
            leads_df = st.session_state["leads_df"]
            st.markdown("#### 🎯 商談確度フィルター")
            selected_ranks = st.multiselect(
                label="対象ランク",
                options=["A", "B", "C", "D", "E"],
                default=["A", "B", "C", "D", "E"],
                help="表示・処理対象の商談確度を選択してください",
            )
            if selected_ranks:
                filtered_count = len(filter_leads_by_rank(leads_df, selected_ranks))
                st.caption(f"対象: **{filtered_count}件** / 全{len(leads_df)}件")

            st.divider()

        # ── セクション5: 機能トグル ─────────────────────────────
        st.markdown("#### ⚙️ 機能設定")
        st.session_state["enable_web_search"] = st.toggle(
            "🌐 Web検索（DuckDuckGo）",
            value=st.session_state.get("enable_web_search", True),
            help="メール生成時に顧客企業の最新ニュースを検索してメールに反映します（APIキー不要）",
        )
        st.session_state["enable_rank_estimation"] = st.toggle(
            "🎯 ランクAI自動推定",
            value=st.session_state.get("enable_rank_estimation", True),
            help="ランク未入力・★形式等の場合、メモ・役職からAIが商談確度を推定します（GPT 1回追加）",
        )
        st.divider()

        st.caption("展示会フォローアップシステム")

    return selected_ranks


# ---------------------------------------------------------------
# CRM連携サイドバーセクション
# ---------------------------------------------------------------
def _render_crm_sidebar_section() -> None:
    """
    サイドバーにCRM CSV連携セクションを描画する。

    - CRM CSVアップロード / デモCRMデータボタン
    - アップロード後: カラムマッピングフォーム（expander内）
    - 確定後: 連携済み表示 + リセットボタン
    """
    st.markdown("#### 📂 CRM情報の連携（任意）")
    st.caption("Salesforce/kintone等のCSVを会社名で自動紐付けします")

    # CRM CSV ファイルアップロード
    crm_uploaded_file = st.file_uploader(
        "CRM CSVをアップロード",
        type=["csv"],
        key="crm_csv_uploader",
        help="Salesforce, kintone, HubSpot等のエクスポートCSVに対応。\n会社名カラムで展示会リードと自動紐付けします。",
    )
    st.caption("🔒 CRM情報もメール生成のためOpenAI APIに送信されます")

    # 新しいCRM CSVが選択された場合: マッピングをリセットして読み込む
    if crm_uploaded_file is not None:
        if crm_uploaded_file.name != st.session_state.get("_last_crm_filename"):
            st.session_state["_last_crm_filename"] = crm_uploaded_file.name
            st.session_state["crm_mapping_confirmed"] = False
            st.session_state["crm_df"] = None
            try:
                raw_crm = load_csv_with_encoding(crm_uploaded_file)
                st.session_state["raw_crm_df"] = raw_crm
                # CRM必須・任意フィールドでカラムマッピングを自動推定
                all_crm_fields = {**Config.CRM_REQUIRED_FIELDS, **Config.CRM_OPTIONAL_FIELDS}
                st.session_state["crm_auto_mapping"] = auto_map_columns(
                    list(raw_crm.columns), all_crm_fields
                )
            except ValueError as e:
                st.error(str(e))
                st.session_state["raw_crm_df"] = None

    # デモCRMデータ読み込みボタン（HubSpot形式を優先）
    if st.button("🗃️ デモCRMデータを使う", use_container_width=True, key="crm_demo_btn",
                 help="data/crm_hubspot_demo.csv（HubSpot形式）または crm_demo.csv を読み込みます"):
        _load_demo_crm()

    # ── CRM連携状態の表示 ─────────────────────────────────────
    if st.session_state["crm_mapping_confirmed"] and st.session_state["crm_df"] is not None:
        crm_df = st.session_state["crm_df"]
        st.success(f"✅ CRM連携済み: {len(crm_df)}社のデータ")
        if st.button("🔄 CRM連携をリセット", use_container_width=True, key="crm_reset_btn"):
            st.session_state["crm_df"] = None
            st.session_state["raw_crm_df"] = None
            st.session_state["crm_auto_mapping"] = {}
            st.session_state["crm_mapping_confirmed"] = False
            st.session_state["_last_crm_filename"] = None
            st.rerun()

    elif st.session_state["raw_crm_df"] is not None and not st.session_state["crm_mapping_confirmed"]:
        # CRMアップロード済みだがマッピング未確定: コンパクトなフォームを表示
        raw_crm_df = st.session_state["raw_crm_df"]
        crm_auto_mapping = st.session_state["crm_auto_mapping"]

        with st.expander("⚙️ CRMカラムマッピングを確認", expanded=True):
            st.caption(f"読み込んだCSV: {len(raw_crm_df)}行 / {len(raw_crm_df.columns)}カラム")

            crm_col_options = [NO_MAPPING_LABEL] + list(raw_crm_df.columns)

            with st.form("crm_mapping_form"):
                crm_selections: Dict[str, Optional[str]] = {}

                # 必須フィールド（メール / 会社名 の2キー）
                st.markdown("**🔴 紐付けキー** ※いずれか1つ以上マッピング（両方推奨）")
                for req_field_key, req_field_def in Config.CRM_REQUIRED_FIELDS.items():
                    current = crm_auto_mapping.get(req_field_key)
                    default_idx = crm_col_options.index(current) if current in crm_col_options else 0
                    selected = st.selectbox(
                        label=f"{req_field_def['label']} `{req_field_key}`",
                        options=crm_col_options,
                        index=default_idx,
                        key=f"crm_req_{req_field_key}",
                        help=req_field_def.get("description", ""),
                    )
                    crm_selections[req_field_key] = None if selected == NO_MAPPING_LABEL else selected

                # 任意フィールド（HubSpot標準フィールド）
                st.markdown("**🔵 任意フィールド（HubSpot標準）**")
                for field_key, field_def in Config.CRM_OPTIONAL_FIELDS.items():
                    current = crm_auto_mapping.get(field_key)
                    default_idx = crm_col_options.index(current) if current in crm_col_options else 0
                    selected = st.selectbox(
                        label=f"{field_def['label']} `{field_key}`",
                        options=crm_col_options,
                        index=default_idx,
                        key=f"crm_opt_{field_key}",
                    )
                    crm_selections[field_key] = None if selected == NO_MAPPING_LABEL else selected

                # 確定ボタン
                if st.form_submit_button("✅ CRMマッピング確定", type="primary", use_container_width=True):
                    has_key = any(
                        crm_selections.get(k) is not None
                        for k in Config.CRM_REQUIRED_FIELDS
                    )
                    if not has_key:
                        st.warning("⚠️ メールアドレスまたは会社名のいずれか1つ以上をマッピングしてください")
                    else:
                        all_crm_fields = {**Config.CRM_REQUIRED_FIELDS, **Config.CRM_OPTIONAL_FIELDS}
                        crm_df = apply_column_mapping(raw_crm_df, crm_selections)
                        st.session_state["crm_df"] = crm_df
                        st.session_state["crm_mapping_confirmed"] = True
                        st.toast(f"✅ CRM連携確定: {len(crm_df)}社", icon="📂")
                        st.rerun()


def _load_demo_crm() -> None:
    """デモCRMデータを読み込んでマッピングを適用する（HubSpot形式を優先）"""
    import os
    # HubSpot形式を優先、なければ従来のcrm_demo.csvにフォールバック
    crm_path = "data/crm_hubspot_demo.csv"
    if not os.path.exists(crm_path):
        crm_path = "data/crm_demo.csv"
    if not os.path.exists(crm_path):
        st.error("デモCRMデータが見つかりません")
        return
    try:
        raw_crm = pd.read_csv(crm_path, dtype=str, encoding="utf-8-sig").fillna("")
        # CRM必須・任意フィールドで自動マッピング
        all_crm_fields = {**Config.CRM_REQUIRED_FIELDS, **Config.CRM_OPTIONAL_FIELDS}
        mapping = auto_map_columns(list(raw_crm.columns), all_crm_fields)
        crm_df = apply_column_mapping(raw_crm, mapping)
        st.session_state["crm_df"] = crm_df
        st.session_state["raw_crm_df"] = raw_crm
        st.session_state["crm_auto_mapping"] = mapping
        st.session_state["crm_mapping_confirmed"] = True
        st.session_state["_last_crm_filename"] = os.path.basename(crm_path)
        fname = os.path.basename(crm_path)
        st.toast(f"✅ デモCRMデータを読み込みました（{len(crm_df)}社 / {fname}）", icon="🗃️")
        st.rerun()
    except Exception as e:
        st.error(f"デモCRMデータの読み込みエラー: {e}")


# ---------------------------------------------------------------
# データ未読み込み時のウェルカム画面
# ---------------------------------------------------------------
def _render_welcome() -> None:
    """CSVがアップロードされていない初期状態のウェルカム画面を表示する"""
    st.markdown("---")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("### 📁 CSVをアップロード")
        st.markdown(
            "左サイドバーから展示会リードCSVをアップロードしてください。\n\n"
            "**対応ツール例:**\n"
            "- Lead Manager\n"
            "- Q-PASS\n"
            "- Sansan\n"
            "- 自社タブレット / Google フォーム\n\n"
            "**対応エンコーディング:** UTF-8 / UTF-8 BOM / Shift_JIS / CP932"
        )

    with col2:
        st.markdown("### 🗂️ デモデータで試す")
        st.markdown(
            "サイドバーの **「🗂️ デモデータを使う」** ボタンをクリックすると、\n\n"
            "架空の展示会リード18件を読み込んで\nすぐに機能を試せます。\n\n"
            "**含まれるデータ:**\n"
            "- 商談確度A〜E のリード各種\n"
            "- 製造業DX関連の架空企業データ\n"
            "- 関心製品・営業メモ付き"
        )

    st.markdown("---")
    st.markdown(
        "#### 💡 使い方の流れ\n"
        "**① データ読込** — CSVアップロード または デモデータ を選択 → カラムマッピング確定\n\n"
        "**② ナレッジベース構築** — サイドバーの「🔨 ① ナレッジベース構築」で製品資料を登録\n\n"
        "**③ メール生成** — メール生成タブでフォローアップメールを生成\n\n"
        "**④ ダウンロード** — 生成結果を CSV でダウンロード"
    )


# ---------------------------------------------------------------
# カラムマッピング確認・確定UI（メインエリア）
# ---------------------------------------------------------------
def _render_column_mapping() -> None:
    """
    アップロードされたCSVのカラムマッピング確認UIを表示する。

    - 自動推定されたマッピングを selectbox で表示
    - 必須フィールドが未マッピングの場合は警告を表示
    - 「✅ マッピング確定」ボタンで leads_df を確定する
    """
    raw_df: pd.DataFrame = st.session_state["raw_uploaded_df"]
    auto_mapping: Dict = st.session_state["auto_mapping"]

    st.subheader("⚙️ カラムマッピングの確認")
    st.caption(
        "アップロードされたCSVのカラム名と、システム内部のフィールドの対応を確認してください。\n"
        "自動推定が異なる場合はドロップダウンで修正できます。"
    )

    # CSVプレビュー（先頭5行）
    with st.expander(f"📋 アップロードされたCSVのプレビュー（先頭5行 / 全{len(raw_df)}件）", expanded=True):
        st.dataframe(raw_df.head(5), use_container_width=True, hide_index=True)
        st.caption(f"カラム数: {len(raw_df.columns)}  |  総行数: {len(raw_df)}")

    st.markdown("---")

    # CSVカラムの選択肢（「マッピングしない」を先頭に追加）
    column_options = [NO_MAPPING_LABEL] + list(raw_df.columns)

    # フォーム形式でマッピング入力を受け付ける
    with st.form("column_mapping_form"):
        # ── 必須フィールド ────────────────────────────────────────
        st.markdown("#### 🔴 必須フィールド")
        st.caption("これらのフィールドは正確なメール生成のために必要です")

        required_selections: Dict[str, Optional[str]] = {}
        req_cols = st.columns(len(Config.REQUIRED_FIELDS))

        for i, (field_key, field_def) in enumerate(Config.REQUIRED_FIELDS.items()):
            current_match = auto_mapping.get(field_key)
            default_idx = column_options.index(current_match) if current_match in column_options else 0
            label = f"{field_def['label']}  \n`{field_key}`"
            help_text = field_def.get("description", "")

            selected = req_cols[i].selectbox(
                label=label,
                options=column_options,
                index=default_idx,
                key=f"mapping_required_{field_key}",
                help=help_text,
            )
            required_selections[field_key] = None if selected == NO_MAPPING_LABEL else selected

        # ── 任意フィールド ────────────────────────────────────────
        st.markdown("#### 🔵 任意フィールド")
        st.caption("マッピングしなくてもメール生成は可能です。あれば活用されます")

        optional_selections: Dict[str, Optional[str]] = {}
        # 4列グリッドで表示
        opt_keys = list(Config.OPTIONAL_FIELDS.keys())
        for row_start in range(0, len(opt_keys), 4):
            row_keys = opt_keys[row_start:row_start + 4]
            opt_cols = st.columns(4)
            for j, field_key in enumerate(row_keys):
                field_def = Config.OPTIONAL_FIELDS[field_key]
                current_match = auto_mapping.get(field_key)
                default_idx = column_options.index(current_match) if current_match in column_options else 0
                label = f"{field_def['label']}  \n`{field_key}`"

                selected = opt_cols[j].selectbox(
                    label=label,
                    options=column_options,
                    index=default_idx,
                    key=f"mapping_optional_{field_key}",
                )
                optional_selections[field_key] = None if selected == NO_MAPPING_LABEL else selected

        # 未マッピングカラムの表示
        all_mapped_cols = {v for v in {**required_selections, **optional_selections}.values() if v}
        unmapped_cols = [c for c in raw_df.columns if c not in all_mapped_cols]
        if unmapped_cols:
            st.info(
                f"**追加情報として保持されるカラム ({len(unmapped_cols)}件):**  \n"
                + "、".join(f"`{c}`" for c in unmapped_cols)
                + "  \nこれらは `extra_カラム名` として保持され、メール生成のコンテキストに活用されます。"
            )

        # ── 展示会情報（任意）────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📅 展示会情報（任意・全リードに共通で使用されます）")
        st.caption("入力しない場合はメール内で「展示会」と表記されます")
        _ex_prev = st.session_state.get("exhibition_info", {})
        ex_cols = st.columns(3)
        exhibition_name_val = ex_cols[0].text_input(
            Config.EXHIBITION_INFO_FIELDS["exhibition_name"]["label"],
            placeholder=Config.EXHIBITION_INFO_FIELDS["exhibition_name"]["placeholder"],
            key="exhib_name_input",
        )
        exhibition_date_val = ex_cols[1].text_input(
            Config.EXHIBITION_INFO_FIELDS["exhibition_date"]["label"],
            placeholder=Config.EXHIBITION_INFO_FIELDS["exhibition_date"]["placeholder"],
            key="exhib_date_input",
        )
        exhibition_venue_val = ex_cols[2].text_input(
            Config.EXHIBITION_INFO_FIELDS["exhibition_venue"]["label"],
            placeholder=Config.EXHIBITION_INFO_FIELDS["exhibition_venue"]["placeholder"],
            key="exhib_venue_input",
        )

        # 確定ボタン
        submitted = st.form_submit_button(
            "✅ マッピング確定 → メール生成へ進む",
            type="primary",
            use_container_width=True,
        )

        if submitted:
            # 必須フィールドのバリデーション
            missing_required = [
                Config.REQUIRED_FIELDS[k]["label"]
                for k, v in required_selections.items()
                if v is None
            ]
            if missing_required:
                st.error(
                    f"🔴 以下の必須フィールドがマッピングされていません: "
                    f"{', '.join(missing_required)}\n\n"
                    "マッピングを設定してから確定してください。"
                )
                return

            # マッピングを適用してDataFrameを標準化
            final_mapping = {**required_selections, **optional_selections}
            leads_df = apply_column_mapping(raw_df, final_mapping)

            # session_state に保存
            st.session_state["leads_df"] = leads_df
            st.session_state["mapping_confirmed"] = True
            st.session_state["show_next_step_kb"] = True
            st.session_state["results"] = []
            st.session_state["single_result"] = None
            st.session_state["exhibition_info"] = {
                "exhibition_name": exhibition_name_val,
                "exhibition_date": exhibition_date_val,
                "exhibition_venue": exhibition_venue_val,
            }
            st.toast(f"✅ マッピング確定。{len(leads_df)}件のリードを読み込みました", icon="✅")
            st.rerun()


# ---------------------------------------------------------------
# タブ1: リード一覧
# ---------------------------------------------------------------
def _render_tab_leads(leads_df: pd.DataFrame, selected_ranks: List[str]) -> None:
    """リード一覧タブを描画する"""
    st.subheader("📋 展示会リード一覧")

    filtered_df = filter_leads_by_rank(leads_df, selected_ranks) if selected_ranks else leads_df
    st.caption(
        f"表示件数: {len(filtered_df)}件"
        f"（フィルタ: ランク {', '.join(selected_ranks) if selected_ranks else '全て'}）"
    )

    if filtered_df.empty:
        st.info("選択した商談確度に該当するリードがありません。")
        return

    # ランク正規化プレビュー（LLMなし・高速）
    from src.rank_estimator import RankEstimator as _RE
    _re = _RE.__new__(_RE)  # LLM初期化なしでインスタンス生成
    _re.llm = None

    def _rank_preview(raw: str) -> str:
        n = _re.normalize_rank(str(raw))
        if n is None:
            return f"{raw or '(空)'} → AI推定"
        return n if str(raw).strip().upper() == n else f"{raw} → {n}"

    display_df = filtered_df.copy()
    if "lead_rank" in display_df.columns:
        display_df["ランク（変換後）"] = display_df["lead_rank"].apply(_rank_preview)

    # 表示列を決定（標準フィールド + extra_ カラム）
    standard_cols = {
        "lead_id": "ID", "visitor_name": "氏名", "company_name": "会社名",
        "department": "部署", "job_title": "役職", "lead_rank": "元ランク",
        "ランク（変換後）": "ランク（変換後）",
        "interested_products": "関心製品", "visit_date": "来場日",
    }
    show_cols = {k: v for k, v in standard_cols.items() if k in display_df.columns}
    display_df = display_df[list(show_cols.keys())].rename(columns=show_cols)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ランク凡例バッジ
    st.markdown(
        "**ランク凡例:**  " + "　".join(
            f'<span style="{RANK_BADGE_STYLE[r]}">{r}</span>'
            for r in ["A", "B", "C", "D", "E"]
        ),
        unsafe_allow_html=True,
    )

    # ランク別件数サマリー
    st.divider()
    cols = st.columns(5)
    for i, rank in enumerate(["A", "B", "C", "D", "E"]):
        if "lead_rank" in filtered_df.columns:
            cnt = len(filtered_df[filtered_df["lead_rank"] == rank])
        else:
            cnt = 0
        cols[i].metric(label=f"ランク {rank}", value=f"{cnt}件")

    # extra_ カラムがある場合は追加情報として折りたたみ表示
    extra_cols = [c for c in filtered_df.columns if c.startswith("extra_")]
    if extra_cols:
        with st.expander(f"📎 追加情報カラム ({len(extra_cols)}件)"):
            st.caption("CSVにあった独自カラムはメール生成のコンテキストに自動的に活用されます")
            st.dataframe(
                filtered_df[["visitor_name"] + extra_cols].head(10),
                use_container_width=True,
                hide_index=True,
            )


# ステータスアイコン定義
_STEP_ICONS: Dict[str, str] = {
    "running": "⏳",
    "done": "✅",
    "skip": "⏭️",
    "warning": "⚠️",
}
_SCORE_COLORS: Dict[str, str] = {
    "高": "#28a745",
    "中": "#ffc107",
    "低": "#dc3545",
}


def _do_single_generate(lead_data: Dict[str, Any]) -> None:
    """
    単一リードのメール生成を st.status + on_step コールバックで実行する。
    結果を session_state["single_result"] に保存し、gen_state をリセットする。
    """
    agent: FollowUpAgent = st.session_state["agent"]
    crm_df: Optional[pd.DataFrame] = st.session_state.get("crm_df")
    exhibition_info = st.session_state.get("exhibition_info", {})
    visitor = lead_data.get("visitor_name", "")
    company = lead_data.get("company_name", "")

    with st.status(f"{visitor}様（{company}）のメールを生成中...", expanded=True) as _status:
        step_display = st.empty()
        steps_state: Dict[int, Dict] = {}

        def _on_step(num: int, name: str, step_status: str, detail: str = "") -> None:
            steps_state[num] = {"name": name, "status": step_status, "detail": detail}
            lines = []
            for i in range(1, 8):
                s = steps_state.get(i)
                if s:
                    icon = _STEP_ICONS.get(s["status"], "○")
                    d = s.get("detail", "")
                    line = f"{icon} **Step {i}　{s['name']}**"
                    if d:
                        line += f"  \n　　{d}"
                    lines.append(line)
            step_display.markdown("\n\n".join(lines))

        try:
            _lead_idx = lead_data.get("_df_index")
            result = agent.process_lead(
                lead_data, crm_df=crm_df, exhibition_info=exhibition_info, on_step=_on_step,
                enable_web_search=st.session_state.get("enable_web_search", True),
                enable_rank_estimation=st.session_state.get("enable_rank_estimation", True),
                sender_company=st.session_state.get("sender_company", ""),
                transcript=st.session_state["audio_transcripts"].get(_lead_idx, ""),
                extracted_needs=st.session_state["audio_needs"].get(_lead_idx),
            )
            st.session_state["single_result"] = result
            st.session_state["gen_state"] = None
            _status.update(label="✅ メール生成が完了しました", state="complete")
            st.toast("✅ メール生成が完了しました", icon="✉️")
        except Exception as e:
            _status.update(label="❌ エラーが発生しました", state="error")
            st.error(f"メール生成エラー: {e}")


# ---------------------------------------------------------------
# タブ2: メール生成
# ---------------------------------------------------------------
def _render_tab_email(leads_df: pd.DataFrame, selected_ranks: List[str]) -> None:
    """メール生成タブを描画する"""
    st.subheader("✉️ フォローアップメール生成")

    if not st.session_state["db_built"]:
        st.error("🔒 このタブを使うには、まず **② KB管理** でナレッジベースの構築が必要です")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success("✅ ① データ読込")
        with col2:
            st.warning("▶️ ② KB構築 ← 次のステップ")
        with col3:
            st.info("🔒 ③ メール生成")
        st.info("💡 サイドバーの **「🔨 ナレッジベース構築」** ボタンを押してください。")
        return

    filtered_df = filter_leads_by_rank(leads_df, selected_ranks) if selected_ranks else leads_df

    if filtered_df.empty:
        st.info("対象リードがありません。サイドバーのランクフィルターを確認してください。")
        return

    # ── 個別メール生成 ──────────────────────────────────────────
    st.markdown("### 個別メール生成")

    lead_options = {
        f"{row.get('visitor_name', '?')}（{row.get('company_name', '?')}）"
        f"[ランク{row.get('lead_rank', '?')}]": idx
        for idx, row in filtered_df.iterrows()
    }
    selected_label = st.selectbox(
        "対象リードを選択",
        options=list(lead_options.keys()),
        help="メールを生成するリードを選択してください",
    )
    selected_idx = lead_options[selected_label]
    selected_lead = filtered_df.loc[selected_idx].to_dict()
    selected_lead["_df_index"] = selected_idx  # 音声コンテキスト伝播用

    # 選択リード情報サマリー
    rank = selected_lead.get("lead_rank", "")
    badge_html = f'<span style="{RANK_BADGE_STYLE.get(rank, "")}">ランク {rank}</span>'
    st.markdown(
        f"**選択中:** {selected_lead.get('visitor_name', '')} 様　{badge_html}",
        unsafe_allow_html=True,
    )
    st.info(format_lead_summary(selected_lead))

    # ── 充足度チェック + 状態機械による生成フロー ─────────────────
    # 選択リードが変わったら状態をリセット
    if selected_label != st.session_state.get("_gen_state_lead_label"):
        st.session_state["_gen_state_lead_label"] = selected_label
        st.session_state["gen_state"] = None
        st.session_state["quality_result"] = {}
        st.session_state["pending_lead"] = {}

    gen_state = st.session_state.get("gen_state")

    if gen_state is None:
        if st.button("📧 メール生成", type="primary", key="single_gen"):
            if not st.session_state["db_built"]:
                st.error("ナレッジベースを先に構築してください。")
            else:
                quality = check_lead_quality(selected_lead)
                if quality["errors"]:
                    st.session_state["gen_state"] = "error"
                    st.session_state["quality_result"] = quality
                    st.rerun()
                elif quality["warnings"]:
                    st.session_state["gen_state"] = "confirm_warning"
                    st.session_state["quality_result"] = quality
                    st.session_state["pending_lead"] = selected_lead.copy()
                    st.rerun()
                else:
                    _do_single_generate(selected_lead)

    elif gen_state == "error":
        quality = st.session_state.get("quality_result", {})
        for err in quality.get("errors", []):
            st.error(f"🔴 {err}のため生成できません")
        if st.button("↩️ 戻る", key="cancel_error_btn"):
            st.session_state["gen_state"] = None
            st.rerun()

    elif gen_state == "confirm_warning":
        quality = st.session_state.get("quality_result", {})
        st.warning("⚠️ 以下の情報が不足しています。このまま生成しますか？")
        for w in quality.get("warnings", []):
            st.markdown(f"- {w}")
        _w_col1, _w_col2, _w_col3 = st.columns(3)
        with _w_col1:
            if st.button("🔧 補完して生成", key="supplement_btn", use_container_width=True):
                st.session_state["gen_state"] = "supplement"
                st.rerun()
        with _w_col2:
            if st.button("✅ このまま生成", key="generate_anyway_btn", use_container_width=True):
                _do_single_generate(st.session_state.get("pending_lead", selected_lead))
        with _w_col3:
            if st.button("↩️ キャンセル", key="cancel_warning_btn", use_container_width=True):
                st.session_state["gen_state"] = None
                st.rerun()

    elif gen_state == "supplement":
        _SUPPL_FIELDS: Dict[str, Dict] = {
            "interested_products": {"label": "関心製品",   "type": "text"},
            "memo":                {"label": "商談メモ",   "type": "textarea"},
            "lead_rank":           {"label": "商談確度",   "type": "rank"},
        }
        pending = st.session_state.get("pending_lead", selected_lead)
        missing_fields = {
            k: v for k, v in _SUPPL_FIELDS.items()
            if not str(pending.get(k, "")).strip()
        }
        st.info("不足情報を入力してください（CSVの元データは変更されません）")
        with st.form("supplement_form"):
            supplement_vals: Dict[str, str] = {}
            for field_key, meta in missing_fields.items():
                if meta["type"] == "textarea":
                    supplement_vals[field_key] = st.text_area(
                        meta["label"], value="", height=80, key=f"suppl_{field_key}"
                    )
                elif meta["type"] == "rank":
                    supplement_vals[field_key] = st.selectbox(
                        meta["label"],
                        options=["A", "B", "C", "D", "E"],
                        index=2,
                        key=f"suppl_{field_key}",
                    )
                else:
                    supplement_vals[field_key] = st.text_input(
                        meta["label"], value="", key=f"suppl_{field_key}"
                    )
            _s_col1, _s_col2 = st.columns(2)
            with _s_col1:
                generate_suppl = st.form_submit_button(
                    "✅ この内容で生成", type="primary", use_container_width=True
                )
            with _s_col2:
                cancel_suppl = st.form_submit_button("↩️ キャンセル", use_container_width=True)

        if generate_suppl:
            merged = {**pending, **{k: v for k, v in supplement_vals.items() if str(v).strip()}}
            _do_single_generate(merged)
        if cancel_suppl:
            st.session_state["gen_state"] = None
            st.rerun()

    # 生成結果の表示
    if st.session_state["single_result"]:
        result = st.session_state["single_result"]
        st.divider()
        st.markdown("#### 生成結果")

        st.text_input("📌 件名", value=result.get("subject", ""), key="edit_subject")
        st.text_area("📝 本文", value=result.get("body", ""), height=400, key="edit_body")
        st.text_area(
            "🎯 営業アクション指示（社内向け）",
            value=result.get("cta", ""),
            height=100,
            key="edit_cta",
            help="LLMが生成した、営業担当者が次に実行すべきアクションの指示です。メール本文には含まれません。",
        )

        # ── CRM紐付け結果の表示 ────────────────────────────────
        crm_structured: Optional[Dict[str, Any]] = result.get("crm_structured")
        crm_source = result.get("crm_source", "none")
        crm_score = result.get("crm_match_score", 0)

        with st.expander("📋 CRM情報（HubSpot連携結果）"):
            if crm_structured and crm_source == "csv":
                # HubSpot CRM からのマッチ結果を構造化表示
                match_method = crm_structured.get("match_method", "")
                method_label = "メール一致" if match_method == "email" else "会社名マッチ"
                st.caption(
                    f"取得元: HubSpot CRM  |  紐付け: {method_label}  |  "
                    f"マッチスコア: {crm_score}/100"
                )
                col1, col2, col3 = st.columns(3)
                col1.metric("Lifecycle stage", crm_structured.get("lifecycle_stage") or "－")
                col2.metric("Lead status", crm_structured.get("lead_status") or "－")
                col3.metric("担当者", crm_structured.get("contact_owner") or "－")

                info_rows = {
                    "最終接触日":   crm_structured.get("last_activity_date", ""),
                    "初回登録日":   crm_structured.get("create_date", ""),
                    "獲得経路":     crm_structured.get("original_source", ""),
                    "Record ID":    crm_structured.get("record_id", ""),
                    "紐付け会社名": crm_structured.get("matched_company", ""),
                }
                for label, val in info_rows.items():
                    if val:
                        st.markdown(f"**{label}:** {val}")

            elif crm_source == "vectordb":
                st.caption("取得元: ナレッジベース（vectordb）")
                ref_crm = result.get("ref_crm", [])
                if ref_crm:
                    for f in ref_crm:
                        st.markdown(f"- `{f}`")
                else:
                    st.caption("該当するCRM記録なし")
            else:
                st.caption("CRM情報: 該当なし")

        with st.expander("📚 RAG参照コンテキスト詳細"):
            tech_chunks = result.get("retrieved_tech_chunks", [])
            crm_chunks = result.get("retrieved_crm_chunks", [])

            # ── 技術資料チャンク ──────────────────────────────
            if tech_chunks:
                st.markdown(f"**🔧 技術資料（{len(tech_chunks)}件ヒット）**")
                for i, chunk in enumerate(tech_chunks, 1):
                    src = chunk.get("source_file", "")
                    src_type = chunk.get("source_type", "tech_doc")
                    lbl = chunk.get("score_label", "？")
                    raw = chunk.get("score", 0)
                    color = _SCORE_COLORS.get(lbl, "#aaa")
                    type_tag = "PDF" if src_type == "pdf_upload" else "Markdown"
                    st.markdown(
                        f'**[{i}] {src}** `{type_tag}` &nbsp;'
                        f'スコア: <span style="color:{color}">●</span> **{lbl}**'
                        f' <small>({raw:.4f})</small>',
                        unsafe_allow_html=True,
                    )
                    if lbl == "低":
                        st.caption("⚠️ 関連性が低い可能性があります")
                    preview = chunk.get("text_preview", "")
                    st.caption(preview[:180] + ("..." if len(preview) > 180 else ""))
                    with st.expander("全文を表示"):
                        st.text(preview)
            else:
                st.info("技術資料の参照なし（インデックス未構築 or ヒットなし）")

            st.divider()

            # ── CRM照合結果 ──────────────────────────────────
            if crm_chunks:
                st.markdown("**🏢 CRM照合結果**")
                for chunk in crm_chunks:
                    src_type = chunk.get("source_type", "")
                    if src_type == "crm_csv":
                        method = chunk.get("match_method", "")
                        method_label = "メール完全一致 ✅" if method == "email" else "会社名マッチ"
                        st.markdown(
                            f"**{chunk.get('company_name', '？')}**"
                            f" — 照合方法: {method_label}"
                            f" (スコア: {chunk.get('match_score', 0)})"
                        )
                        if chunk.get("deal_stage"):
                            st.caption(f"ライフサイクルステージ: {chunk['deal_stage']}")
                        if chunk.get("lead_status"):
                            st.caption(f"リードステータス: {chunk['lead_status']}")
                    else:
                        src = chunk.get("source_file", "")
                        lbl = chunk.get("score_label", "？")
                        color = _SCORE_COLORS.get(lbl, "#aaa")
                        st.markdown(
                            f'**{src}** スコア: <span style="color:{color}">●</span> **{lbl}**',
                            unsafe_allow_html=True,
                        )
                        meta_parts = []
                        if chunk.get("company_name"):
                            meta_parts.append(f"企業: {chunk['company_name']}")
                        if chunk.get("deal_stage"):
                            meta_parts.append(f"ステージ: {chunk['deal_stage']}")
                        if meta_parts:
                            st.caption(" / ".join(meta_parts))
                        if chunk.get("text_preview"):
                            with st.expander("全文を表示"):
                                st.text(chunk["text_preview"])
            else:
                st.info("CRM情報の参照なし")

            st.divider()

            # ── Web検索結果 ──────────────────────────────────────
            web_results = result.get("web_search_results", [])
            if web_results:
                profile_results = [r for r in web_results if r.get("section") == "profile"]
                news_results = [r for r in web_results if r.get("section") == "news"]
                other_results = [r for r in web_results if r.get("section") not in ("profile", "news")]

                def _render_web_items(items, label):
                    if not items:
                        return
                    st.markdown(f"**{label}（{len(items)}件）**")
                    for wr in items:
                        st.markdown(f"**{wr.get('title', '')}**")
                        snippet = wr.get("snippet", "")
                        if snippet:
                            st.caption(snippet)
                        url = wr.get("url", "")
                        if url:
                            st.markdown(
                                f"<small><a href='{url}' target='_blank'>{url[:80]}</a></small>",
                                unsafe_allow_html=True,
                            )

                _render_web_items(profile_results, "🏢 事業内容・製品・サービス")
                _render_web_items(news_results, "📰 最新動向")
                _render_web_items(other_results, "🌐 Web検索結果")
            else:
                st.info("Web検索結果なし（無効または結果なし）")

            st.divider()

            # ── ランク推定情報 ──────────────────────────────────
            rank_info = result.get("rank_info", {})
            if rank_info:
                method = rank_info.get("method", "")
                original = rank_info.get("original", "")
                rank = rank_info.get("rank", "")
                confidence = rank_info.get("confidence", "")
                _METHOD_LABELS = {
                    "existing": "そのまま使用",
                    "normalized": "変換",
                    "llm_estimated": "AI推定",
                    "default": "デフォルト(C)",
                }
                method_label = _METHOD_LABELS.get(method, method)
                conf_color = {"high": "#22c55e", "medium": "#f59e0b", "low": "#ef4444"}.get(confidence, "#aaa")
                st.markdown(
                    f"**🎯 ランク判定:** `{rank}` &nbsp;"
                    f"({method_label}"
                    + (f" / 元値: `{original}`" if original and original != rank else "")
                    + f") &nbsp; <span style='color:{conf_color}'>●</span> 信頼度: {confidence}",
                    unsafe_allow_html=True,
                )

    st.divider()

    # ── 全件一括生成 ──────────────────────────────────────────
    st.markdown("### 全件一括生成")
    st.caption(f"対象: {len(filtered_df)}件のリードに対してメールを一括生成します")

    # ── コスト見積もりヘルパー ──────────────────────────────────────
    def _calc_cost_estimate(n: int, enable_web: bool, enable_rank: bool) -> dict:
        from src.config import Config
        input_tokens = Config.EST_INPUT_TOKENS_PER_LEAD
        if enable_rank:
            input_tokens += Config.EST_RANK_EXTRA_INPUT_TOKENS
        total_input = n * input_tokens
        total_output = n * Config.EST_OUTPUT_TOKENS_PER_LEAD
        cost_input = total_input / 1_000_000 * Config.LLM_PRICE_INPUT_PER_1M
        cost_output = total_output / 1_000_000 * Config.LLM_PRICE_OUTPUT_PER_1M
        total_cost = cost_input + cost_output
        secs = n * Config.EST_SECONDS_PER_LEAD_BASE
        if enable_web:
            secs += n * Config.EST_SECONDS_PER_LEAD_WEB
        return {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "cost_input": cost_input,
            "cost_output": cost_output,
            "total_cost": total_cost,
            "est_minutes": secs / 60,
        }

    total_leads = len(filtered_df)

    if not st.session_state.get("batch_confirm_pending", False):
        # ── Step 1: 生成ボタン ─────────────────────────────────────
        if st.button("🔄 全件一括生成", key="batch_gen"):
            if not st.session_state["db_built"]:
                st.error("ナレッジベースを先に構築してください。")
            else:
                st.session_state["batch_confirm_pending"] = True
                st.rerun()
    else:
        # ── Step 2: コスト確認ダイアログ ──────────────────────────
        enable_web = st.session_state.get("enable_web_search", True)
        enable_rank = st.session_state.get("enable_rank_estimation", True)

        already_done = len(st.session_state.get("results", []))
        remaining = total_leads - already_done
        can_resume = 0 < already_done < total_leads

        target_n = remaining if can_resume else total_leads
        est = _calc_cost_estimate(target_n, enable_web=enable_web, enable_rank=enable_rank)

        st.warning("### ⚠️ 実行前に確認してください", icon="⚠️")

        if can_resume:
            st.info(f"✅ **{already_done}件** は生成済みです。続きから **{remaining}件** のみ実行できます。")

        st.markdown(
            f"""
| 項目 | 見積もり |
|------|---------|
| 対象件数 | **{target_n:,} 件**{"（残り）" if can_resume else ""} |
| 推定入力トークン | {est['total_input_tokens']:,} トークン |
| 推定出力トークン | {est['total_output_tokens']:,} トークン |
| 推定コスト（入力） | **${est['cost_input']:.4f}** |
| 推定コスト（出力） | **${est['cost_output']:.4f}** |
| **合計推定コスト** | **${est['total_cost']:.4f}** |
| 推定所要時間 | 約 **{est['est_minutes']:.0f} 分** |
"""
        )
        st.caption(
            "※ gpt-5.4-nano 料金（入力 $0.20/1M・出力 $1.25/1M）をもとに算出した目安です。"
            "実際の料金はキャッシュ利用状況等により変動します。"
        )
        st.info("途中で止まった場合でも、「生成履歴・ダウンロード」タブから生成済み件数分のCSVをダウンロードできます。")

        execute = False
        restart = False
        if can_resume:
            col_resume, col_fresh, col_cancel = st.columns([3, 3, 1])
            with col_resume:
                execute = st.button(
                    f"▶️ 続きから実行（残り{remaining}件）",
                    key="batch_run_resume",
                    type="primary",
                    use_container_width=True,
                )
            with col_fresh:
                restart = st.button(
                    "🔄 最初から実行（生成済みを上書き）",
                    key="batch_run_fresh",
                    use_container_width=True,
                )
            with col_cancel:
                if st.button("❌", key="batch_cancel", use_container_width=True):
                    st.session_state["batch_confirm_pending"] = False
                    st.rerun()
        else:
            col_ok, col_cancel = st.columns([1, 1])
            with col_ok:
                execute = st.button("✅ 実行する", key="batch_run_confirmed", type="primary")
            with col_cancel:
                if st.button("❌ キャンセル", key="batch_cancel"):
                    st.session_state["batch_confirm_pending"] = False
                    st.rerun()

        if execute or restart:
            st.session_state["batch_confirm_pending"] = False
            if restart:
                st.session_state["results"] = []

            agent: FollowUpAgent = st.session_state["agent"]
            total = total_leads
            results = list(st.session_state.get("results", []))
            start_idx = len(results)

            progress_bar = st.progress(
                start_idx / total if total > 0 else 0, text="生成準備中..."
            )
            status_text = st.empty()
            status_step = st.empty()

            crm_df_batch: Optional[pd.DataFrame] = st.session_state.get("crm_df")
            exhibition_info_batch = st.session_state.get("exhibition_info", {})

            def _on_step_batch(num: int, name: str, step_status: str, detail: str = "") -> None:
                if step_status != "skip":
                    icon = _STEP_ICONS.get(step_status, "○")
                    status_step.caption(f"    {icon} Step {num}: {name} — {detail}")

            for i, (_, row) in enumerate(filtered_df.iterrows()):
                if i < start_idx:
                    continue  # 生成済みをスキップ

                lead = row.to_dict()
                visitor = lead.get("visitor_name", "")
                company = lead.get("company_name", "")
                status_text.text(f"[{i+1}/{total}] {visitor}様（{company}）のメール生成中...")
                progress_bar.progress(i / total, text=f"生成中... {i}/{total}件")

                try:
                    result = agent.process_lead(
                        lead, crm_df=crm_df_batch, exhibition_info=exhibition_info_batch,
                        on_step=_on_step_batch,
                        enable_web_search=st.session_state.get("enable_web_search", True),
                        enable_rank_estimation=st.session_state.get("enable_rank_estimation", True),
                        sender_company=st.session_state.get("sender_company", ""),
                        transcript=st.session_state["audio_transcripts"].get(row.name, ""),
                        extracted_needs=st.session_state["audio_needs"].get(row.name),
                    )
                    results.append(result)
                except Exception as e:
                    results.append({
                        "lead_id": lead.get("lead_id"),
                        "visitor_name": visitor,
                        "company_name": company,
                        "lead_rank": lead.get("lead_rank"),
                        "email_to": lead.get("email"),
                        "subject": "ERROR",
                        "body": str(e),
                        "cta": "",
                        "ref_tech_docs": [],
                        "ref_crm": [],
                        "quality_score": check_lead_quality(lead).get("score", 0),
                        "retrieved_tech_chunks": [],
                        "retrieved_crm_chunks": [],
                    })
                st.session_state["results"] = results  # 逐次保存

            progress_bar.progress(1.0, text=f"✅ 完了 {total}/{total}件")
            status_text.empty()

            error_count = sum(1 for r in results if r.get("subject") == "ERROR")
            if error_count > 0:
                st.warning(f"⚠️ {total}件中 {error_count}件でエラーが発生しました。CSVの「件名」列が 'ERROR' の行を確認してください。")
            st.toast(f"✅ {total}件のメール生成が完了しました", icon="✅")
            st.rerun()


# ---------------------------------------------------------------
# タブ3: 生成履歴・ダウンロード
# ---------------------------------------------------------------
def _render_tab_history() -> None:
    """生成履歴・ダウンロードタブを描画する"""
    st.subheader("📊 生成履歴・ダウンロード")

    results: List[Dict] = st.session_state.get("results", [])

    if not results:
        st.info("まだメールが生成されていません。「✉️ メール生成」タブから一括生成を実行してください。")
        return

    st.caption(f"生成済み: {len(results)}件")

    # サマリーテーブル
    summary_rows = []
    for r in results:
        _score = r.get("quality_score")
        if r.get("subject") == "ERROR":
            _status = "❌ エラー"
        elif _score is not None and _score < 60:
            _status = "⚠️ 情報不足"
        else:
            _status = "✅ 正常"
        summary_rows.append({
            "ID": r.get("lead_id", ""),
            "氏名": r.get("visitor_name", ""),
            "会社名": r.get("company_name", ""),
            "ランク": r.get("lead_rank", ""),
            "充足度": f"{_score}%" if _score is not None else "—",
            "宛先メール": r.get("email_to", ""),
            "件名": r.get("subject", ""),
            "ステータス": _status,
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.divider()

    # 各メール詳細（折りたたみ）
    st.markdown("#### メール詳細")
    for r in results:
        rank = r.get("lead_rank", "")
        label = f"{r.get('visitor_name')}（{r.get('company_name')}） [{rank}]"
        with st.expander(label):
            st.markdown(f"**件名:** {r.get('subject', '')}")
            st.text_area(
                "本文",
                value=r.get("body", ""),
                height=300,
                key=f"history_body_{r.get('lead_id')}",
                disabled=True,
            )
            if r.get("cta"):
                st.markdown(f"**🎯 営業アクション指示:** {r.get('cta', '')}")
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"参照技術資料: {', '.join(r.get('ref_tech_docs', [])) or 'なし'}")
            with col2:
                st.caption(f"参照CRM: {', '.join(r.get('ref_crm', [])) or 'なし'}")

    st.divider()

    # CSVダウンロード
    st.markdown("#### CSVダウンロード")
    download_rows = [
        {
            "lead_id": r.get("lead_id", ""),
            "visitor_name": r.get("visitor_name", ""),
            "company_name": r.get("company_name", ""),
            "lead_rank": r.get("lead_rank", ""),
            "email_to": r.get("email_to", ""),
            "subject": r.get("subject", ""),
            "body": r.get("body", ""),
            "cta": r.get("cta", ""),
            "ref_tech_docs": ", ".join(r.get("ref_tech_docs", [])),
            "ref_crm": ", ".join(r.get("ref_crm", [])),
        }
        for r in results
    ]
    csv_bytes = (
        pd.DataFrame(download_rows)
        .to_csv(index=False, encoding="utf-8-sig")
        .encode("utf-8-sig")
    )
    st.download_button(
        label="📥 生成結果をCSVでダウンロード",
        data=csv_bytes,
        file_name="followup_emails.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )


# ---------------------------------------------------------------
# タブ5: 音声管理
# ---------------------------------------------------------------
def _render_tab_audio(leads_df) -> None:
    """音声管理タブを描画する（Step A: アップロード → Step B: 紐づけ → Step C: 文字起こし）"""
    from src.audio_processor import AudioProcessor
    from src.audio_matcher import AudioMatcher, MatchResult
    from src.config import Config

    st.subheader("🎙️ 音声管理")
    st.caption(
        "展示会当日の録音ファイルをアップロードし、来場者リードに紐づけて文字起こし・ニーズ抽出を行います。"
    )

    # ── Step A: ファイルアップロード ─────────────────────────────
    st.markdown("### Step A: 音声ファイルアップロード")
    st.caption(
        "📂 ファイル命名規則: `YYYYMMDD_担当者名_連番.{mp3/m4a/wav}`　例: `20260424_営業A_001.m4a`  \n"
        "担当者名がファイル名に含まれていると、リードへの自動紐づけ精度が上がります。"
    )

    uploaded_audios = st.file_uploader(
        "音声ファイルをアップロード",
        type=["mp3", "wav", "m4a"],
        accept_multiple_files=True,
        key="audio_uploader",
        help="複数ファイルを同時にアップロード可能。25MB超のファイルは Whisper API の制限により非対応です。",
    )

    if uploaded_audios:
        processor = AudioProcessor(api_key=Config.OPENAI_API_KEY)
        matcher = AudioMatcher()

        # メタデータ取得・コスト概算
        meta_list = []
        oversized = []
        total_duration = 0.0
        for f in uploaded_audios:
            file_bytes = f.read()
            meta = processor.get_audio_metadata(file_bytes, f.name)
            rep_name = matcher.parse_rep_from_filename(f.name)
            meta["filename"] = f.name
            meta["rep_name"] = rep_name
            meta["file_bytes"] = file_bytes
            meta_list.append(meta)
            if meta["size_mb"] > 25:
                oversized.append(f.name)
            else:
                total_duration += meta["duration_sec"]

        # session_state に保存
        st.session_state["audio_files_meta"] = meta_list

        # コスト概算表示
        est_cost = processor.estimate_cost(total_duration)
        col_a, col_b = st.columns(2)
        col_a.metric("アップロード件数", f"{len(meta_list)} 件")
        col_b.metric("合計録音時間", f"約 {int(total_duration / 60)} 分")
        st.info(
            f"💰 文字起こし費用の目安: **約 ${est_cost:.3f}**"
            f"（Whisper API: $0.006/分、合計 {int(total_duration / 60)} 分）"
        )
        if oversized:
            st.warning(
                f"⚠️ 以下のファイルは 25MB を超えるため文字起こしできません。短く分割してください:\n"
                + "\n".join(f"- {n}" for n in oversized)
            )

    files_meta = st.session_state.get("audio_files_meta", [])
    if not files_meta:
        st.info("音声ファイルをアップロードすると、紐づけと文字起こしを行えます。")
        return

    st.divider()

    # ── Step A.5: 紐づけCSVアップロード（推奨）─────────────────────────
    st.markdown("### Step A.5: 紐づけCSVアップロード（推奨）")
    st.caption(
        "担当者ごとに作成した紐づけCSVをアップロードしてください。  \n"
        "ファイル名: `YYYYMMDD_担当者名_紐づけ.csv`（例: `20260425_営業A_紐づけ.csv`）  \n"
        "CSVフォーマット: 1行目ヘッダ `filename,visitor_name`、2行目以降にデータ"
    )

    uploaded_mapping_csvs = st.file_uploader(
        "紐づけCSVをアップロード",
        type=["csv"],
        accept_multiple_files=True,
        key="audio_mapping_csv_uploader",
        help="担当者ごとにCSVを分けてください。複数ファイルを同時アップロード可能。",
    )

    mapping_csv_data = []  # [{rep_name, df, csv_name}]
    if uploaded_mapping_csvs:
        _csv_matcher = AudioMatcher()
        for f in uploaded_mapping_csvs:
            try:
                df_csv = pd.read_csv(f)
                if "filename" not in df_csv.columns or "visitor_name" not in df_csv.columns:
                    st.warning(
                        f"⚠️ `{f.name}` に `filename` と `visitor_name` 列が必要です。スキップします。"
                    )
                    continue
                rep = _csv_matcher.parse_rep_from_csv_filename(f.name)
                mapping_csv_data.append({"rep_name": rep or "（不明）", "df": df_csv, "csv_name": f.name})
            except Exception as e:
                st.warning(f"⚠️ `{f.name}` の読み込みに失敗しました: {e}")

        if mapping_csv_data:
            for item in mapping_csv_data:
                st.markdown(f"- **{item['rep_name']}**: {len(item['df'])}件 （`{item['csv_name']}`）")

            if st.button("✅ 紐づけCSVを使って確定", type="primary", key="audio_csv_confirm_btn"):
                _csv_match = AudioMatcher()
                all_csv_results: List[MatchResult] = []
                for item in mapping_csv_data:
                    all_csv_results.extend(
                        _csv_match.match_with_csv(
                            mapping_df=item["df"],
                            rep_name=item["rep_name"],
                            audio_meta_list=files_meta,
                            leads_df=leads_df,
                        )
                    )
                assoc = {r.audio_filename: r.lead_idx for r in all_csv_results if r.lead_idx is not None}
                st.session_state["audio_associations"] = assoc
                st.session_state["audio_match_results"] = all_csv_results
                st.session_state["audio_mapping_csvs"] = [item["df"] for item in mapping_csv_data]
                st.session_state["audio_mapping_mode"] = "csv"
                st.toast(f"✅ {len(assoc)}件の紐づけを確定しました（CSVモード）", icon="🎙️")
                st.rerun()

    st.markdown("---")
    st.markdown("*または* ── 紐づけCSVなしで自動紐づけ ──")
    st.markdown("---")

    # ── Step B: 紐づけ確認 ────────────────────────────────────────
    st.markdown("### Step B: リードへの紐づけ確認")

    matcher = AudioMatcher()
    _llm = st.session_state.get("email_gen") and st.session_state["email_gen"].llm
    _rep_col = "rep_name" if "rep_name" in leads_df.columns else None
    _timestamp_col = matcher.detect_timestamp_col(leads_df, llm=_llm)
    match_results: List[MatchResult] = matcher.match(
        audio_meta_list=files_meta,
        leads_df=leads_df,
        rep_col=_rep_col,
        timestamp_col=_timestamp_col,
        tolerance_minutes=Config.AUDIO_TIMESTAMP_TOLERANCE_MINUTES,
    )
    if st.session_state.get("audio_mapping_mode") != "csv":
        st.session_state["audio_match_results"] = match_results

    # 赤フラグ率チェック
    red_rate = matcher.get_red_flag_rate(match_results)
    if red_rate > Config.AUDIO_RED_FLAG_WARNING_THRESHOLD:
        red_count = sum(1 for r in match_results if r.confidence == "red")
        st.warning(
            f"⚠️ **命名ルールが守られていないファイルが多数あります（{red_count}/{len(match_results)}件）**  \n"
            f"ファイル名を `YYYYMMDD_担当者名_連番.m4a` の形式に変更してから再アップロードしてください。"
        )

    # 既存の手動設定を読み込む
    saved_assoc: dict = st.session_state.get("audio_associations", {})

    lead_options = ["（紐づけなし）"] + [
        f"{row.get('visitor_name', '')} ({row.get('company_name', '')})"
        for _, row in leads_df.iterrows()
    ]
    lead_idx_map = {i: idx for i, (idx, _) in enumerate(leads_df.iterrows(), start=1)}

    pending_associations: dict = {}

    # ── 🟢🟡 自動紐づけ結果 ─────────────────────────────────────────
    for r in [r for r in match_results if r.confidence != "red"]:
        conf_icon = {"green": "🟢", "yellow": "🟡"}.get(r.confidence, "⬜")
        rep_label = r.rep_name or "（不明）"
        col_icon, col_audio, col_lead = st.columns([1, 3, 4])
        col_icon.markdown(f"**{conf_icon}**")
        col_audio.markdown(
            f"`{r.audio_filename}`  \n"
            f"<small>担当者: {rep_label} / {int(r.duration_sec)}秒 / {r.note}</small>",
            unsafe_allow_html=True,
        )
        if r.lead_idx is not None:
            lead_row = leads_df.loc[r.lead_idx]
            col_lead.markdown(
                f"→ {lead_row.get('visitor_name', '')} ({lead_row.get('company_name', '')})"
            )
            pending_associations[r.audio_filename] = r.lead_idx
        else:
            col_lead.markdown("→ 紐づけ候補なし")

    # ── 🔴 担当者名不明ファイル（個別選択 or まとめてスキップ）────────
    red_results = [r for r in match_results if r.confidence == "red"]
    if red_results:
        with st.expander(
            f"🔴 担当者名不明 {len(red_results)}件 — 個別に紐づけるか、スキップしてください",
            expanded=True,
        ):
            st.caption(
                "「（紐づけなし）」のままにすると音声コンテキストはメール生成に使用されません。"
                "「✅ 紐づけを確定」を押すとスキップとして確定されます。"
            )
            for r in red_results:
                col_icon, col_audio, col_lead = st.columns([1, 3, 4])
                col_icon.markdown("**🔴**")
                col_audio.markdown(
                    f"`{r.audio_filename}`  \n"
                    f"<small>担当者: （不明） / {int(r.duration_sec)}秒 / {r.note}</small>",
                    unsafe_allow_html=True,
                )
                default_idx = 0
                if r.audio_filename in saved_assoc:
                    saved_lead_idx = saved_assoc[r.audio_filename]
                    for li, di in lead_idx_map.items():
                        if di == saved_lead_idx:
                            default_idx = li
                            break
                sel = col_lead.selectbox(
                    "リードを選択",
                    options=range(len(lead_options)),
                    format_func=lambda i: lead_options[i],
                    index=default_idx,
                    key=f"audio_sel_{r.audio_filename}",
                    label_visibility="collapsed",
                )
                if sel > 0:
                    pending_associations[r.audio_filename] = lead_idx_map[sel]

    col_confirm, col_skip = st.columns([2, 3])
    if col_confirm.button("✅ 紐づけを確定", type="primary", key="audio_confirm_btn"):
        st.session_state["audio_associations"] = pending_associations
        st.session_state["audio_match_results"] = match_results
        st.session_state["audio_mapping_mode"] = "auto"
        st.toast(f"✅ {len(pending_associations)}件の紐づけを確定しました", icon="🎙️")
        st.rerun()
    if red_results:
        col_skip.caption(f"🔴 未紐づけ {len(red_results)}件はそのまま「確定」を押すとスキップされます")

    confirmed_assoc = st.session_state.get("audio_associations", {})
    if not confirmed_assoc:
        return

    # ── 録音し忘れ検出 ────────────────────────────────────────────
    audio_results_saved = st.session_state.get("audio_match_results")
    if audio_results_saved:
        rep_col = "rep_name" if "rep_name" in leads_df.columns else None
        mapping_dfs_saved = (
            st.session_state.get("audio_mapping_csvs")
            if st.session_state.get("audio_mapping_mode") == "csv"
            else None
        )
        gaps = matcher.detect_gaps(audio_results_saved, leads_df, rep_col, mapping_dfs=mapping_dfs_saved)
        if gaps:
            lines = []
            for g in gaps:
                names_str = "、".join(g["likely_unrecorded"])
                suffix = (
                    f"（{names_str}さんの録音がない可能性）"
                    if names_str
                    else f"（{g['missing_count']}件の録音がない可能性）"
                )
                lines.append(
                    f"- {g['rep_name']}：リード{g['lead_count']}件に対して"
                    f"音声{g['audio_count']}件{suffix}"
                )
            st.warning("⚠️ **録音し忘れの可能性があります**\n" + "\n".join(lines))

    st.divider()

    # ── Step C: 文字起こし・ニーズ抽出 ───────────────────────────
    st.markdown("### Step C: 文字起こし・ニーズ抽出")
    st.caption(
        f"確定済み紐づけ: {len(confirmed_assoc)}件  \n"
        "Whisper API で文字起こしを行い、LLM でニーズを構造化抽出します。"
    )

    # 完了済みスキップ
    done_count = sum(
        1 for v in st.session_state.get("audio_transcripts", {}).values() if v
    )
    if done_count:
        st.success(f"✅ {done_count}件の文字起こしが完了しています。")

    if st.button(
        f"▶ {len(confirmed_assoc)}件を文字起こし・ニーズ抽出",
        type="primary",
        key="audio_transcribe_btn",
    ):
        processor = AudioProcessor(
            api_key=Config.OPENAI_API_KEY,
            llm=st.session_state.get("email_gen") and st.session_state["email_gen"].llm,
        )
        transcripts: dict = dict(st.session_state.get("audio_transcripts", {}))
        needs_map: dict = dict(st.session_state.get("audio_needs", {}))
        file_bytes_map = {m["filename"]: m["file_bytes"] for m in files_meta}

        prog = st.progress(0)
        stat = st.empty()
        total = len(confirmed_assoc)

        for j, (filename, lead_idx) in enumerate(confirmed_assoc.items()):
            stat.text(f"[{j+1}/{total}] {filename} を処理中...")
            if lead_idx in transcripts and transcripts[lead_idx]:
                prog.progress((j + 1) / total)
                continue  # スキップ（既に完了）

            file_bytes = file_bytes_map.get(filename)
            if not file_bytes:
                stat.warning(f"⚠️ {filename} のバイトデータが見つかりません")
                prog.progress((j + 1) / total)
                continue

            try:
                transcript = processor.transcribe(file_bytes, filename)
                extracted = processor.extract_needs(transcript)
                transcripts[lead_idx] = transcript
                needs_map[lead_idx] = extracted
            except Exception as e:
                st.warning(f"⚠️ {filename}: {e}")

            prog.progress((j + 1) / total)

        st.session_state["audio_transcripts"] = transcripts
        st.session_state["audio_needs"] = needs_map
        stat.text("完了")
        st.toast(f"✅ 文字起こし・ニーズ抽出が完了しました（{total}件）", icon="🎙️")
        st.rerun()

    # 結果プレビュー
    transcripts = st.session_state.get("audio_transcripts", {})
    needs_map = st.session_state.get("audio_needs", {})
    if transcripts:
        st.divider()
        st.markdown("#### 抽出結果プレビュー")
        for lead_idx, transcript in transcripts.items():
            if not transcript:
                continue
            try:
                lead_row = leads_df.loc[lead_idx]
                name = f"{lead_row.get('visitor_name', '')} ({lead_row.get('company_name', '')})"
            except KeyError:
                name = f"リード index {lead_idx}"
            needs = needs_map.get(lead_idx, {})
            with st.expander(f"🎙️ {name}"):
                if needs.get("summary"):
                    st.markdown(f"**要約**: {needs['summary']}")
                cols = st.columns(2)
                for i, (key, label) in enumerate([
                    ("issues", "課題"), ("needs", "ニーズ"),
                    ("budget", "予算感"), ("temperature", "温度感"),
                    ("decision_maker", "決裁者"),
                ]):
                    cols[i % 2].markdown(f"**{label}**: {needs.get(key, '—')}")
                with st.expander("文字起こし全文"):
                    st.text(transcript)


# ---------------------------------------------------------------
# タブ4: ナレッジベース確認
# ---------------------------------------------------------------
def _render_tab_knowledge() -> None:
    """ナレッジベース確認タブを描画する"""
    st.subheader("🗄️ ナレッジベース確認")

    db: VectorDBManager = st.session_state["vectordb"]

    if not st.session_state.get("db_built") or not db.is_index_built():
        st.info("ナレッジベースが未構築です。サイドバーの「🔨 ナレッジベース構築」を実行してください。")
        return

    summary = db.get_index_summary()
    total = summary.get("total_chunks", 0)
    by_type = summary.get("by_source_type", {})
    by_product = summary.get("by_product", {})
    parent_chunks = summary.get("parent_chunks", 0)
    parent_store_size_kb = summary.get("parent_store_size_kb", 0.0)
    parent_store_path = summary.get("parent_store_path", "")

    # ── チャンキング戦略 ──────────────────────────────────────
    st.markdown("### チャンキング戦略")
    with st.expander("現在の設定", expanded=False):
        st.markdown(
            "**戦略**: 親子チャンク（Parent-Child Chunking）\n\n"
            "| 種別 | 子チャンクサイズ | 親チャンクサイズ |\n"
            "|------|-----------------|------------------|\n"
            "| Markdown技術資料 | 250文字 | 1000文字 |\n"
            "| CRM記録 | 350文字 | 1200文字 |\n"
            "| PDFアップロード | 400文字 | 1400文字 |\n\n"
            "検索は小さい**子チャンク**で精度高く行い、LLMには**親チャンク**の広いコンテキストを渡します。"
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("子チャンク総数（検索対象）", f"{total} 件")
        c2.metric("親チャンク総数（LLMコンテキスト）", f"{parent_chunks} 件")
        c3.metric("親ストアファイルサイズ", f"{parent_store_size_kb} KB")
        if parent_store_path:
            st.caption(f"親ストア: `{parent_store_path}`")

    # ── インデックスサマリー ─────────────────────────────────
    st.markdown("### インデックスサマリー")
    st.metric("総チャンク数（子）", f"{total} 件")

    _TYPE_LABELS = {
        "tech_doc":   "Markdown技術資料",
        "crm_record": "CRM記録",
        "pdf_upload": "PDFアップロード",
    }
    if by_type:
        rows = []
        for type_key, data in by_type.items():
            rows.append({
                "種別":       _TYPE_LABELS.get(type_key, type_key),
                "チャンク数": data["count"],
                "ファイル数": len(data["files"]),
                "ファイル一覧": ", ".join(data["files"][:5]) + ("..." if len(data["files"]) > 5 else ""),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── アップロード済みPDF管理 ──────────────────────────────
    pdf_files_in_db = by_type.get("pdf_upload", {}).get("files", [])
    st.markdown("### 📄 アップロード済みPDF")
    if not pdf_files_in_db:
        st.caption("現在PDFは登録されていません。サイドバーからPDFをアップロードできます。")
    else:
        st.caption(f"{len(pdf_files_in_db)}件のPDFが登録されています。不要なファイルは削除できます。")
        for pdf_name in pdf_files_in_db:
            col_name, col_del = st.columns([5, 1])
            col_name.markdown(f"📄 {pdf_name}")
            if col_del.button("🗑️", key=f"del_pdf_{pdf_name}", help=f"{pdf_name} を削除"):
                removed = db.remove_document(pdf_name)
                st.session_state["db_built"] = db.is_index_built()
                st.toast(f"✅ '{pdf_name}' を削除しました（{removed}チャンク）", icon="🗑️")
                st.rerun()

    # ── 製品別チャンク数 ─────────────────────────────────────
    if by_product:
        st.markdown("#### 製品別チャンク数（技術資料 + PDF）")
        max_cnt = max(by_product.values()) if by_product else 1
        for pname, cnt in by_product.items():
            bar_len = max(1, int(cnt / max_cnt * 20))
            st.markdown(f"**{pname}** `{'█' * bar_len}` {cnt}件")

    st.divider()

    # ── 検索テスト ──────────────────────────────────────────
    st.markdown("### 🔍 検索テスト")
    st.caption("任意のクエリを入力して、どのチャンクがヒットするか確認できます")

    _col1, _col2 = st.columns([3, 1])
    kb_query = _col1.text_input(
        "検索クエリ",
        placeholder="例: プレス機の異常検知",
        key="kb_search_query",
    )
    kb_filter = _col2.selectbox(
        "フィルタ",
        options=["all", "tech", "crm"],
        format_func=lambda x: {"all": "全て", "tech": "技術資料", "crm": "CRM記録"}[x],
        key="kb_search_filter",
    )

    if st.button("🔍 検索", key="kb_search_btn", disabled=not kb_query):
        with st.spinner("検索中..."):
            hits = db.search_for_display(kb_query, source_type_filter=kb_filter, top_k=5)

        if hits:
            max_s = max(r.get("score", 0) for r in hits) or 1.0
            for idx, r in enumerate(hits, 1):
                raw = r.get("score", 0)
                norm = raw / max_s
                lbl = "高" if norm >= 0.7 else ("中" if norm >= 0.4 else "低")
                color = _SCORE_COLORS.get(lbl, "#aaa")
                src_type = r["metadata"].get("source_type", "")
                src_file = r["metadata"].get("source_file", "")
                st.markdown(
                    f"**[{idx}] {src_file}** &nbsp; `{src_type}` &nbsp; "
                    f'スコア: <span style="color:{color}">●</span> **{lbl}**'
                    f" ({raw:.4f})",
                    unsafe_allow_html=True,
                )
                has_parent = r.get("has_parent", False)
                with st.expander(f"テキスト表示 [{idx}]" + (" 🔗親チャンク拡張" if has_parent else "")):
                    child_txt = r.get("child_text", "")
                    parent_txt = r.get("text", "")
                    if child_txt and has_parent:
                        st.caption("**ヒットした子チャンク（検索対象）**")
                        st.text(child_txt[:300])
                        st.caption("**親チャンク（LLMへ渡すコンテキスト）**")
                        st.text(parent_txt[:600])
                        if len(parent_txt) > 600:
                            st.caption(f"...（全 {len(parent_txt)} 文字）")
                    else:
                        st.text(parent_txt[:500])
                        if len(parent_txt) > 500:
                            st.caption(f"...（全 {len(parent_txt)} 文字）")
        else:
            st.info("検索結果が見つかりませんでした。")


# ---------------------------------------------------------------
# メイン
# ---------------------------------------------------------------
def main() -> None:
    """Streamlitアプリのメイン関数"""

    st.set_page_config(
        page_title="展示会フォローアップエージェント",
        page_icon="🏭",
        layout="wide",
    )

    _init_session_state()

    if not _initialize_components():
        st.stop()

    st.title("🏭 展示会フォローアップエージェント")
    st.caption(
        "製造業DX展示会のリード情報をもとに、商談確度・関心製品・過去商談記録を踏まえた"
        "パーソナライズされたフォローアップメールを自動生成します。"
    )

    # ── プライバシー通知（セッション初回のみ表示）─────────────────
    if not st.session_state.get("privacy_notice_acknowledged", False):
        with st.warning("", icon="🔒"):
            st.markdown(
                "**【データ取り扱いに関するご確認】**\n\n"
                "このアプリはメール生成のために、アップロードされた情報（来場者氏名・メールアドレス・"
                "会社名・商談メモ・CRM情報など）を **OpenAI API** に送信します。\n\n"
                "送信前に、取り扱う顧客情報が **貴社の情報セキュリティポリシー** に照らして"
                "外部サービスへの送信が許可されていることをご確認ください。\n\n"
                "社外秘・機密情報を含む場合は、IT担当者または上長にご相談ください。"
            )
            if st.button("✅ 確認しました", key="privacy_ack_btn", type="primary"):
                st.session_state["privacy_notice_acknowledged"] = True
                st.rerun()

    # サイドバーを描画して選択ランクを取得
    selected_ranks = _render_sidebar()

    # ── メインエリアの表示切り替え ──────────────────────────────
    leads_df: Optional[pd.DataFrame] = st.session_state.get("leads_df")
    mapping_confirmed: bool = st.session_state.get("mapping_confirmed", False)
    raw_uploaded_df: Optional[pd.DataFrame] = st.session_state.get("raw_uploaded_df")

    if not mapping_confirmed and raw_uploaded_df is None:
        # ① データ未読み込み: ウェルカム画面を表示
        _render_welcome()

    elif not mapping_confirmed and raw_uploaded_df is not None:
        # ② CSVアップロード済みだがマッピング未確定: マッピングUIを表示
        _render_column_mapping()

    else:
        # ③ マッピング確定済み: 4タブUIを表示
        if st.session_state.get("raw_uploaded_df") is not None:
            _redo_col, _ = st.columns([2, 5])
            if _redo_col.button("↩️ カラムマッピングをやり直す", key="redo_mapping_main"):
                st.session_state["mapping_confirmed"] = False
                st.rerun()

        # ── 次ステップ案内バナー ─────────────────────────────────
        if st.session_state.get("show_next_step_kb") and not st.session_state.get("db_built"):
            st.info(
                "✅ **ステップ①完了！** 次は左サイドバーの "
                "**「🔨 ① ナレッジベース構築」** ボタンを押してください。\n\n"
                "製品資料（Markdown / PDF）をAIが読み込み、メール生成の精度が上がります。"
            )
        elif st.session_state.get("db_built") and not st.session_state.get("results"):
            st.success(
                "✅ **ステップ②完了！** **「✉️ ③ メール生成」** タブでフォローアップメールを生成できます。"
            )

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 ① リード確認",
            "🗄️ ② KB管理",
            "✉️ ③ メール生成",
            "📊 生成履歴・ダウンロード",
            "🎙️ 音声管理",
        ])

        with tab1:
            _render_tab_leads(leads_df, selected_ranks)
        with tab2:
            _render_tab_knowledge()
        with tab3:
            _render_tab_email(leads_df, selected_ranks)
        with tab4:
            _render_tab_history()
        with tab5:
            _render_tab_audio(leads_df)


if __name__ == "__main__":
    main()
