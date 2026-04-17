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
    filter_leads_by_rank,
    format_lead_summary,
    load_csv_with_encoding,
    load_leads,
)
from src.vectordb import VectorDBManager

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

    if st.session_state["agent"] is None:
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
        st.markdown("## 🏭 NTX株式会社")
        st.markdown("**展示会フォローアップシステム**")
        st.divider()

        # ── セクション1: データ読み込み ───────────────────────────
        st.markdown("#### 📁 データ読み込み")

        # CSVアップロード
        uploaded_file = st.file_uploader(
            "CSVファイルをアップロード",
            type=["csv"],
            help="Lead Manager, Q-PASS, Sansan等のエクスポートCSVに対応。\nUTF-8 / Shift_JIS / BOM付き UTF-8 をサポートします。",
        )

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

        if st.button("🔨 ナレッジベース構築", use_container_width=True):
            db: VectorDBManager = st.session_state["vectordb"]
            with st.spinner("ベクトルDBを構築中..."):
                try:
                    db.build_index(Config.TECH_DOCS_DIR, Config.CRM_RECORDS_DIR)
                    st.session_state["db_built"] = True
                    st.toast("✅ ナレッジベースを構築しました", icon="✅")
                    st.rerun()
                except Exception as e:
                    st.error(f"構築エラー: {e}")

        # ── PDF アップロード ──────────────────────────────────────
        st.markdown("#### 📄 製品資料PDFアップロード")
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
            has_import_error = False
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
                    except ImportError:
                        has_import_error = True
                        st.error(
                            "**pypdf がインストールされていません。**\n\n"
                            "新しいターミナルウィンドウで以下を実行してください:\n\n"
                            "```\npip install pypdf\n```\n\n"
                            "インストール後、Streamlitを再起動してください。"
                        )
                        break  # 全ファイルで同じエラーになるので中断
                    except Exception as e:
                        st.warning(f"⚠️ '{pdf_file.name}' の取り込みに失敗: {e}")
            if added_count > 0:
                st.session_state["db_built"] = True
                st.success(f"✅ {added_count}件のPDFを取り込みました（合計{total_chunks}チャンク）")
            elif not has_import_error and added_count == 0 and len(pdf_files) > 0:
                # ImportError以外の理由で0件の場合のみ表示
                pass  # 個別ファイルのメッセージで十分

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

        st.caption("NTX株式会社 展示会フォローアップシステム")

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
        "1. **CSVアップロード** または **デモデータ** を選択\n"
        "2. **カラムマッピング** を確認・調整して確定\n"
        "3. サイドバーで **ナレッジベース構築**\n"
        "4. **メール生成** タブでフォローアップメールを生成\n"
        "5. 生成結果を **CSV ダウンロード**"
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
                st.warning(
                    f"⚠️ 以下の必須フィールドがマッピングされていません: "
                    f"{', '.join(missing_required)}\n\n"
                    "このまま続行するとメール生成の品質が低下する場合があります。"
                )

            # マッピングを適用してDataFrameを標準化
            final_mapping = {**required_selections, **optional_selections}
            leads_df = apply_column_mapping(raw_df, final_mapping)

            # session_state に保存
            st.session_state["leads_df"] = leads_df
            st.session_state["mapping_confirmed"] = True
            st.session_state["results"] = []
            st.session_state["single_result"] = None
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

    # 表示列を決定（標準フィールド + extra_ カラム）
    standard_cols = {
        "lead_id": "ID", "visitor_name": "氏名", "company_name": "会社名",
        "department": "部署", "job_title": "役職", "lead_rank": "確度",
        "interested_products": "関心製品", "visit_date": "来場日",
    }
    # 存在するカラムのみ表示
    show_cols = {k: v for k, v in standard_cols.items() if k in filtered_df.columns}
    display_df = filtered_df[list(show_cols.keys())].rename(columns=show_cols)

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


# ---------------------------------------------------------------
# タブ2: メール生成
# ---------------------------------------------------------------
def _render_tab_email(leads_df: pd.DataFrame, selected_ranks: List[str]) -> None:
    """メール生成タブを描画する"""
    st.subheader("✉️ フォローアップメール生成")

    if not st.session_state["db_built"]:
        st.warning("⚠️ ナレッジベースが未構築です。サイドバーの「🔨 ナレッジベース構築」ボタンを押してください。")

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

    # 選択リード情報サマリー
    rank = selected_lead.get("lead_rank", "")
    badge_html = f'<span style="{RANK_BADGE_STYLE.get(rank, "")}">ランク {rank}</span>'
    st.markdown(
        f"**選択中:** {selected_lead.get('visitor_name', '')} 様　{badge_html}",
        unsafe_allow_html=True,
    )
    st.info(format_lead_summary(selected_lead))

    # メール生成ボタン
    if st.button("📧 メール生成", type="primary", key="single_gen"):
        if not st.session_state["db_built"]:
            st.error("ナレッジベースを先に構築してください。")
        else:
            agent: FollowUpAgent = st.session_state["agent"]
            # CRM CSV がある場合はファジーマッチング、ない場合は vectordb にフォールバック
            crm_df: Optional[pd.DataFrame] = st.session_state.get("crm_df")
            visitor = selected_lead.get("visitor_name", "")
            company = selected_lead.get("company_name", "")
            with st.spinner(f"{visitor}様（{company}）のメールを生成中..."):
                try:
                    result = agent.process_lead(selected_lead, crm_df=crm_df)
                    st.session_state["single_result"] = result
                    st.toast("✅ メール生成が完了しました", icon="✉️")
                except Exception as e:
                    st.error(f"メール生成エラー: {e}")

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

        with st.expander("📚 参照資料の詳細"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**参照した技術資料**")
                for f in result.get("ref_tech_docs", []) or ["なし"]:
                    st.markdown(f"- `{f}`")
            with col2:
                st.markdown("**参照したCRM記録**")
                for f in result.get("ref_crm", []) or ["なし"]:
                    st.markdown(f"- `{f}`")

    st.divider()

    # ── 全件一括生成 ──────────────────────────────────────────
    st.markdown("### 全件一括生成")
    st.caption(f"対象: {len(filtered_df)}件のリードに対してメールを一括生成します")

    if st.button("🔄 全件一括生成", key="batch_gen"):
        if not st.session_state["db_built"]:
            st.error("ナレッジベースを先に構築してください。")
        else:
            agent: FollowUpAgent = st.session_state["agent"]
            total = len(filtered_df)
            results = []
            progress_bar = st.progress(0, text="生成準備中...")
            status_text = st.empty()

            for i, (_, row) in enumerate(filtered_df.iterrows()):
                lead = row.to_dict()
                visitor = lead.get("visitor_name", "")
                company = lead.get("company_name", "")
                status_text.text(f"[{i+1}/{total}] {visitor}様（{company}）のメール生成中...")
                progress_bar.progress(i / total, text=f"生成中... {i}/{total}件")

                try:
                    crm_df: Optional[pd.DataFrame] = st.session_state.get("crm_df")
                    result = agent.process_lead(lead, crm_df=crm_df)
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
                    })

            progress_bar.progress(1.0, text=f"✅ 完了 {total}/{total}件")
            status_text.empty()
            st.session_state["results"] = results
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
    summary_rows = [
        {
            "ID": r.get("lead_id", ""),
            "氏名": r.get("visitor_name", ""),
            "会社名": r.get("company_name", ""),
            "ランク": r.get("lead_rank", ""),
            "宛先メール": r.get("email_to", ""),
            "件名": r.get("subject", ""),
            "ステータス": "✅ 正常" if r.get("subject") != "ERROR" else "❌ エラー",
        }
        for r in results
    ]
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
# メイン
# ---------------------------------------------------------------
def main() -> None:
    """Streamlitアプリのメイン関数"""

    st.set_page_config(
        page_title="NTX 展示会フォローアップエージェント",
        page_icon="🏭",
        layout="wide",
    )

    _init_session_state()

    if not _initialize_components():
        st.stop()

    st.title("🏭 NTX 展示会フォローアップエージェント")
    st.caption(
        "製造業DX展示会のリード情報をもとに、商談確度・関心製品・過去商談記録を踏まえた"
        "パーソナライズされたフォローアップメールを自動生成します。"
    )

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
        # ③ マッピング確定済み: 通常の3タブUIを表示
        tab1, tab2, tab3 = st.tabs(["📋 リード一覧", "✉️ メール生成", "📊 生成履歴・ダウンロード"])

        with tab1:
            _render_tab_leads(leads_df, selected_ranks)
        with tab2:
            _render_tab_email(leads_df, selected_ranks)
        with tab3:
            _render_tab_history()


if __name__ == "__main__":
    main()
