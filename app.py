"""
Streamlitアプリケーション エントリーポイント

NTX株式会社 展示会フォローアップエージェントのWebUI。
リード一覧の表示・フォローアップメール生成・結果のダウンロードを提供する。
"""

from typing import Dict, List

import pandas as pd
import streamlit as st

from src.config import Config
from src.agent import FollowUpAgent
from src.email_generator import EmailGenerator
from src.utils import (
    filter_leads_by_rank,
    format_lead_summary,
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


# ---------------------------------------------------------------
# session_state の初期化
# ---------------------------------------------------------------
def _init_session_state() -> None:
    """アプリ起動時に session_state のキーを初期化する"""
    defaults = {
        "vectordb": None,       # VectorDBManager インスタンス
        "agent": None,          # FollowUpAgent インスタンス
        "email_gen": None,      # EmailGenerator インスタンス
        "db_built": False,      # インデックス構築済みフラグ
        "results": [],          # 一括生成結果リスト
        "single_result": None,  # 単一メール生成結果
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
    # APIキー未設定チェック
    if not Config.OPENAI_API_KEY:
        st.error(
            "⚠️ OPENAI_API_KEY が設定されていません。\n\n"
            "`.env` ファイルに `OPENAI_API_KEY=your-key` を設定してから再起動してください。"
        )
        return False

    # VectorDBManager 初期化（まだの場合のみ）
    if st.session_state["vectordb"] is None:
        st.session_state["vectordb"] = VectorDBManager()

    # EmailGenerator 初期化（まだの場合のみ）
    if st.session_state["email_gen"] is None:
        st.session_state["email_gen"] = EmailGenerator()

    # FollowUpAgent 初期化（まだの場合のみ）
    if st.session_state["agent"] is None:
        st.session_state["agent"] = FollowUpAgent(
            vectordb_manager=st.session_state["vectordb"],
            email_generator=st.session_state["email_gen"],
        )

    # インデックス構築状態を同期
    if st.session_state["vectordb"] is not None:
        st.session_state["db_built"] = st.session_state["vectordb"].is_index_built()

    return True


# ---------------------------------------------------------------
# サイドバー
# ---------------------------------------------------------------
def _render_sidebar(leads_df: pd.DataFrame) -> List[str]:
    """
    サイドバーを描画し、選択された商談確度リストを返す。

    Parameters
    ----------
    leads_df : pd.DataFrame
        全リードデータ（件数表示に使用）

    Returns
    -------
    List[str]
        選択された商談確度のリスト
    """
    with st.sidebar:
        # タイトル
        st.markdown("## 🏭 NTX株式会社")
        st.markdown("**展示会フォローアップシステム**")
        st.divider()

        # ナレッジベース構築ボタン
        st.markdown("#### ナレッジベース")

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

        st.divider()

        # 商談確度フィルター
        st.markdown("#### 商談確度フィルター")
        selected_ranks = st.multiselect(
            label="対象ランク",
            options=["A", "B", "C", "D", "E"],
            default=["A", "B", "C", "D", "E"],
            help="表示・処理対象の商談確度を選択してください",
        )

        # フィルタ後件数の表示
        if selected_ranks:
            filtered_count = len(filter_leads_by_rank(leads_df, selected_ranks))
            st.caption(f"対象リード: **{filtered_count}件** / 全{len(leads_df)}件")

        st.divider()
        st.caption("NTX株式会社 展示会フォローアップシステム")

    return selected_ranks


# ---------------------------------------------------------------
# タブ1: リード一覧
# ---------------------------------------------------------------
def _render_tab_leads(leads_df: pd.DataFrame, selected_ranks: List[str]) -> None:
    """リード一覧タブを描画する"""

    st.subheader("📋 展示会リード一覧")

    # フィルタリング
    filtered_df = filter_leads_by_rank(leads_df, selected_ranks) if selected_ranks else leads_df
    st.caption(f"表示件数: {len(filtered_df)}件（フィルタ: ランク {', '.join(selected_ranks) if selected_ranks else '全て'}）")

    if filtered_df.empty:
        st.info("選択した商談確度に該当するリードがありません。")
        return

    # 商談確度バッジをHTMLで付与した表示用DataFrameを作成
    display_df = filtered_df.copy()
    display_df["ランク"] = display_df["lead_rank"].apply(
        lambda r: f'<span style="{RANK_BADGE_STYLE.get(r, "")}">{r}</span>'
    )

    # 表示列の整理（カラム名を日本語化）
    show_df = display_df.rename(columns={
        "lead_id": "ID",
        "visitor_name": "氏名",
        "company_name": "会社名",
        "department": "部署",
        "job_title": "役職",
        "email": "メール",
        "lead_rank": "確度",
        "interested_products": "関心製品",
        "future_requests": "今後の要望",
        "visit_date": "来場日",
    })

    # st.dataframe で表示（HTMLレンダリングは非対応のためシンプルに表示）
    st.dataframe(
        show_df[[
            "ID", "氏名", "会社名", "部署", "役職",
            "確度", "関心製品", "今後の要望", "来場日"
        ]],
        use_container_width=True,
        hide_index=True,
    )

    # 商談確度の凡例バッジ
    st.markdown("**ランク凡例:**  " + "　".join(
        f'<span style="{RANK_BADGE_STYLE[r]}">{r}</span>'
        for r in ["A", "B", "C", "D", "E"]
    ), unsafe_allow_html=True)

    # ランク別件数のサマリー
    st.divider()
    cols = st.columns(5)
    for i, rank in enumerate(["A", "B", "C", "D", "E"]):
        cnt = len(filtered_df[filtered_df["lead_rank"] == rank])
        cols[i].metric(label=f"ランク {rank}", value=f"{cnt}件")


# ---------------------------------------------------------------
# タブ2: メール生成
# ---------------------------------------------------------------
def _render_tab_email(leads_df: pd.DataFrame, selected_ranks: List[str]) -> None:
    """メール生成タブを描画する"""

    st.subheader("✉️ フォローアップメール生成")

    # ナレッジベース未構築の警告
    if not st.session_state["db_built"]:
        st.warning("⚠️ ナレッジベースが未構築です。サイドバーの「ナレッジベース構築」ボタンを押してください。")

    filtered_df = filter_leads_by_rank(leads_df, selected_ranks) if selected_ranks else leads_df

    if filtered_df.empty:
        st.info("対象リードがありません。サイドバーのランクフィルターを確認してください。")
        return

    # ── 単一メール生成セクション ────────────────────────────────
    st.markdown("### 個別メール生成")

    # リード選択ドロップダウン
    lead_options = {
        f"{row['visitor_name']}（{row['company_name']}）[ランク{row['lead_rank']}]": idx
        for idx, row in filtered_df.iterrows()
    }
    selected_label = st.selectbox(
        "対象リードを選択",
        options=list(lead_options.keys()),
        help="メールを生成するリードを選択してください",
    )
    selected_idx = lead_options[selected_label]
    selected_lead = filtered_df.loc[selected_idx].to_dict()

    # 選択リードの情報サマリー
    with st.container():
        rank = selected_lead.get("lead_rank", "")
        badge_html = f'<span style="{RANK_BADGE_STYLE.get(rank, "")}">ランク {rank}</span>'
        st.markdown(f"**選択中:** {selected_lead.get('visitor_name')} 様　{badge_html}", unsafe_allow_html=True)
        st.info(format_lead_summary(selected_lead))

    # メール生成ボタン
    if st.button("📧 メール生成", type="primary", key="single_gen"):
        if not st.session_state["db_built"]:
            st.error("ナレッジベースを先に構築してください。")
        else:
            agent: FollowUpAgent = st.session_state["agent"]
            visitor = selected_lead.get("visitor_name", "")
            company = selected_lead.get("company_name", "")
            with st.spinner(f"{visitor}様（{company}）のメールを生成中..."):
                try:
                    result = agent.process_lead(selected_lead)
                    st.session_state["single_result"] = result
                    st.toast("✅ メール生成が完了しました", icon="✉️")
                except Exception as e:
                    st.error(f"メール生成エラー: {e}")

    # 生成結果の表示
    if st.session_state["single_result"]:
        result = st.session_state["single_result"]
        st.divider()
        st.markdown("#### 生成結果")

        # 件名（編集可能 ── 編集内容はUIに反映されるがセッション保存は行わない）
        st.text_input(
            "📌 件名",
            value=result.get("subject", ""),
            key="edit_subject",
        )

        # 本文（編集可能）
        st.text_area(
            "📝 本文",
            value=result.get("body", ""),
            height=400,
            key="edit_body",
        )

        # 営業アクション指示（編集可能 ── 社内向けの次のアクション）
        st.text_area(
            "🎯 営業アクション指示（社内向け）",
            value=result.get("cta", ""),
            height=100,
            key="edit_cta",
            help="LLMが生成した、営業担当者が次に実行すべきアクションの指示です。メール本文には含まれません。",
        )

        # 参照資料の表示
        with st.expander("📚 参照資料の詳細"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**参照した技術資料**")
                tech_docs = result.get("ref_tech_docs", [])
                if tech_docs:
                    for f in tech_docs:
                        st.markdown(f"- `{f}`")
                else:
                    st.caption("なし")
            with col2:
                st.markdown("**参照したCRM記録**")
                crm_docs = result.get("ref_crm", [])
                if crm_docs:
                    for f in crm_docs:
                        st.markdown(f"- `{f}`")
                else:
                    st.caption("なし")

    st.divider()

    # ── 全件一括生成セクション ────────────────────────────────
    st.markdown("### 全件一括生成")
    st.caption(f"対象: {len(filtered_df)}件のリードに対してメールを一括生成します")

    if st.button("🔄 全件一括生成", key="batch_gen"):
        if not st.session_state["db_built"]:
            st.error("ナレッジベースを先に構築してください。")
        else:
            agent: FollowUpAgent = st.session_state["agent"]
            total = len(filtered_df)
            results = []

            # プログレスバー
            progress_bar = st.progress(0, text="生成準備中...")
            status_text = st.empty()

            for i, (_, row) in enumerate(filtered_df.iterrows()):
                lead = row.to_dict()
                visitor = lead.get("visitor_name", "")
                company = lead.get("company_name", "")
                status_text.text(f"[{i+1}/{total}] {visitor}様（{company}）のメール生成中...")
                progress_bar.progress((i) / total, text=f"生成中... {i}/{total}件")

                try:
                    result = agent.process_lead(lead)
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

    # サマリーテーブルの表示
    summary_rows = []
    for r in results:
        summary_rows.append({
            "ID": r.get("lead_id", ""),
            "氏名": r.get("visitor_name", ""),
            "会社名": r.get("company_name", ""),
            "ランク": r.get("lead_rank", ""),
            "宛先メール": r.get("email_to", ""),
            "件名": r.get("subject", ""),
            "ステータス": "✅ 正常" if r.get("subject") != "ERROR" else "❌ エラー",
        })

    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.divider()

    # 各メールの詳細をexpanderで表示
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

    # ダウンロード用DataFrameを作成（ref_* はリストなので文字列化）
    download_rows = []
    for r in results:
        download_rows.append({
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
        })

    csv_df = pd.DataFrame(download_rows)
    # Excel で文字化けしないよう UTF-8 BOM 付きで出力
    csv_bytes = csv_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

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

    # ページ設定
    st.set_page_config(
        page_title="NTX 展示会フォローアップエージェント",
        page_icon="🏭",
        layout="wide",
    )

    # session_state 初期化
    _init_session_state()

    # コンポーネント初期化（APIキーチェック含む）
    if not _initialize_components():
        st.stop()  # APIキー未設定の場合はここで停止

    # タイトル
    st.title("🏭 NTX 展示会フォローアップエージェント")
    st.caption(
        "製造業DX展示会のリード情報をもとに、商談確度・関心製品・過去商談記録を踏まえた"
        "パーソナライズされたフォローアップメールを自動生成します。"
    )

    # リードデータ読み込み
    try:
        leads_df = load_leads(Config.LEADS_CSV_PATH)
    except FileNotFoundError as e:
        st.error(f"リードデータが見つかりません: {e}")
        st.stop()

    # サイドバーを描画して選択ランクを取得
    selected_ranks = _render_sidebar(leads_df)

    # 3タブ構成
    tab1, tab2, tab3 = st.tabs(["📋 リード一覧", "✉️ メール生成", "📊 生成履歴・ダウンロード"])

    with tab1:
        _render_tab_leads(leads_df, selected_ranks)

    with tab2:
        _render_tab_email(leads_df, selected_ranks)

    with tab3:
        _render_tab_history()


if __name__ == "__main__":
    main()
