"""
ユーティリティモジュール

データ読み込み・前処理・ログ出力など、汎用的なヘルパー関数を提供する。
"""

import io
import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    ロガーを設定して返す。

    Parameters
    ----------
    name : str
        ロガー名
    level : int
        ログレベル（デフォルト: INFO）

    Returns
    -------
    logging.Logger
        設定済みロガーインスタンス
    """
    logger = logging.getLogger(name)

    # すでにハンドラが設定済みの場合は重複追加しない
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # コンソール出力ハンドラ
    handler = logging.StreamHandler()
    handler.setLevel(level)

    # フォーマット: 時刻 [ロガー名] レベル: メッセージ
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def load_leads(csv_path: str) -> pd.DataFrame:
    """
    leads.csvを読み込みDataFrameとして返す。

    Parameters
    ----------
    csv_path : str
        CSVファイルのパス

    Returns
    -------
    pd.DataFrame
        リードデータのDataFrame
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"leads.csvが見つかりません: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8", dtype=str)

    # 空白のトリミング
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    # NaN を空文字に統一
    df = df.fillna("")

    return df


def filter_leads_by_rank(df: pd.DataFrame, ranks: List[str]) -> pd.DataFrame:
    """
    商談確度でリードをフィルタリングする。

    Parameters
    ----------
    df : pd.DataFrame
        リードデータ
    ranks : List[str]
        対象の商談確度リスト（例: ['A', 'B']）

    Returns
    -------
    pd.DataFrame
        フィルタリング後のDataFrame
    """
    if not ranks:
        return df

    # 大文字に統一して比較
    upper_ranks = [r.upper() for r in ranks]
    return df[df["lead_rank"].str.upper().isin(upper_ranks)].reset_index(drop=True)


def parse_interested_products(products_str: str) -> List[str]:
    """
    カンマ区切りの製品文字列をリストに変換する。

    Parameters
    ----------
    products_str : str
        例: 'Sorani,EdgeGuard' または '"Sorani,EdgeGuard"'

    Returns
    -------
    List[str]
        製品名のリスト（空白トリム済み）
    """
    if not products_str:
        return []

    # 前後のクォートを除去してからカンマ分割
    cleaned = products_str.strip().strip('"').strip("'")
    return [p.strip() for p in cleaned.split(",") if p.strip()]


def format_lead_summary(lead: Dict[str, Any]) -> str:
    """
    リード情報を人間が読みやすい形式の文字列にフォーマットする。

    Parameters
    ----------
    lead : Dict[str, Any]
        リード情報の辞書

    Returns
    -------
    str
        フォーマット済みサマリー文字列
    """
    products = parse_interested_products(str(lead.get("interested_products", "")))
    products_str = "、".join(products) if products else "（なし）"

    lines = [
        f"【リードID】 {lead.get('lead_id', '')}",
        f"【氏名】     {lead.get('visitor_name', '')}",
        f"【会社名】   {lead.get('company_name', '')}",
        f"【部署・役職】{lead.get('department', '')} / {lead.get('job_title', '')}",
        f"【メール】   {lead.get('email', '')}",
        f"【商談確度】 {lead.get('lead_rank', '')}",
        f"【関心製品】 {products_str}",
        f"【今後の要望】{lead.get('future_requests', '')}",
        f"【営業メモ】 {lead.get('memo', '')}",
        f"【来場日】   {lead.get('visit_date', '')}",
    ]
    return "\n".join(lines)


def load_csv_with_encoding(file_obj: Any) -> pd.DataFrame:
    """
    アップロードされたファイルオブジェクトをエンコーディング自動判定して読み込む。

    UTF-8 BOM → UTF-8 → Shift_JIS → CP932 の順に試みる。
    BOM付きCSV（Excelエクスポート等）にも対応している。

    Parameters
    ----------
    file_obj : file-like object
        st.file_uploader 等で取得したファイルオブジェクト

    Returns
    -------
    pd.DataFrame
        読み込んだDataFrame（全カラムstr型、空白トリム済み）

    Raises
    ------
    ValueError
        全エンコーディングで読み込みに失敗した場合
    """
    # バイト列として一度読み込み、複数エンコーディングで再試行できるようにする
    raw_bytes = file_obj.read()

    # 試行するエンコーディングの優先順（BOM付きUTF-8を最初に試す）
    encodings = ["utf-8-sig", "utf-8", "shift_jis", "cp932"]

    for enc in encodings:
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), encoding=enc, dtype=str)
            # 空白のトリミング・NaN統一
            df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
            df = df.fillna("")
            return df
        except (UnicodeDecodeError, Exception):
            continue

    raise ValueError(
        "CSVファイルのエンコーディングを判定できませんでした。\n"
        "UTF-8 または Shift_JIS で保存されているか確認してください。"
    )


def auto_map_columns(df_columns: List[str], field_definitions: Dict) -> Dict[str, Optional[str]]:
    """
    CSVのカラム名リストとフィールド定義を受け取り、自動マッピングを推定して返す。

    マッチングロジック（優先順）:
    1. 完全一致（大文字小文字・前後空白を無視）
    2. 部分一致（カラム名に候補文字列が含まれる、または候補文字列がカラム名に含まれる）
    3. どちらもマッチしない場合は None

    1つのCSVカラムは1つのフィールドにのみマッピングされる（重複割り当て防止）。

    Parameters
    ----------
    df_columns : List[str]
        CSVのカラム名リスト
    field_definitions : Dict
        Config.REQUIRED_FIELDS または OPTIONAL_FIELDS の形式の辞書

    Returns
    -------
    Dict[str, Optional[str]]
        {"visitor_name": "氏名", "company_name": "会社名", "email": None, ...}
        マッチしなかったフィールドの値は None
    """
    mapping: Dict[str, Optional[str]] = {}
    # 既にマッピング済みのCSVカラムを追跡（重複割り当て防止）
    used_columns: set = set()

    for field_key, field_def in field_definitions.items():
        candidates: List[str] = field_def.get("候補カラム名", [])
        matched: Optional[str] = None

        # ── フェーズ1: 完全一致（大文字小文字無視）──────────────────
        for candidate in candidates:
            for col in df_columns:
                if col not in used_columns and col.strip().lower() == candidate.lower():
                    matched = col
                    break
            if matched:
                break

        # ── フェーズ2: 部分一致 ────────────────────────────────────
        if not matched:
            for candidate in candidates:
                for col in df_columns:
                    if col in used_columns:
                        continue
                    col_lower = col.strip().lower()
                    cand_lower = candidate.lower()
                    # カラム名が候補を含む、または候補がカラム名を含む
                    if cand_lower in col_lower or col_lower in cand_lower:
                        matched = col
                        break
                if matched:
                    break

        if matched:
            used_columns.add(matched)
        mapping[field_key] = matched

    return mapping


def apply_column_mapping(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    """
    マッピング辞書に基づいてDataFrameのカラム名をリネームし標準形式に変換する。

    処理内容:
    1. マッピング辞書に従って元カラム名 → 標準フィールド名にリネーム
    2. マッピングされなかった余分なカラムは「extra_元カラム名」として保持
       （独自アンケート項目等をメール生成コンテキストで活用可能にする）
    3. 標準フィールドのうち不足しているものは空文字カラムとして補完
    4. lead_id が存在しない場合は「L001, L002, ...」で自動採番

    Parameters
    ----------
    df : pd.DataFrame
        アップロードされた生のDataFrame
    mapping : Dict[str, Optional[str]]
        {標準フィールド名: 元CSVカラム名} の辞書（Noneはマッピングなし）

    Returns
    -------
    pd.DataFrame
        標準化されたDataFrame
    """
    # マッピングされたCSVカラム → 標準フィールド名 の逆引き辞書
    col_to_field: Dict[str, str] = {
        orig_col: field_key
        for field_key, orig_col in mapping.items()
        if orig_col is not None
    }

    # カラムリネーム辞書を構築
    rename_dict: Dict[str, str] = {}
    for col in df.columns:
        if col in col_to_field:
            # マッピング済み: 標準フィールド名に変換
            rename_dict[col] = col_to_field[col]
        else:
            # 未マッピング: extra_ プレフィックスを付けて保持
            # （アンケート回答等の追加情報としてメール生成に活用）
            rename_dict[col] = f"extra_{col}"

    result_df = df.rename(columns=rename_dict)

    # 全標準フィールドが存在することを確認（不足分は空文字で補完）
    from src.config import Config
    all_standard_fields = list(Config.REQUIRED_FIELDS.keys()) + list(Config.OPTIONAL_FIELDS.keys())
    for field in all_standard_fields:
        if field not in result_df.columns:
            result_df[field] = ""

    # lead_id が存在しない場合は自動採番（L001, L002, ...）
    if "lead_id" not in result_df.columns:
        result_df.insert(0, "lead_id", [f"L{i+1:03d}" for i in range(len(result_df))])

    return result_df


def save_results_to_csv(results: List[Dict], output_path: str) -> None:
    """
    メール生成結果をCSVファイルに保存する。

    Parameters
    ----------
    results : List[Dict]
        生成結果のリスト
    output_path : str
        保存先のCSVパス
    """
    if not results:
        raise ValueError("保存する結果データがありません。")

    # 出力先ディレクトリが存在しない場合は作成
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")  # BOM付きでExcel対応
    print(f"結果を保存しました: {output_path}（{len(results)}件）")
