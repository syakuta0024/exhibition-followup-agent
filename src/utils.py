"""
ユーティリティモジュール

データ読み込み・前処理・ログ出力など、汎用的なヘルパー関数を提供する。
"""

import logging
import os
from typing import Any, Dict, List

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
