"""
音声紐づけモジュール

音声ファイルをリードCSVの来場者に自動紐づけする。
ファイル名から担当者名を解析し、タイムスタンプ + 順番で照合する。

ファイル命名規則: YYYYMMDD_担当者名_連番.mp3 (例: 20260424_営業A_001.mp3)
"""

import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from rapidfuzz import fuzz

from src.utils import setup_logger

logger = setup_logger(__name__)

# YYYYMMDD_担当者名_連番.ext または 担当者名_連番.ext
# 貪欲マッチ(.*)で最後の「_連番」の直前までを担当者名として取得する。
# これにより「20260424_営業部_山田太郎_001.mp3」→「営業部_山田太郎」と正しく解析される。
_FILENAME_PATTERN = re.compile(
    r"^(?:\d{6,8}[_\-])?(.*)[_\-]\d+\.[a-zA-Z0-9]+$"
)

# YYYYMMDD_担当者名_任意テキスト.csv（紐づけCSV用）
_CSV_FILENAME_PATTERN = re.compile(
    r"^(?:\d{6,8}[_\-])?(.*)[_\-][^_\-]+\.[a-zA-Z0-9]+$"
)


@dataclass
class MatchResult:
    """音声ファイルとリードの紐づけ結果"""
    audio_filename: str
    rep_name: Optional[str]        # ファイル名から解析した担当者名
    lead_idx: Optional[int]        # リードDataFrameのインデックス（未紐づけは None）
    confidence: str                # "green" | "yellow" | "red"
    method: str                    # "timestamp" | "sequential" | "unmatched"
    note: str = ""                 # UI表示用メモ
    duration_sec: float = 0.0     # 録音時間（複数候補の優先判定に使用）
    manually_set: bool = False     # 手動確定フラグ


class AudioMatcher:
    """
    音声ファイル ↔ リード自動紐づけエンジン。

    crm_matcher.py と同じクラス設計パターンを踏襲。
    """

    def parse_rep_from_filename(self, filename: str) -> Optional[str]:
        """
        ファイル名から担当者名を抽出する。

        対応フォーマット:
        - 20260424_営業A_001.mp3  → "営業A"
        - 営業B_002.wav           → "営業B"
        - untitled.mp3            → None
        """
        # 拡張子を除いたベース名で照合
        basename = filename.rsplit(".", 1)[0] if "." in filename else filename
        m = _FILENAME_PATTERN.match(basename + ".ext")  # パターンには拡張子が必要なため付与
        if m:
            rep = m.group(1).strip()
            return rep if rep else None
        return None

    def parse_rep_from_csv_filename(self, filename: str) -> Optional[str]:
        """
        紐づけCSVファイル名から担当者名を抽出する。

        対応フォーマット:
        - 20260425_営業A_紐づけ.csv  → "営業A"
        - 営業B_mapping.csv          → "営業B"
        """
        basename = filename.rsplit(".", 1)[0] if "." in filename else filename
        m = _CSV_FILENAME_PATTERN.match(basename + ".ext")
        if m:
            rep = m.group(1).strip()
            return rep if rep else None
        return None

    def match(
        self,
        audio_meta_list: List[Dict],
        leads_df: pd.DataFrame,
        rep_col: Optional[str] = None,
        timestamp_col: Optional[str] = None,
        tolerance_minutes: int = 10,
    ) -> List[MatchResult]:
        """
        音声ファイルリストをリードDataFrameに紐づける。

        Parameters
        ----------
        audio_meta_list : list of dict
            [{filename, duration_sec, start_time, file_bytes}, ...]
        leads_df : pd.DataFrame
            標準化済みリードDataFrame
        rep_col : str, optional
            リードCSVの担当者名列名
        timestamp_col : str, optional
            リードCSVのQRスキャン時刻列名
        tolerance_minutes : int
            タイムスタンプ一致の許容誤差（分）

        Returns
        -------
        list of MatchResult
        """
        results: List[MatchResult] = []

        if not audio_meta_list or leads_df is None or leads_df.empty:
            return results

        # 担当者別に音声ファイルをグループ化
        rep_groups: Dict[Optional[str], List[Dict]] = {}
        for meta in audio_meta_list:
            rep = self.parse_rep_from_filename(meta["filename"])
            rep_groups.setdefault(rep, []).append(meta)

        # 担当者別にリードをグループ化（rep_col が指定されている場合）
        lead_rep_groups: Dict[Optional[str], List[int]] = {}
        if rep_col and rep_col in leads_df.columns:
            for idx, row in leads_df.iterrows():
                rep = str(row[rep_col]).strip() if pd.notna(row[rep_col]) else None
                lead_rep_groups.setdefault(rep, []).append(idx)
        else:
            lead_rep_groups[None] = list(leads_df.index)

        for rep_name, audios in rep_groups.items():
            # 担当者名が不明 → 全リードを対象にするが信頼度は red
            if rep_name is None:
                for meta in audios:
                    results.append(MatchResult(
                        audio_filename=meta["filename"],
                        rep_name=None,
                        lead_idx=None,
                        confidence="red",
                        method="unmatched",
                        note="ファイル名から担当者名を解析できません",
                        duration_sec=meta.get("duration_sec", 0.0),
                    ))
                continue

            # 担当者名が一致するリードを取得（完全一致 → ファジーマッチの順で試みる）
            candidate_idxs = lead_rep_groups.get(rep_name, [])
            if not candidate_idxs:
                # 完全一致なし → ファジーマッチ（表記ゆれ対応: "営業Aさん" ≈ "営業A"）
                best_key = _fuzzy_find_rep(rep_name, list(lead_rep_groups.keys()))
                if best_key is not None:
                    candidate_idxs = lead_rep_groups[best_key]
                else:
                    # 担当者名が CSV に存在しない → 全リードを候補に (yellow)
                    candidate_idxs = list(leads_df.index)

            # 担当者別に音声をソート: start_time 優先、なければファイル名の連番
            audios_sorted = sorted(
                audios,
                key=lambda m: (
                    0 if m.get("start_time") else 1,
                    m.get("start_time") or datetime.min,
                    _extract_seq_num(m["filename"]),
                ),
            )

            # タイムスタンプによる紐づけを試みる
            used_lead_idxs: set = set()
            # filename → (lead_idx, timedelta) : 時刻差も保持して重複時に比較する
            timestamp_matched: Dict[str, tuple] = {}

            if (
                timestamp_col
                and timestamp_col in leads_df.columns
                and any(m.get("start_time") for m in audios_sorted)
            ):
                leads_sorted = sorted(
                    candidate_idxs,
                    key=lambda i: _parse_dt(leads_df.loc[i, timestamp_col])
                    or datetime.min,
                )
                tolerance = timedelta(minutes=tolerance_minutes)

                for meta in audios_sorted:
                    audio_start = meta.get("start_time")
                    if not audio_start:
                        continue
                    best_idx = None
                    best_delta = None
                    for lead_idx in leads_sorted:
                        if lead_idx in used_lead_idxs:
                            continue
                        scan_dt = _parse_dt(leads_df.loc[lead_idx, timestamp_col])
                        if not scan_dt:
                            continue
                        delta = abs(scan_dt - audio_start)
                        if delta <= tolerance:
                            if best_delta is None or delta < best_delta:
                                best_delta = delta
                                best_idx = lead_idx
                    if best_idx is not None:
                        # 同一リードへの複数候補 → タイムスタンプが近い方を採用
                        existing_entry = next(
                            ((f, li, d) for f, (li, d) in timestamp_matched.items() if li == best_idx),
                            None,
                        )
                        if existing_entry is not None:
                            existing_fname, _, existing_delta = existing_entry
                            if best_delta < existing_delta:
                                del timestamp_matched[existing_fname]
                                used_lead_idxs.discard(best_idx)
                            else:
                                continue
                        timestamp_matched[meta["filename"]] = (best_idx, best_delta)
                        used_lead_idxs.add(best_idx)

            # 残りは順番で紐づけ。scan_time 列があれば時刻昇順でソートして対応順を揃える
            if timestamp_col and timestamp_col in leads_df.columns:
                sequential_idxs = sorted(
                    [i for i in candidate_idxs if i not in used_lead_idxs],
                    key=lambda i: _parse_dt(leads_df.loc[i, timestamp_col]) or datetime.min,
                )
            else:
                sequential_idxs = [i for i in candidate_idxs if i not in used_lead_idxs]
            sequential_queue = list(sequential_idxs)

            for meta in audios_sorted:
                fname = meta["filename"]
                if fname in timestamp_matched:
                    matched_lead_idx, matched_delta = timestamp_matched[fname]
                    results.append(MatchResult(
                        audio_filename=fname,
                        rep_name=rep_name,
                        lead_idx=matched_lead_idx,
                        confidence="green",
                        method="timestamp",
                        note=f"タイムスタンプ一致（時刻差{int(matched_delta.total_seconds())}秒）",
                        duration_sec=meta.get("duration_sec", 0.0),
                    ))
                elif sequential_queue:
                    lead_idx = sequential_queue.pop(0)
                    has_ts = bool(meta.get("start_time"))
                    results.append(MatchResult(
                        audio_filename=fname,
                        rep_name=rep_name,
                        lead_idx=lead_idx,
                        confidence="yellow",
                        method="sequential",
                        note="担当者名一致・順番で仮紐づけ" + ("（タイムスタンプなし）" if not has_ts else ""),
                        duration_sec=meta.get("duration_sec", 0.0),
                    ))
                else:
                    results.append(MatchResult(
                        audio_filename=fname,
                        rep_name=rep_name,
                        lead_idx=None,
                        confidence="yellow",
                        method="sequential",
                        note="紐づけ候補となるリードがありません",
                        duration_sec=meta.get("duration_sec", 0.0),
                    ))

        logger.info(
            f"音声紐づけ完了: {len(results)}件 / "
            f"緑={sum(1 for r in results if r.confidence=='green')} / "
            f"黄={sum(1 for r in results if r.confidence=='yellow')} / "
            f"赤={sum(1 for r in results if r.confidence=='red')}"
        )
        return results

    def match_with_csv(
        self,
        mapping_df: pd.DataFrame,
        rep_name: str,
        audio_meta_list: List[Dict],
        leads_df: pd.DataFrame,
    ) -> List["MatchResult"]:
        """
        紐づけCSV（filename, visitor_name 2列）を使って音声ファイルとリードを照合する。

        visitor_name の照合: 完全一致 → rapidfuzz partial_ratio >= 85 のファジーマッチ。
        audio_meta_list に存在しない filename はwarningログを出してスキップする。
        leads_df に存在しない visitor_name は confidence="yellow" で返す。
        """
        results: List[MatchResult] = []
        if mapping_df is None or mapping_df.empty or leads_df is None or leads_df.empty:
            return results

        audio_by_filename: Dict[str, Dict] = {m["filename"]: m for m in audio_meta_list}

        lead_name_to_idx: Dict[str, int] = {}
        for idx, row in leads_df.iterrows():
            name = str(row.get("visitor_name", "")).strip()
            if name:
                lead_name_to_idx[name] = idx

        lead_names = list(lead_name_to_idx.keys())

        for _, row in mapping_df.iterrows():
            filename = str(row.get("filename", "")).strip()
            visitor_name = str(row.get("visitor_name", "")).strip()

            if not filename or not visitor_name:
                continue

            if filename not in audio_by_filename:
                logger.warning(f"紐づけCSVのファイル名が音声リストにありません: {filename}")
                continue

            meta = audio_by_filename[filename]
            duration_sec = meta.get("duration_sec", 0.0)

            lead_idx: Optional[int] = lead_name_to_idx.get(visitor_name)
            confidence = "green"
            note = "紐づけCSVによる確定"

            if lead_idx is None:
                best_name = _fuzzy_find_rep(visitor_name, lead_names, threshold=85)
                if best_name is not None:
                    lead_idx = lead_name_to_idx[best_name]
                    note = f"紐づけCSV（ファジーマッチ: {best_name}）"
                else:
                    confidence = "yellow"
                    note = "リストに該当者なし"

            results.append(MatchResult(
                audio_filename=filename,
                rep_name=rep_name,
                lead_idx=lead_idx,
                confidence=confidence,
                method="manual_csv",
                note=note,
                duration_sec=duration_sec,
            ))

        logger.info(
            f"CSV紐づけ完了（{rep_name}）: {len(results)}件 / "
            f"緑={sum(1 for r in results if r.confidence=='green')} / "
            f"黄={sum(1 for r in results if r.confidence=='yellow')}"
        )
        return results

    def detect_timestamp_col(self, leads_df: pd.DataFrame, llm=None) -> Optional[str]:
        """
        leads_df からスキャン時刻列を自動検出する。

        1. "scan_time" 列があればそのまま返す（auto_map_columns 済みの場合）
        2. datetime 解析可能な列を全試行して候補を絞る
        3. 候補が複数ある場合、キーワードヒューリスティックを試みる
        4. それでも絞れない場合、LLM で判定する（llm 指定時）
        """
        if "scan_time" in leads_df.columns:
            return "scan_time"

        datetime_cols = [
            col for col in leads_df.columns
            if any(_parse_dt(v) is not None for v in leads_df[col].dropna().head(5))
        ]

        if not datetime_cols:
            return None
        if len(datetime_cols) == 1:
            return datetime_cols[0]

        # キーワードヒューリスティック
        priority_keywords = ["時刻", "スキャン", "scan", "visit", "timestamp", "来場", "受付"]
        for col in datetime_cols:
            if any(kw.lower() in col.lower() for kw in priority_keywords):
                logger.info(f"タイムスタンプ列をキーワードで検出: {col}")
                return col

        # LLM 判定
        if llm:
            return self._llm_detect_timestamp_col(datetime_cols, leads_df, llm)

        logger.info(f"タイムスタンプ列を先頭候補で代用: {datetime_cols[0]}")
        return datetime_cols[0]

    def _llm_detect_timestamp_col(
        self, datetime_cols: List[str], leads_df: pd.DataFrame, llm
    ) -> Optional[str]:
        """LLM を使ってスキャン時刻列を特定する。"""
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            samples = {
                col: leads_df[col].dropna().head(3).tolist()
                for col in datetime_cols
            }
            sample_text = "\n".join(
                f'- "{col}": {vals}' for col, vals in samples.items()
            )
            messages = [
                SystemMessage(content=(
                    "あなたは展示会リード管理システムのアシスタントです。"
                    "以下の列名と値のサンプルを見て、QRスキャン時刻・来場時刻を表す列名を1つだけ答えてください。"
                    "列名だけを返してください（他のテキスト不要）。"
                )),
                HumanMessage(content=f"候補列:\n{sample_text}"),
            ]
            response = llm.invoke(messages)
            detected = response.content.strip().strip('"').strip("'")
            if detected in datetime_cols:
                logger.info(f"タイムスタンプ列をLLMで検出: {detected}")
                return detected
        except Exception as e:
            logger.warning(f"タイムスタンプ列のLLM検出に失敗: {e}")
        return datetime_cols[0]

    def get_red_flag_rate(self, results: List[MatchResult]) -> float:
        """赤フラグ率を返す（0.0 〜 1.0）"""
        if not results:
            return 0.0
        return sum(1 for r in results if r.confidence == "red") / len(results)

    def detect_gaps(
        self,
        audio_results: List[MatchResult],
        leads_df: pd.DataFrame,
        rep_col: Optional[str] = None,
        mapping_dfs: Optional[List[pd.DataFrame]] = None,
    ) -> List[dict]:
        """
        担当者ごとにリード件数 vs 音声件数を比較し録音し忘れを検出する。

        mapping_dfs が指定された場合（CSVモード）:
          各DataFrameのvisitor_name列に載っていないリードを録音なしとみなす。
          rep_col がない場合は全担当者まとめて比較する。

        mapping_dfs が None の場合（既存ロジック）:
          matched_idxs（audio_results由来）と rep_col ベースで比較する。
        """
        if leads_df is None or leads_df.empty:
            return []

        if mapping_dfs is not None:
            csv_names: set = set()
            for df in mapping_dfs:
                if "visitor_name" in df.columns:
                    csv_names.update(str(v).strip() for v in df["visitor_name"].dropna())

            def _is_covered(name: str) -> bool:
                if name in csv_names:
                    return True
                return any(fuzz.partial_ratio(name, n) >= 85 for n in csv_names)

            if rep_col and rep_col in leads_df.columns:
                rep_grps: Dict[str, list] = {}
                for idx, row in leads_df.iterrows():
                    rep = str(row[rep_col]).strip() if pd.notna(row[rep_col]) else None
                    if rep:
                        rep_grps.setdefault(rep, []).append(idx)
                groups = list(rep_grps.items())
            else:
                groups = [("全担当者", list(leads_df.index))]

            gaps = []
            for grp_name, lead_idxs in groups:
                unrecorded = []
                for idx in lead_idxs:
                    name = str(leads_df.loc[idx].get("visitor_name", "")).strip()
                    if name and not _is_covered(name):
                        unrecorded.append(name)
                if unrecorded:
                    lead_count = len(lead_idxs)
                    gaps.append({
                        "rep_name": grp_name,
                        "lead_count": lead_count,
                        "audio_count": lead_count - len(unrecorded),
                        "missing_count": len(unrecorded),
                        "likely_unrecorded": unrecorded,
                    })
            return gaps

        # 既存ロジック（後方互換）
        if not rep_col or rep_col not in leads_df.columns:
            return []

        rep_to_lead_idxs: Dict[str, list] = {}
        for idx, row in leads_df.iterrows():
            rep = str(row[rep_col]).strip() if pd.notna(row[rep_col]) else None
            if rep:
                rep_to_lead_idxs.setdefault(rep, []).append(idx)

        matched_idxs: set = {r.lead_idx for r in audio_results if r.lead_idx is not None}

        gaps = []
        for rep_name, lead_idxs in rep_to_lead_idxs.items():
            lead_count = len(lead_idxs)
            audio_count = sum(1 for idx in lead_idxs if idx in matched_idxs)
            if audio_count >= lead_count:
                continue
            unrecorded = [
                str(leads_df.loc[idx].get("visitor_name", ""))
                for idx in lead_idxs
                if idx not in matched_idxs and str(leads_df.loc[idx].get("visitor_name", ""))
            ]
            gaps.append({
                "rep_name": rep_name,
                "lead_count": lead_count,
                "audio_count": audio_count,
                "missing_count": lead_count - audio_count,
                "likely_unrecorded": unrecorded,
            })
        return gaps


def _fuzzy_find_rep(rep_name: str, keys: List[Optional[str]], threshold: int = 80) -> Optional[str]:
    """
    担当者名の表記ゆれを吸収するファジーマッチ。
    "営業Aさん" → "営業A" のようなケースを score 80以上で一致とみなす。
    None キーはスキップ。
    """
    best_key = None
    best_score = 0
    for key in keys:
        if key is None:
            continue
        score = fuzz.partial_ratio(rep_name, key)
        if score >= threshold and score > best_score:
            best_score = score
            best_key = key
    return best_key


def _extract_seq_num(filename: str) -> int:
    """ファイル名末尾の連番を返す（001→1）。取得できない場合は sys.maxsize。"""
    m = re.search(r'[_\-](\d+)\.[a-zA-Z0-9]+$', filename)
    return int(m.group(1)) if m else sys.maxsize


def _parse_dt(value) -> Optional[datetime]:
    """各種フォーマットの日時文字列を datetime に変換する"""
    if value is None or (isinstance(value, float) and value != value):
        return None
    if isinstance(value, datetime):
        return value
    try:
        ts = pd.to_datetime(str(value))
        return ts.to_pydatetime()
    except Exception:
        return None
