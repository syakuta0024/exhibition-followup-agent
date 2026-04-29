"""
音声処理モジュール

Whisper API による音声文字起こしと、LLM によるニーズ構造化抽出を提供する。
"""

import io
import json
from datetime import datetime
from typing import Optional

import pandas as pd

import openai
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.utils import setup_logger

logger = setup_logger(__name__)

def _extract_recording_time(audio) -> Optional[datetime]:
    """
    複数フォーマット・タグから録音日時を抽出する。

    対応優先順:
    1. TDRC       — MP3/ID3v2.4（Android録音アプリ、ICレコーダー等）
    2. ©day       — M4A/MP4 iTunes形式（iPhone Voice Memo 等）
    3. TYER+TDAT  — MP3/ID3v2.3（古いファームウェアの録音機器）
    4. TDRL       — ID3v2.4 リリース日（TDRC が存在しない場合の代替）
    """
    if audio is None or audio.tags is None:
        return None
    tags = audio.tags

    def _parse(raw: str) -> Optional[datetime]:
        try:
            ts = pd.to_datetime(str(raw))
            return ts.to_pydatetime().replace(tzinfo=None)
        except Exception:
            return None

    # 1. TDRC (ID3v2.4 — MP3, WAV with embedded ID3)
    tdrc = tags.get("TDRC")
    if tdrc and hasattr(tdrc, "text") and tdrc.text:
        result = _parse(str(tdrc.text[0]))
        if result:
            return result

    # 2. ©day (MP4/M4A iTunes — iPhone Voice Memo など)
    day_tag = tags.get("©day")
    if day_tag:
        raw = day_tag[0] if isinstance(day_tag, (list, tuple)) else str(day_tag)
        result = _parse(str(raw))
        if result:
            return result

    # 3. TYER + TDAT + TIME (ID3v2.3 — 旧フォーマット)
    tyer = tags.get("TYER")
    if tyer and hasattr(tyer, "text") and tyer.text:
        year = str(tyer.text[0]).strip()
        date_str = year
        tdat = tags.get("TDAT")
        if tdat and hasattr(tdat, "text") and tdat.text:
            ddmm = str(tdat.text[0]).strip()  # ID3v2.3 TDAT は DDMM 形式
            if len(ddmm) == 4:
                date_str = f"{year}-{ddmm[2:]}-{ddmm[:2]}"  # DDMM → YYYY-MM-DD
        ttime_tag = tags.get("TIME")
        if ttime_tag and hasattr(ttime_tag, "text") and ttime_tag.text:
            hhmm = str(ttime_tag.text[0]).strip()  # HHMM 形式
            if len(hhmm) == 4:
                date_str += f" {hhmm[:2]}:{hhmm[2:]}"
        result = _parse(date_str)
        if result:
            return result

    # 4. TDRL (ID3v2.4 リリース日 — TDRC の代替として使われることがある)
    tdrl = tags.get("TDRL")
    if tdrl and hasattr(tdrl, "text") and tdrl.text:
        result = _parse(str(tdrl.text[0]))
        if result:
            return result

    return None


_NEEDS_EXTRACTION_SYSTEM = """あなたは展示会営業のアシスタントです。
来場者との会話録音の文字起こしを読み、以下の情報を日本語でJSON形式で抽出してください。
情報がない場合は空文字列 "" を返してください。

出力フォーマット（JSONのみ、他のテキスト不要）:
{
  "summary": "200文字以内の会話要約",
  "issues": "顧客が抱える課題・悩み",
  "needs": "顧客のニーズ・要望",
  "budget": "予算感・導入規模",
  "decision_maker": "決裁者・意思決定プロセスに関する情報",
  "temperature": "商談温度感（高/中/低）"
}"""


class AudioProcessor:
    """Whisper API による文字起こし + LLM によるニーズ抽出"""

    PRICE_PER_MIN: float = 0.006  # $/分（Whisper API）

    def __init__(self, api_key: str, llm: Optional[ChatOpenAI] = None):
        """
        Parameters
        ----------
        api_key : str
            OpenAI APIキー
        llm : ChatOpenAI, optional
            ニーズ抽出用LLM。None の場合はニーズ抽出は利用不可。
        """
        self._openai = openai.OpenAI(api_key=api_key)
        self._llm = llm

    # ------------------------------------------------------------------
    # メタデータ取得
    # ------------------------------------------------------------------

    def get_audio_metadata(self, file_bytes: bytes, filename: str) -> dict:
        """
        音声ファイルのメタデータを取得する。

        Parameters
        ----------
        file_bytes : bytes
            音声ファイルのバイト列
        filename : str
            ファイル名（拡張子で形式を判別）

        Returns
        -------
        dict
            {
              "duration_sec": float,            # 長さ（秒）
              "start_time": Optional[datetime], # 録音開始時刻（取得不可なら None）
              "size_mb": float,                 # ファイルサイズ（MB）
            }
        """
        try:
            from mutagen import File as MutagenFile  # オプション依存

            bio = io.BytesIO(file_bytes)
            audio = MutagenFile(fileobj=bio, filename=filename)
            duration_sec = float(audio.info.length) if audio is not None and audio.info is not None else 0.0
            start_time = _extract_recording_time(audio)

        except ImportError:
            logger.warning("mutagen が未インストールのため、音声メタデータを取得できません")
            duration_sec = 0.0
            start_time = None
        except Exception as e:
            logger.warning(f"音声メタデータ取得エラー ({filename}): {e}")
            duration_sec = 0.0
            start_time = None

        size_mb = len(file_bytes) / (1024 * 1024)
        return {"duration_sec": duration_sec, "start_time": start_time, "size_mb": size_mb}

    # ------------------------------------------------------------------
    # コスト見積もり
    # ------------------------------------------------------------------

    @classmethod
    def estimate_cost(cls, duration_sec: float) -> float:
        """文字起こしコストを概算する（ドル）"""
        return (duration_sec / 60) * cls.PRICE_PER_MIN

    # ------------------------------------------------------------------
    # 文字起こし
    # ------------------------------------------------------------------

    def transcribe(self, file_bytes: bytes, filename: str, language: str = "ja") -> str:
        """
        Whisper API で音声ファイルを文字起こしする。

        Parameters
        ----------
        file_bytes : bytes
            音声ファイルのバイト列
        filename : str
            ファイル名（APIへのヒントとして使用）
        language : str
            言語コード（デフォルト: "ja"）

        Returns
        -------
        str
            文字起こしテキスト

        Raises
        ------
        ValueError
            ファイルサイズが 25MB を超える場合
        RuntimeError
            API エラー
        """
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > 25:
            raise ValueError(
                f"ファイルサイズ {size_mb:.1f}MB が Whisper API の上限（25MB）を超えています。"
                "録音を短く分割してアップロードしてください。"
            )

        _MIME = {
            "mp3": "audio/mpeg", "mpga": "audio/mpeg", "mpeg": "audio/mpeg",
            "mp4": "audio/mp4",  "m4a": "audio/mp4",
            "wav": "audio/wav",
            "webm": "audio/webm",
            "ogg": "audio/ogg",
            "flac": "audio/flac",
        }
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        content_type = _MIME.get(ext, "audio/mpeg")

        try:
            bio = io.BytesIO(file_bytes)
            response = self._openai.audio.transcriptions.create(
                model="whisper-1",
                file=(filename, bio, content_type),
                language=language,
            )
            logger.info(f"文字起こし完了: {filename} ({len(response.text)}文字)")
            return response.text

        except Exception as e:
            logger.error(f"Whisper API エラー ({filename}): {e}")
            raise RuntimeError(f"文字起こしに失敗しました: {e}") from e

    # ------------------------------------------------------------------
    # ニーズ抽出
    # ------------------------------------------------------------------

    def extract_needs(self, transcript: str) -> dict:
        """
        文字起こしテキストから構造化ニーズを LLM で抽出する。

        Parameters
        ----------
        transcript : str
            文字起こしテキスト

        Returns
        -------
        dict
            {summary, issues, needs, budget, decision_maker, temperature}
            抽出できなかった項目は空文字列。
        """
        empty = {
            "summary": "", "issues": "", "needs": "",
            "budget": "", "decision_maker": "", "temperature": "",
        }
        if not self._llm or not transcript.strip():
            return empty

        try:
            messages = [
                SystemMessage(content=_NEEDS_EXTRACTION_SYSTEM),
                HumanMessage(content=f"【文字起こし】\n{transcript[:8000]}"),
            ]
            response = self._llm.invoke(messages)
            raw = response.content.strip()

            # JSON 部分だけ抽出（コードブロック対応）
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()

            result = json.loads(raw)
            logger.info("ニーズ抽出完了")
            return {k: str(result.get(k, "")) for k in empty}

        except json.JSONDecodeError as e:
            logger.warning(f"ニーズ抽出 JSON パースエラー: {e}")
            return empty
        except Exception as e:
            logger.error(f"ニーズ抽出エラー: {e}")
            return empty
