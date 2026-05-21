"""
VLM（vision model）を使った PDF テキスト化モジュール。

PyMuPDF でページごとに画像化し、OpenAI vision API で Markdown テキストとして抽出する。
"""

import base64
import logging
import os

logger = logging.getLogger(__name__)

_PAGE_WARNING_THRESHOLD = 10


def is_pdf(file_path: str) -> bool:
    """ファイルが PDF かどうかを拡張子で判定する"""
    return file_path.lower().endswith(".pdf")


def extract_text_from_pdf_vlm(pdf_path: str, client, model: str) -> str:
    """
    PDF ファイルを VLM でテキスト化する。

    PyMuPDF (fitz) でページごとに PNG 画像に変換（解像度 2x）し、
    OpenAI vision API に送信して Markdown テキストとして抽出する。
    全ページの結果を結合して返す。

    Parameters
    ----------
    pdf_path : str
        対象 PDF ファイルのパス
    client : openai.OpenAI
        OpenAI クライアントインスタンス
    model : str
        使用するビジョンモデル名（例: "gpt-5.4-nano"）

    Returns
    -------
    str
        全ページ分の Markdown テキスト（ページ間を --- で区切る）

    Raises
    ------
    ImportError
        PyMuPDF (pymupdf) がインストールされていない場合
    """
    try:
        import fitz  # pymupdf
    except ImportError:
        raise ImportError(
            "PDF の VLM 処理には PyMuPDF が必要です。\n"
            "インストール: .venv/Scripts/pip install pymupdf"
        )

    doc = fitz.open(pdf_path)
    page_count = len(doc)
    filename = os.path.basename(pdf_path)

    if page_count > _PAGE_WARNING_THRESHOLD:
        print(f"  警告: {filename} は {page_count} ページあります。処理に時間とコストがかかります。")

    # コスト見積もり表示（vision API の入力トークン概算）
    est_tokens_per_page = 500
    try:
        from src.config import Config
        price_per_1m = Config.LLM_PRICE_INPUT_PER_1M
    except Exception:
        price_per_1m = 0.20
    est_cost_usd = page_count * est_tokens_per_page * price_per_1m / 1_000_000
    print(f"  {page_count} ページ × 約 {est_tokens_per_page} tokens ≒ 約 ${est_cost_usd:.4f}")

    prompt = (
        "この PDF ページの内容を Markdown 形式でテキスト化してください。"
        "表は Markdown テーブル形式で、図・画像は [図: 内容の説明] の形式で記載してください。"
        "ページ番号やヘッダー・フッターは除外してください。"
    )

    pages_text = []
    for page_num in range(page_count):
        try:
            page = doc[page_num]
            mat = fitz.Matrix(2.0, 2.0)  # 2x 解像度でレンダリング
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            b64_image = base64.b64encode(img_bytes).decode("utf-8")

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64_image}",
                                    "detail": "high",
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=2000,
            )
            page_text = response.choices[0].message.content or ""
            pages_text.append(page_text)
            print(f"  ページ {page_num + 1}/{page_count} 完了")
        except Exception as e:
            logger.warning("ページ %d の処理に失敗しました: %s", page_num + 1, e)
            print(f"  警告: ページ {page_num + 1} をスキップしました: {e}")

    doc.close()
    return "\n\n---\n\n".join(pages_text)
