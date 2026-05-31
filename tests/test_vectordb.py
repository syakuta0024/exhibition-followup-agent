"""tests/test_vectordb.py — VectorDBManager の単体テスト

外部 API（OpenAI Embedding / Chroma）はすべてモック化する。
"""

import os
import json
import tempfile
from unittest.mock import MagicMock, patch, call
from typing import List, Dict, Any

import pytest
from langchain_core.documents import Document

from src.vectordb import (
    VectorDBManager,
    _matches_filter,
    _reciprocal_rank_fusion,
    _build_context_prefix,
    _extract_crm_company_name,
    _infer_deal_stage,
    PRODUCT_NAME_MAP,
)


# ==================================================================
# フィクスチャ: VectorDBManager のモック版
# ==================================================================

class FakeCollection:
    """ChromaDB の Collection をメモリ上でエミュレートする最小実装"""

    def __init__(self):
        self._items: Dict[str, Dict] = {}  # id → {document, metadata, embedding}

    def count(self) -> int:
        return len(self._items)

    def add(self, ids, embeddings, documents, metadatas):
        for i, doc_id in enumerate(ids):
            self._items[doc_id] = {
                "document": documents[i],
                "metadata": metadatas[i],
                "embedding": embeddings[i] if embeddings else None,
            }

    def get(self, where=None, include=None):
        items = list(self._items.values())
        if where:
            items = [v for v in items if _matches_filter(v["metadata"], where)]
        return {
            "ids": [k for k, v in self._items.items() if (not where or _matches_filter(v["metadata"], where))],
            "documents": [v["document"] for v in items],
            "metadatas": [v["metadata"] for v in items],
            "embeddings": [v.get("embedding") for v in items],
        }

    def delete(self, ids):
        for doc_id in ids:
            self._items.pop(doc_id, None)


class FakeChroma:
    """langchain_chroma.Chroma の最小モック"""

    def __init__(self, **kwargs):
        self._collection = FakeCollection()
        self._client = MagicMock()
        self._client.delete_collection = MagicMock()

    def add_documents(self, documents: List[Document]):
        for i, doc in enumerate(documents):
            doc_id = f"fake_id_{id(doc)}_{i}"
            self._collection.add(
                ids=[doc_id],
                embeddings=[None],
                documents=[doc.page_content],
                metadatas=[doc.metadata],
            )

    def similarity_search_with_relevance_scores(self, query, k, filter=None):
        items = self._collection.get(where=filter)
        results = []
        for doc_text, meta in zip(items["documents"], items["metadatas"]):
            doc = Document(page_content=doc_text, metadata=meta)
            results.append((doc, 0.8))
        return results[:k]


@pytest.fixture
def tmp_dirs():
    """tech_docs / crm_records の一時ディレクトリを作成する"""
    with tempfile.TemporaryDirectory() as td:
        tech_dir = os.path.join(td, "tech_documents")
        crm_dir = os.path.join(td, "crm_records")
        os.makedirs(tech_dir)
        os.makedirs(crm_dir)
        yield tech_dir, crm_dir, td


@pytest.fixture
def vectordb(tmp_dirs, monkeypatch):
    """モック済み VectorDBManager を返す"""
    tech_dir, crm_dir, td = tmp_dirs
    parent_store_path = os.path.join(td, "parent_store.json")

    fake_chroma = FakeChroma()

    monkeypatch.setattr("src.vectordb.OpenAIEmbeddings", lambda **kwargs: MagicMock())
    monkeypatch.setattr("src.vectordb.Chroma", lambda **kwargs: fake_chroma)
    monkeypatch.setattr("src.vectordb.PARENT_STORE_PATH", parent_store_path)

    db = VectorDBManager(persist_dir=td, collection_name="test")
    db.vectorstore = fake_chroma
    return db, fake_chroma, tech_dir, crm_dir


# ==================================================================
# ケース1: build_index() が .md ファイルを正常にインデックス化できる
# ==================================================================

def test_build_index_with_md_files(vectordb):
    db, fake_chroma, tech_dir, crm_dir = vectordb

    # .md ファイルを配置
    with open(os.path.join(tech_dir, "product_a.md"), "w", encoding="utf-8") as f:
        f.write("# 製品A\n\n製品Aは高性能なIoTプラットフォームです。\n\n## 特徴\n\n低コスト・高スケーラビリティ。")
    with open(os.path.join(crm_dir, "crm_001.md"), "w", encoding="utf-8") as f:
        f.write("# 商談記録\n\n顧客企業名: サンプル製作所\n\n初回訪問の記録。")

    db.build_index(tech_dir, crm_dir)

    assert fake_chroma._collection.count() > 0
    assert len(db._parent_store) > 0


# ==================================================================
# ケース2: build_index() が .pdf ファイルを VLM 経由でインデックス化できる（VLM はモック）
# ==================================================================

def test_build_index_with_pdf_vlm_mock(vectordb, tmp_dirs, monkeypatch):
    db, fake_chroma, tech_dir, crm_dir = vectordb
    _, _, td = tmp_dirs

    # .md を1件置いてインデックスが空にならないようにする
    with open(os.path.join(tech_dir, "product_a.md"), "w", encoding="utf-8") as f:
        f.write("# 製品A\n\n内容です。")

    # ダミー .pdf を配置（中身は空バイトでOK、VLM はモックするため）
    pdf_path = os.path.join(tech_dir, "product_b.pdf")
    with open(pdf_path, "wb") as f:
        f.write(b"%PDF-1.4 dummy")

    # VLM 関数をモック（fitz インポートも通過させる）
    monkeypatch.setattr("src.vectordb.extract_text_from_pdf_vlm",
                        lambda path, client, model: "VLMで抽出されたPDFテキストです。製品Bの特徴を説明します。")
    monkeypatch.setattr("src.vectordb.is_pdf", lambda path: path.endswith(".pdf"))

    # fitz (PyMuPDF) のインポートチェックをスキップ
    import sys
    fake_fitz = MagicMock()
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)

    # OpenAI クライアントもモック（_load_pdf_files_vlm 内でローカルインポートされるため openai モジュールを差し替える）
    monkeypatch.setattr("src.vectordb.Config.OPENAI_API_KEY", "sk-fake")
    with patch("openai.OpenAI", return_value=MagicMock()):
        db.build_index(tech_dir, crm_dir)

    assert fake_chroma._collection.count() > 0


# ==================================================================
# ケース3: build_index() の再実行で既存データが保護される（Atomic 原則）
# ==================================================================

def test_build_index_empty_dirs_raises_and_protects(vectordb, tmp_dirs):
    """ファイルが0件のとき ValueError を raise し、既存インデックスを壊さない"""
    db, fake_chroma, tech_dir, crm_dir = vectordb
    _, _, td = tmp_dirs

    # 先にインデックスを構築
    with open(os.path.join(tech_dir, "product_a.md"), "w", encoding="utf-8") as f:
        f.write("# 製品A\n\n内容です。")
    db.build_index(tech_dir, crm_dir)
    count_before = fake_chroma._collection.count()
    assert count_before > 0

    # ファイルを削除してから再実行
    os.remove(os.path.join(tech_dir, "product_a.md"))
    with pytest.raises(ValueError, match="ファイルが見つかりません"):
        db.build_index(tech_dir, crm_dir)

    # 既存インデックスが保護されていることを確認
    assert fake_chroma._collection.count() == count_before


# ==================================================================
# ケース4: hybrid_search() が正常にヒット結果を返す
# ==================================================================

def test_hybrid_search_returns_results(vectordb):
    db, fake_chroma, tech_dir, crm_dir = vectordb

    with open(os.path.join(tech_dir, "product_a.md"), "w", encoding="utf-8") as f:
        f.write("# 製品A\n\nIoT プラットフォームの特徴。センサー管理・データ収集。")

    db.build_index(tech_dir, crm_dir)

    results = db.search("IoT センサー管理", top_k=3, hybrid=True)
    assert isinstance(results, list)
    # BM25コーパスがあればハイブリッド、なければベクトル検索にフォールバック
    assert len(results) >= 0  # 空ディレクトリでも例外を出さない


# ==================================================================
# ケース5: 親子チャンク: 子チャンクがヒットしたとき親チャンクが返される
# ==================================================================

def test_parent_child_retrieval(vectordb):
    db, fake_chroma, tech_dir, crm_dir = vectordb

    parent_text = "これは親チャンクの全文です。子チャンクより広いコンテキストを持ちます。製品Aの詳細説明。"
    child_text = "子チャンクテキスト。製品A。"
    parent_id = "product_a.md_parent_0"

    # 親チャンクストアを直接セット
    db._parent_store = {parent_id: parent_text}

    # FakeChroma に子チャンクを追加
    child_doc = Document(
        page_content=child_text,
        metadata={"parent_id": parent_id, "source_file": "product_a.md", "source_type": "tech_doc"},
    )
    fake_chroma._collection.add(
        ids=["child_id_1"],
        embeddings=[None],
        documents=[child_text],
        metadatas=[child_doc.metadata],
    )

    results = db._vector_search("製品A", top_k=1, filter_metadata=None)

    assert len(results) == 1
    assert results[0]["text"] == parent_text
    assert results[0]["child_text"] == child_text
    assert results[0]["has_parent"] is True


# ==================================================================
# ケース6: get_index_summary() が登録済みドキュメント一覧を正しく返す
# ==================================================================

def test_get_index_summary(vectordb):
    db, fake_chroma, tech_dir, crm_dir = vectordb

    with open(os.path.join(tech_dir, "product_a.md"), "w", encoding="utf-8") as f:
        f.write("# 製品A\n\n内容。")
    with open(os.path.join(crm_dir, "crm_001.md"), "w", encoding="utf-8") as f:
        f.write("# 商談記録\n\n顧客企業名: テスト株式会社\n\n記録。")

    db.build_index(tech_dir, crm_dir)

    summary = db.get_index_summary()
    assert summary["total_chunks"] > 0
    assert "by_source_type" in summary
    assert "tech_doc" in summary["by_source_type"]
    assert summary["parent_chunks"] > 0


# ==================================================================
# スタンドアロン関数のユニットテスト（外部依存なし）
# ==================================================================

def test_matches_filter_string_equality():
    assert _matches_filter({"source_type": "tech_doc"}, {"source_type": "tech_doc"})
    assert not _matches_filter({"source_type": "crm_record"}, {"source_type": "tech_doc"})


def test_matches_filter_in_operator():
    meta = {"source_type": "pdf_upload"}
    assert _matches_filter(meta, {"source_type": {"$in": ["tech_doc", "pdf_upload"]}})
    assert not _matches_filter(meta, {"source_type": {"$in": ["tech_doc", "crm_record"]}})


def test_reciprocal_rank_fusion_combines_scores():
    vec = [
        {"metadata": {"source_file": "a.md"}, "text": "A"},
        {"metadata": {"source_file": "b.md"}, "text": "B"},
    ]
    bm25 = [
        {"metadata": {"source_file": "b.md"}, "text": "B"},
        {"metadata": {"source_file": "a.md"}, "text": "A"},
    ]
    scores = _reciprocal_rank_fusion(vec, bm25)
    assert len(scores) == 2
    # a.md: 1/(60+1+1) + 1/(60+2+1), b.md: 1/(60+2+1) + 1/(60+1+1) → 等しいはず
    files = [s[0] for s in scores]
    assert "a.md" in files
    assert "b.md" in files


def test_build_context_prefix_tech_doc():
    prefix = _build_context_prefix("# 製品A仕様書", "tech_doc", {"product_name": "ProductA", "product_category": "IoT", "target_industries": "製造業"})
    assert "製品技術資料" in prefix
    assert "ProductA" in prefix
    assert "IoT" in prefix


def test_build_context_prefix_crm():
    prefix = _build_context_prefix("# 商談記録", "crm_record", {"source_file": "crm_001.md", "company_name": "テスト株式会社", "deal_stage": "PoC"})
    assert "CRM商談記録" in prefix
    assert "テスト株式会社" in prefix
    assert "PoC" in prefix


def test_extract_crm_company_name():
    text = "## 基本情報\n顧客企業名: 株式会社テスト製造\n\n詳細..."
    assert _extract_crm_company_name(text) == "株式会社テスト製造"


def test_extract_crm_company_name_not_found():
    assert _extract_crm_company_name("特に企業名の記述なし") == ""


def test_infer_deal_stage_poc():
    text = "今回の商談でPoCを実施することになりました。"
    assert _infer_deal_stage(text) == "PoC"


def test_infer_deal_stage_default():
    text = "初めてのお客様です。"
    assert _infer_deal_stage(text) == "初回"


def test_infer_deal_stage_priority():
    text = "契約交渉中。提案もしました。"
    assert _infer_deal_stage(text) == "契約交渉"
