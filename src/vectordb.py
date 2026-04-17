"""
ベクトルデータベース操作モジュール

ChromaDBを使ったドキュメントのインデックス構築・検索機能を提供する。
技術資料（tech_documents）とCRM商談記録（crm_records）を格納し、
エージェントからのRAGクエリに対応する。

【主な改善点】
- Contextual Retrieval: 各チャンク先頭にドキュメントタイトルと種別を付加
- メタデータ強化: 製品カテゴリ・対象業種・商談ステージ等を自動付与
- ハイブリッド検索: ベクトル検索（ChromaDB）+ BM25 → RRF統合
- チャンキング最適化: 日本語向けに chunk_size=600, overlap=150 に変更
"""

import os
import re
import glob
from typing import List, Dict, Any, Optional, Tuple

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import Config

# BM25ライブラリをオプション依存として読み込む
# インストールされていない場合はハイブリッド検索をスキップしてベクトル検索のみで動作する
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


# ---------------------------------------------------------------
# ファイル名 → 製品名・カテゴリのマッピング
# ---------------------------------------------------------------
PRODUCT_NAME_MAP: Dict[str, str] = {
    "sorani_iot_platform": "Sorani",
    "digima_digital_twin": "DigiMA",
    "smartvision_smart_glass": "SmartVision",
    "ntx_ocr": "NTX-OCR",
    "factorybrain_production": "FactoryBrain",
    "edgeguard_anomaly": "EdgeGuard",
}

# 製品カテゴリのマッピング
PRODUCT_CATEGORY_MAP: Dict[str, str] = {
    "sorani_iot_platform": "IoT",
    "digima_digital_twin": "デジタルツイン",
    "smartvision_smart_glass": "スマートグラス",
    "ntx_ocr": "OCR",
    "factorybrain_production": "生産管理",
    "edgeguard_anomaly": "異常検知",
}

# 製品ごとの対象業種（テキストから自動推定する代わりの既知データ）
PRODUCT_TARGET_INDUSTRIES_MAP: Dict[str, str] = {
    "sorani_iot_platform": "製造業,食品,化学,自動車部品",
    "digima_digital_twin": "製造業,自動車,電子機器,重工業",
    "smartvision_smart_glass": "製造業,建設,点検,保守",
    "ntx_ocr": "製造業,物流,建設,医療",
    "factorybrain_production": "製造業,食品,電子部品,プラスチック",
    "edgeguard_anomaly": "製造業,プレス,切削,溶接,射出成形",
}

# 商談ステージキーワード → ステージ名のマッピング（CRM記録から自動抽出）
DEAL_STAGE_KEYWORDS: Dict[str, List[str]] = {
    "契約交渉": ["契約", "発注", "購買", "価格交渉", "見積"],
    "PoC": ["PoC", "概念検証", "パイロット", "試験導入", "検証"],
    "提案中": ["提案", "デモ", "説明会", "資料送付", "プレゼン"],
    "初回": ["初回", "初めて", "はじめて", "新規", "来場"],
}


class VectorDBManager:
    """ChromaDBの初期化・ドキュメント登録・ハイブリッド検索を管理するクラス"""

    def __init__(
        self,
        persist_dir: str = Config.CHROMA_DB_DIR,
        collection_name: str = Config.CHROMA_COLLECTION_NAME,
    ):
        """
        Parameters
        ----------
        persist_dir : str
            ChromaDBの永続化ディレクトリパス
        collection_name : str
            コレクション名
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # OpenAI Embeddingモデルの初期化
        self.embeddings = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            openai_api_key=Config.OPENAI_API_KEY,
        )

        # テキスト分割器の初期化（日本語最適化: 600文字, 重複150文字）
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            # " "（半角スペース）を除外し日本語テキストに最適化
            separators=["\n## ", "\n### ", "\n\n", "\n", "。", "、"],
        )

        # ChromaDBクライアントの初期化（永続化モード）
        self.vectorstore: Optional[Chroma] = self._load_vectorstore()

        # BM25検索用のコーパスキャッシュ（インデックス構築時に更新）
        # key=source_file, value={"text": str, "metadata": dict}
        self._bm25_corpus: List[Dict[str, Any]] = []

        # 既存インデックスが存在する場合はBM25コーパスを自動再構築する
        # （Streamlit再起動等でsession_stateがリセットされた際の対応）
        if self.is_index_built():
            self._try_rebuild_bm25_corpus()

        if not BM25_AVAILABLE:
            print("警告: rank_bm25 がインストールされていません。ハイブリッド検索はベクトル検索にフォールバックします。")
            print("      pip install rank_bm25 でインストールしてください。")

    def _load_vectorstore(self) -> Chroma:
        """既存のChromaDBを読み込む（なければ空で初期化）"""
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )

    # ==================================================================
    # インデックス構築
    # ==================================================================

    def build_index(self, tech_docs_dir: str, crm_records_dir: str) -> None:
        """
        技術資料とCRM記録を読み込み、ベクトルインデックスを構築する。
        BM25用コーパスも同時に構築する。

        Parameters
        ----------
        tech_docs_dir : str
            技術資料のMarkdownファイルが格納されたディレクトリ
        crm_records_dir : str
            CRM商談記録のMarkdownファイルが格納されたディレクトリ
        """
        print("インデックス構築を開始します...")

        # アップロード済みPDFチャンクを退避（build_index後に復元）
        pdf_saved: Dict[str, Any] = {
            "ids": [], "embeddings": [], "documents": [], "metadatas": []
        }
        if self.is_index_built():
            pdf_saved = self._get_pdf_chunks()
            pdf_count = len(pdf_saved.get("ids", []))
            if pdf_count > 0:
                print(f"  アップロード済みPDFチャンク: {pdf_count}件を退避")
            print("既存インデックスをクリアします...")
            self.clear_index()
            self.vectorstore = self._load_vectorstore()

        # 技術資料の読み込み（強化メタデータ付き）
        tech_docs = self._load_markdown_files(tech_docs_dir, source_type="tech_doc")
        print(f"  技術資料: {len(tech_docs)} ファイル読み込み完了")

        # CRM記録の読み込み（強化メタデータ付き）
        crm_docs = self._load_markdown_files(crm_records_dir, source_type="crm_record")
        print(f"  CRM記録: {len(crm_docs)} ファイル読み込み完了")

        all_raw_docs = tech_docs + crm_docs

        # BM25コーパスを構築（ファイル単位で保持）
        self._bm25_corpus = [
            {"text": doc["text"], "metadata": doc["metadata"]}
            for doc in all_raw_docs
        ]

        # チャンク分割（Contextual Retrievalのためにタイトルをチャンク先頭に付加）
        all_chunks: List[Document] = []
        for raw_doc in all_raw_docs:
            # ドキュメント全体から先頭行（タイトル）を取得
            first_line = raw_doc["text"].split("\n")[0].strip()

            # チャンク分割
            chunks = self.text_splitter.create_documents(
                texts=[raw_doc["text"]],
                metadatas=[raw_doc["metadata"]],
            )

            # 各チャンク先頭にコンテキスト情報を付加（Contextual Retrieval）
            for chunk in chunks:
                source_type = chunk.metadata.get("source_type", "")
                context_prefix = _build_context_prefix(first_line, source_type, chunk.metadata)
                chunk.page_content = context_prefix + chunk.page_content

            all_chunks.extend(chunks)

        print(f"  Markdownチャンク数: {len(all_chunks)}")

        # ChromaDBに格納
        self.vectorstore.add_documents(all_chunks)

        # 退避したPDFチャンクを再投入（再Embedding不要 / コスト0）
        if pdf_saved["ids"]:
            self.vectorstore._collection.add(
                ids=pdf_saved["ids"],
                embeddings=pdf_saved["embeddings"],
                documents=pdf_saved["documents"],
                metadatas=pdf_saved["metadatas"],
            )
            # BM25コーパスにもPDFチャンクを追加
            for text, meta in zip(pdf_saved["documents"], pdf_saved["metadatas"]):
                self._bm25_corpus.append({"text": text, "metadata": meta})
            print(f"  アップロード済みPDFチャンク: {len(pdf_saved['ids'])}件を復元")

        total = self.vectorstore._collection.count()
        print(f"インデックス構築が完了しました（総チャンク数: {total}）")

    # ==================================================================
    # Markdownファイル読み込み（メタデータ強化）
    # ==================================================================

    def _load_markdown_files(self, directory: str, source_type: str) -> List[Dict]:
        """
        指定ディレクトリ内のMarkdownファイルを読み込み、強化メタデータを付与する。

        Parameters
        ----------
        directory : str
            読み込み対象ディレクトリ
        source_type : str
            ドキュメント種別タグ（'tech_doc' or 'crm_record'）

        Returns
        -------
        List[Dict]
            ドキュメントのリスト（text, metadata）
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"ディレクトリが見つかりません: {directory}")

        md_files = sorted(glob.glob(os.path.join(directory, "*.md")))
        if not md_files:
            print(f"  警告: {directory} にMarkdownファイルが見つかりません")
            return []

        docs = []
        for filepath in md_files:
            filename = os.path.splitext(os.path.basename(filepath))[0]

            with open(filepath, encoding="utf-8") as f:
                text = f.read()

            # 基本メタデータ
            metadata: Dict[str, Any] = {
                "source_type": source_type,
                "source_file": os.path.basename(filepath),
            }

            if source_type == "tech_doc":
                # ── 技術資料メタデータの強化 ──────────────────────────
                product_name = PRODUCT_NAME_MAP.get(filename, filename)
                metadata["product_name"] = product_name

                # 製品カテゴリ（IoT/デジタルツイン/スマートグラス/OCR/生産管理/異常検知）
                metadata["product_category"] = PRODUCT_CATEGORY_MAP.get(filename, "その他")

                # 対象業種（既知マッピングを使用）
                metadata["target_industries"] = PRODUCT_TARGET_INDUSTRIES_MAP.get(filename, "製造業")

                # 主要キーワード（ファイル名と見出しから抽出）
                keywords = _extract_keywords_from_text(text, product_name)
                metadata["keywords"] = ",".join(keywords)

            elif source_type == "crm_record":
                # ── CRM記録メタデータの強化 ──────────────────────────
                # 顧客企業名（「## 基本情報」セクションの「顧客企業名:」行から正規表現で抽出）
                company_name = _extract_crm_company_name(text)
                if company_name:
                    metadata["company_name"] = company_name

                # 商談で議論された製品名（テキスト中の製品名キーワードを抽出）
                products_discussed = _extract_products_discussed(text)
                if products_discussed:
                    metadata["products_discussed"] = ",".join(products_discussed)

                # 商談ステージ（初回/提案中/PoC/契約交渉）
                deal_stage = _infer_deal_stage(text)
                metadata["deal_stage"] = deal_stage

            docs.append({"text": text, "metadata": metadata})

        return docs

    # ==================================================================
    # 検索メソッド
    # ==================================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None,
        hybrid: bool = True,
    ) -> List[Dict]:
        """
        クエリに類似するドキュメントを検索する。

        hybrid=True の場合、ベクトル検索（ChromaDB）と BM25 キーワード検索を
        Reciprocal Rank Fusion（RRF）で統合したハイブリッド検索を行う。
        BM25 が利用できない場合はベクトル検索にフォールバックする。

        Parameters
        ----------
        query : str
            検索クエリ文字列
        top_k : int
            取得する上位件数
        filter_metadata : Dict, optional
            メタデータによるフィルタリング条件（例: {"source_type": "tech_doc"}）
        hybrid : bool
            True の場合ハイブリッド検索、False の場合ベクトル検索のみ（デフォルト: True）

        Returns
        -------
        List[Dict]
            検索結果のリスト（text, metadata, score）
        """
        if not self.is_index_built():
            raise RuntimeError("インデックスが構築されていません。build_index() を先に実行してください。")

        # ハイブリッド検索: BM25が使用可能かつhybridモードが有効な場合
        if hybrid and BM25_AVAILABLE and self._bm25_corpus:
            return self._hybrid_search(query, top_k, filter_metadata)

        # フォールバック: ベクトル検索のみ
        return self._vector_search(query, top_k, filter_metadata)

    def _vector_search(
        self,
        query: str,
        top_k: int,
        filter_metadata: Optional[Dict],
    ) -> List[Dict]:
        """
        ベクトル検索のみを実行する（ChromaDB）。

        Parameters
        ----------
        query : str
            検索クエリ
        top_k : int
            取得件数
        filter_metadata : Dict, optional
            メタデータフィルタ

        Returns
        -------
        List[Dict]
            検索結果（text, metadata, score）
        """
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=top_k,
            filter=filter_metadata,
        )
        return [
            {"text": doc.page_content, "metadata": doc.metadata, "score": score}
            for doc, score in results
        ]

    def _bm25_search(
        self,
        query: str,
        corpus: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict]:
        """
        BM25 キーワード検索を実行する。

        日本語対応として、文字ユニグラム（1文字単位）と単語区切りを組み合わせた
        簡易トークナイザを使用する。

        Parameters
        ----------
        query : str
            検索クエリ
        corpus : List[Dict]
            検索対象コーパス（{"text": str, "metadata": dict} のリスト）
        top_k : int
            取得件数

        Returns
        -------
        List[Dict]
            BM25スコア順の検索結果（text, metadata, score）
        """
        if not corpus:
            return []

        # 日本語簡易トークナイザ: スペース区切り + 2文字N-gramで分割
        def tokenize(text: str) -> List[str]:
            # スペース・改行で分割した上で、2文字以上のトークンにn-gramを追加
            tokens = text.split()
            ngrams = [text[i:i+2] for i in range(len(text) - 1)]
            return tokens + ngrams

        # コーパスをトークナイズ
        tokenized_corpus = [tokenize(doc["text"]) for doc in corpus]

        # BM25モデル構築
        bm25 = BM25Okapi(tokenized_corpus)

        # クエリをトークナイズしてスコア計算
        tokenized_query = tokenize(query)
        scores = bm25.get_scores(tokenized_query)

        # スコア上位 top_k 件を返す
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [
            {
                "text": corpus[i]["text"],
                "metadata": corpus[i]["metadata"],
                "score": float(scores[i]),
            }
            for i in top_indices
            if scores[i] > 0  # スコアが0以下の結果は除外
        ]

    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        filter_metadata: Optional[Dict],
    ) -> List[Dict]:
        """
        ハイブリッド検索（ベクトル検索 + BM25）を実行し、RRFで統合する。

        ベクトル検索は ChromaDB、キーワード検索は BM25Okapi を使用。
        両者のランキングを Reciprocal Rank Fusion（RRF）でスコア統合し、
        上位 top_k 件を返す。

        Parameters
        ----------
        query : str
            検索クエリ
        top_k : int
            最終的に返す件数
        filter_metadata : Dict, optional
            メタデータフィルタ（ベクトル検索に適用）

        Returns
        -------
        List[Dict]
            RRFスコア順の検索結果（text, metadata, score）
        """
        # メタデータフィルタに合致するコーパスだけをBM25対象にする
        # $in 演算子（{"source_type": {"$in": [...]}}) に対応
        if filter_metadata:
            filtered_corpus = [
                doc for doc in self._bm25_corpus
                if _matches_filter(doc["metadata"], filter_metadata)
            ]
        else:
            filtered_corpus = self._bm25_corpus

        # ベクトル検索（多めに取得してRRFに使う）
        fetch_k = max(top_k * 3, 10)
        vector_results = self._vector_search(query, top_k=fetch_k, filter_metadata=filter_metadata)

        # BM25検索
        bm25_results = self._bm25_search(query, corpus=filtered_corpus, top_k=fetch_k)

        # RRF でスコア統合
        rrf_scores = _reciprocal_rank_fusion(vector_results, bm25_results)

        # RRFスコア上位のドキュメントを取得
        # source_file をキーに、元の検索結果からテキストとメタデータを引き出す
        all_results_map: Dict[str, Dict] = {}
        for r in vector_results + bm25_results:
            key = r["metadata"].get("source_file", "")
            if key and key not in all_results_map:
                all_results_map[key] = r

        # RRFスコア順に並べて返す
        output = []
        for source_file, rrf_score in rrf_scores[:top_k]:
            if source_file in all_results_map:
                doc = all_results_map[source_file].copy()
                doc["score"] = rrf_score  # RRFスコアに置き換え
                output.append(doc)

        return output

    def search_tech_docs(self, query: str, top_k: int = 3, hybrid: bool = True) -> List[Dict]:
        """
        技術資料を対象に検索する（Markdown由来の tech_doc とPDF由来の pdf_upload を対象）。

        Parameters
        ----------
        query : str
            検索クエリ
        top_k : int
            取得する上位件数
        hybrid : bool
            ハイブリッド検索を使用するか（デフォルト: True）

        Returns
        -------
        List[Dict]
            検索結果のリスト
        """
        return self.search(
            query=query,
            top_k=top_k,
            filter_metadata={"source_type": {"$in": ["tech_doc", "pdf_upload"]}},
            hybrid=hybrid,
        )

    def search_crm(self, query: str, top_k: int = 3, hybrid: bool = True) -> List[Dict]:
        """
        CRM商談記録のみを対象に検索する（source_type="crm_record" フィルタ）。

        Parameters
        ----------
        query : str
            検索クエリ
        top_k : int
            取得する上位件数
        hybrid : bool
            ハイブリッド検索を使用するか（デフォルト: True）

        Returns
        -------
        List[Dict]
            検索結果のリスト
        """
        return self.search(
            query=query,
            top_k=top_k,
            filter_metadata={"source_type": "crm_record"},
            hybrid=hybrid,
        )

    # ==================================================================
    # インデックス管理
    # ==================================================================

    def is_index_built(self) -> bool:
        """
        インデックスが既に構築済みかどうかを確認する。

        Returns
        -------
        bool
            ドキュメントが1件以上格納されていれば True
        """
        try:
            count = self.vectorstore._collection.count()
            return count > 0
        except Exception:
            return False

    def _get_pdf_chunks(self) -> Dict[str, Any]:
        """
        ChromaDB から source_type="pdf_upload" のチャンクを取得する。

        build_index() 前に退避し、再構築後に復元するために使用。
        embeddings を含めて取得することで、再Embedding（API費用）なしに復元できる。

        Returns
        -------
        Dict[str, Any]
            ChromaDB collection.get() の結果（ids / embeddings / documents / metadatas）
        """
        try:
            results = self.vectorstore._collection.get(
                where={"source_type": "pdf_upload"},
                include=["embeddings", "documents", "metadatas"],
            )
            return results
        except Exception as e:
            print(f"  警告: PDFチャンクの退避に失敗しました: {e}")
            return {"ids": [], "embeddings": [], "documents": [], "metadatas": []}

    def clear_index(self) -> None:
        """インデックスをリセットする（コレクションを削除して再作成）"""
        try:
            self.vectorstore._client.delete_collection(self.collection_name)
            print(f"コレクション '{self.collection_name}' を削除しました。")
        except Exception as e:
            print(f"コレクション削除時の警告: {e}")

        # BM25コーパスもクリア
        self._bm25_corpus = []

    def add_pdf(self, pdf_file, source_name: str = None) -> int:
        """
        PDFファイルを読み込み、テキスト抽出→チャンク分割→ベクトルDBに追加する。

        既存のContextual Retrieval機構を活用し、技術資料（tech_doc）として追加する。
        追加後は BM25 コーパスにも反映されるため、ハイブリッド検索でも参照可能。

        Parameters
        ----------
        pdf_file : file-like object
            読み込むPDFファイル（Streamlitの UploadedFile など）
        source_name : str, optional
            ソースファイル名。未指定の場合は pdf_file.name または "unknown.pdf" を使用。

        Returns
        -------
        int
            追加したチャンク数。テキストが空の場合は 0。
        """
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise ImportError(
                "PDF取り込みには pypdf が必要です。`pip install pypdf` を実行してください。"
            ) from e

        import io

        # ソース名の決定
        if source_name is None:
            source_name = getattr(pdf_file, "name", "unknown.pdf")
        filename = os.path.basename(source_name)

        # PDFテキスト抽出
        raw_bytes = getattr(pdf_file, "getvalue", None)
        if raw_bytes:
            pdf_bytes = raw_bytes()
        else:
            pdf_bytes = pdf_file.read()

        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = "\n".join(
            (page.extract_text() or "").strip()
            for page in reader.pages
        ).strip()

        if not text:
            print(f"  警告: '{filename}' からテキストを抽出できませんでした（スキャンPDF等）")
            return 0

        # メタデータ構築（pdf_upload として管理することで build_index() 時に保持される）
        base_name = os.path.splitext(filename)[0]
        metadata: Dict[str, Any] = {
            "source_type": "pdf_upload",
            "source_file": filename,
            "product_name": base_name,
            "product_category": "その他",
            "target_industries": "製造業",
            "keywords": base_name,
        }

        # BM25コーパスに追加
        self._bm25_corpus.append({"text": text, "metadata": metadata})

        # チャンク分割 + Contextual Retrieval プレフィックス付加
        first_line = text.split("\n")[0].strip()[:100]
        chunks = self.text_splitter.create_documents(
            texts=[text],
            metadatas=[metadata],
        )
        for chunk in chunks:
            context_prefix = _build_context_prefix(first_line, "pdf_upload", metadata)
            chunk.page_content = context_prefix + chunk.page_content

        # ChromaDBに追加
        self.vectorstore.add_documents(chunks)
        print(f"  PDF追加完了: '{filename}' → {len(chunks)} チャンク")
        return len(chunks)

    def _try_rebuild_bm25_corpus(self) -> None:
        """
        BM25コーパスをデフォルトのデータディレクトリから再構築する。

        Streamlit 再起動時等に既存 ChromaDB インデックスを読み込んだ場合、
        `_bm25_corpus` が空になるため、本メソッドで自動再構築する。
        データディレクトリが存在しない場合は静かにスキップする。
        """
        tech_dir = Config.TECH_DOCS_DIR
        crm_dir = Config.CRM_RECORDS_DIR

        if not (os.path.exists(tech_dir) and os.path.exists(crm_dir)):
            return  # ディレクトリが存在しない場合はスキップ

        try:
            tech_docs = self._load_markdown_files(tech_dir, source_type="tech_doc")
            crm_docs = self._load_markdown_files(crm_dir, source_type="crm_record")
            self._bm25_corpus = [
                {"text": doc["text"], "metadata": doc["metadata"]}
                for doc in tech_docs + crm_docs
            ]
        except Exception:
            # コーパス再構築に失敗してもアプリを止めない（ベクトル検索にフォールバック）
            self._bm25_corpus = []


# ==================================================================
# スタンドアロン関数群
# ==================================================================

def _matches_filter(metadata: Dict, filter_metadata: Dict) -> bool:
    """
    メタデータがフィルタ条件に合致するか判定する。

    ChromaDB互換の演算子をサポート:
    - 文字列値: 等価比較（例: {"source_type": "crm_record"}）
    - {"$in": [...]}: リスト内包比較（例: {"source_type": {"$in": ["tech_doc", "pdf_upload"]}}）

    Parameters
    ----------
    metadata : Dict
        チェック対象のメタデータ
    filter_metadata : Dict
        フィルタ条件

    Returns
    -------
    bool
        全条件に合致すれば True
    """
    for key, condition in filter_metadata.items():
        val = metadata.get(key)
        if isinstance(condition, dict):
            if "$in" in condition and val not in condition["$in"]:
                return False
        else:
            if val != condition:
                return False
    return True


def _reciprocal_rank_fusion(
    vector_results: List[Dict],
    bm25_results: List[Dict],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion（RRF）でベクトル検索とBM25の結果を統合する。

    RRFスコア = Σ 1 / (k + rank + 1) をソースファイル単位で集計する。
    kはランキング安定化パラメータ（デフォルト60は文献上の推奨値）。

    Parameters
    ----------
    vector_results : List[Dict]
        ベクトル検索結果（ランク順）
    bm25_results : List[Dict]
        BM25検索結果（ランク順）
    k : int
        RRFの安定化パラメータ（デフォルト: 60）

    Returns
    -------
    List[Tuple[str, float]]
        (source_file, rrf_score) のリスト（スコア降順）
    """
    scores: Dict[str, float] = {}

    # ベクトル検索のランクを加算
    for rank, doc in enumerate(vector_results):
        doc_id = doc["metadata"].get("source_file", f"vec_{rank}")
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    # BM25検索のランクを加算
    for rank, doc in enumerate(bm25_results):
        doc_id = doc["metadata"].get("source_file", f"bm25_{rank}")
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    # スコア降順でソートして返す
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _build_context_prefix(title: str, source_type: str, metadata: Dict) -> str:
    """
    Contextual Retrieval 用のコンテキストプレフィックスを生成する。

    各チャンクの先頭に付加することで、チャンク単体でも文書の文脈が
    わかるようにし、検索精度を向上させる。

    Parameters
    ----------
    title : str
        ドキュメントの先頭行（タイトル）
    source_type : str
        'tech_doc' または 'crm_record'
    metadata : Dict
        ドキュメントのメタデータ

    Returns
    -------
    str
        チャンク先頭に付加するプレフィックス文字列
    """
    if source_type in ("tech_doc", "pdf_upload"):
        product = metadata.get("product_name", "不明")
        category = metadata.get("product_category", "")
        industries = metadata.get("target_industries", "")
        tag = "PDF技術資料" if source_type == "pdf_upload" else "製品技術資料"
        prefix = f"[{tag}: {product}（{category}）] {title}"
        if industries:
            prefix += f"\n対象業種: {industries}"
        return prefix + "\n\n"
    elif source_type == "crm_record":
        source_file = metadata.get("source_file", "")
        crm_id = os.path.splitext(source_file)[0].upper()
        company = metadata.get("company_name", "")
        stage = metadata.get("deal_stage", "")
        prefix = f"[CRM商談記録: {crm_id}] {title}"
        if company:
            prefix += f" / 顧客: {company}"
        if stage:
            prefix += f" / ステージ: {stage}"
        return prefix + "\n\n"
    return ""


def _extract_keywords_from_text(text: str, product_name: str) -> List[str]:
    """
    テキストの見出し（##行）から主要キーワードを抽出する。

    Parameters
    ----------
    text : str
        Markdownテキスト
    product_name : str
        製品名（必ずキーワードに含める）

    Returns
    -------
    List[str]
        キーワードリスト（重複なし、最大10件）
    """
    keywords = [product_name]

    # ## 見出しからキーワードを収集
    headings = re.findall(r"^#{1,3}\s+(.+)$", text, re.MULTILINE)
    for h in headings:
        # 「#」や記号を除去してクリーニング
        cleaned = re.sub(r"[「」【】『』《》〔〕]", "", h).strip()
        if cleaned and cleaned not in keywords:
            keywords.append(cleaned)

    return keywords[:10]


def _extract_crm_company_name(text: str) -> str:
    """
    CRM記録テキストから顧客企業名を正規表現で抽出する。

    「顧客企業名:」「企業名:」「会社名:」などのパターンに対応。

    Parameters
    ----------
    text : str
        CRM商談記録テキスト

    Returns
    -------
    str
        顧客企業名（見つからない場合は空文字）
    """
    # パターン: 「顧客企業名: ○○株式会社」などの行を検索
    patterns = [
        r"顧客企業名[:\s：|]+(.+)",
        r"顧客企業\s*\|+\s*(.+)",
        r"企業名[:\s：|]+(.+)",
        r"会社名[:\s：|]+(.+)",
        r"顧客名[:\s：|]+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            name = match.group(1).strip()
            # Markdownの表（|）、余分な記号・空白を除去
            name = re.sub(r"\|.*$", "", name)       # 末尾の | 以降を削除（表形式）
            name = re.sub(r"[|\s]+$", "", name)     # 末尾の | と空白を除去
            name = re.sub(r"\s+", "", name)
            return name.strip()

    return ""


def _extract_products_discussed(text: str) -> List[str]:
    """
    CRM記録テキストから商談で議論された製品名を抽出する。

    Parameters
    ----------
    text : str
        CRM商談記録テキスト

    Returns
    -------
    List[str]
        製品名リスト（重複なし）
    """
    products = []
    product_keywords = list(PRODUCT_NAME_MAP.values())  # 既知の製品名リスト

    for product in product_keywords:
        if product in text:
            products.append(product)

    return products


def _infer_deal_stage(text: str) -> str:
    """
    CRM記録テキストから商談ステージを推定する。

    DEAL_STAGE_KEYWORDS に定義されたキーワードで本文を走査し、
    最初にマッチしたステージを返す。どれもマッチしない場合は「初回」。

    Parameters
    ----------
    text : str
        CRM商談記録テキスト

    Returns
    -------
    str
        商談ステージ（初回/提案中/PoC/契約交渉）
    """
    # 優先度順（高い順）でチェック
    priority_order = ["契約交渉", "PoC", "提案中", "初回"]

    for stage in priority_order:
        keywords = DEAL_STAGE_KEYWORDS.get(stage, [])
        for kw in keywords:
            if kw in text:
                return stage

    return "初回"


# -------------------------
# 動作確認用スクリプト
# -------------------------
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("VectorDBManager 動作確認（ハイブリッド検索テスト付き）")
    print("=" * 60)

    # 設定の検証
    try:
        Config.validate()
    except EnvironmentError as e:
        print(f"設定エラー: {e}")
        sys.exit(1)

    print(f"BM25利用可否: {'✅ 利用可能' if BM25_AVAILABLE else '❌ 未インストール（pip install rank_bm25）'}")

    # VectorDBManager の初期化
    db = VectorDBManager()

    # インデックス構築（チャンクサイズ変更のため再構築が必要）
    print("\n[1] インデックス構築（既存インデックスをクリアして再構築）")
    db.build_index(
        tech_docs_dir=Config.TECH_DOCS_DIR,
        crm_records_dir=Config.CRM_RECORDS_DIR,
    )

    # ---- テスト検索 1: ベクトル検索 ----
    print("\n[2] テスト検索1: 「プレス機の異常検知」でベクトル検索（hybrid=False）")
    print("-" * 50)
    results_vec = db.search_tech_docs("プレス機の異常検知", top_k=3, hybrid=False)
    for i, r in enumerate(results_vec, 1):
        print(f"  [{i}] スコア: {r['score']:.4f}")
        print(f"       製品: {r['metadata'].get('product_name', '-')}")
        print(f"       カテゴリ: {r['metadata'].get('product_category', '-')}")
        print(f"       テキスト冒頭: {r['text'][:80]}...")
        print()

    # ---- テスト検索 2: ハイブリッド検索（比較） ----
    print("[3] テスト検索2: 「プレス機の異常検知」でハイブリッド検索（hybrid=True）")
    print("-" * 50)
    results_hybrid = db.search_tech_docs("プレス機の異常検知", top_k=3, hybrid=True)
    for i, r in enumerate(results_hybrid, 1):
        print(f"  [{i}] RRFスコア: {r['score']:.6f}")
        print(f"       製品: {r['metadata'].get('product_name', '-')}")
        print(f"       カテゴリ: {r['metadata'].get('product_category', '-')}")
        print(f"       テキスト冒頭: {r['text'][:80]}...")
        print()

    # ---- テスト検索 3: CRM検索（メタデータ強化確認）----
    print("[4] テスト検索3: 「中部鉄鋼工業」でCRM検索")
    print("-" * 50)
    results_crm = db.search_crm("中部鉄鋼工業", top_k=3)
    for i, r in enumerate(results_crm, 1):
        print(f"  [{i}] スコア: {r['score']:.6f}")
        print(f"       ファイル: {r['metadata'].get('source_file', '-')}")
        print(f"       企業名: {r['metadata'].get('company_name', '-')}")
        print(f"       議論製品: {r['metadata'].get('products_discussed', '-')}")
        print(f"       商談ステージ: {r['metadata'].get('deal_stage', '-')}")
        print(f"       テキスト冒頭: {r['text'][:80]}...")
        print()

    # ---- テスト検索 4: 全体ハイブリッド検索 ----
    print("[5] テスト検索4: 「帳票のデジタル化」で全体ハイブリッド検索")
    print("-" * 50)
    results_all = db.search("帳票のデジタル化", top_k=5)
    for i, r in enumerate(results_all, 1):
        print(f"  [{i}] RRFスコア: {r['score']:.6f}")
        print(f"       種別: {r['metadata'].get('source_type', '-')}")
        print(f"       ファイル: {r['metadata'].get('source_file', '-')}")
        print(f"       テキスト冒頭: {r['text'][:80]}...")
        print()

    print("=" * 60)
    print("動作確認完了")
    print("=" * 60)
