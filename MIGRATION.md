# MIGRATION.md — 展示会フォローアップAIエージェント コード棚卸し

> 目的: 本リポジトリを他プロジェクトへ移植・流用する際のリファレンス。機能単位の責務、コアロジックの所在、UI依存の範囲、流用可否を一覧化する。

---

## 1. 機能単位リスト

| # | 機能名 | 概要 |
|---|--------|------|
| 1 | CSVインポート＆カラムマッピング | リードCSVを読み込み、RX Japan / Q-PASS / Sansan 等の多様なカラム名を正規化 |
| 2 | ランク正規化・LLM推定 | ★5・数字などを A-E ランクへ変換。変換不能時はLLMで推定 |
| 3 | ベクトルDB構築（RAG） | 技術資料・CRM記録のMarkdownを親子チャンクで ChromaDB + BM25 に格納 |
| 4 | ハイブリッド検索 | ベクトル検索 + BM25 → RRF統合スコアで最適チャンクを取得 |
| 5 | CRM名寄せ | メール完全一致 → 会社名ファジーマッチ（rapidfuzz）でHubSpot形式CSVと紐づけ |
| 6 | Web検索（企業情報） | DuckDuckGo 2クエリで企業プロフィール＋最新動向を取得（APIキー不要） |
| 7 | メール生成（LLM） | ランク別ポリシー + RAG + CRM + Web + 音声コンテキストをもとにLLMでメール生成 |
| 8 | 音声文字起こし | Whisper API で m4a/mp3/wav を文字起こし。mutagen でメタデータ取得 |
| 9 | 音声ニーズ抽出 | 文字起こしテキストからLLMで構造化ニーズ（課題・予算・温度感・意思決定者）を抽出 |
| 10 | 音声リード紐づけ | ファイル名解析 + タイムスタンプ照合 + 順番フォールバックでリードに紐づけ |
| 11 | PDF取り込み | pypdf でテキスト抽出 → ChromaDB に追加（既存チャンク保護付き） |
| 12 | コスト見積もり | トークン数・API料金・所要時間を事前推定（バッチ実行前の確認ダイアログ用） |
| 13 | CLIインターフェース | Click + Rich による CUI。Claude Code 連携用 |
| 14 | Streamlit UI | タブ型 Web UI（カラムマッピング・メール生成・音声管理・KB管理） |

---

## 2. 各機能のコアロジック — ファイル・関数マッピング

### 機能 1: CSVインポート＆カラムマッピング

| 関数/クラス | ファイル | 役割 |
|------------|---------|------|
| `load_leads(csv_path)` | [src/utils.py](src/utils.py) | CSV読み込み・空白除去・NaN→空文字 |
| `load_csv_with_encoding(file_obj)` | [src/utils.py](src/utils.py) | UTF-8 BOM / Shift_JIS / CP932 自動判定 |
| `auto_map_columns(df_columns, field_definitions)` | [src/utils.py](src/utils.py) | 2フェーズ照合（完全一致 → ファジー）でカラム名を正規化 |
| `apply_column_mapping(df, mapping)` | [src/utils.py](src/utils.py) | カラムリネーム・`extra_*` プレフィックス付与・lead_id 自動生成 |
| `Config.REQUIRED_FIELDS` / `OPTIONAL_FIELDS` | [src/config.py](src/config.py) | 正規化後のフィールド定義（候補カラム名リスト） |

### 機能 2: ランク正規化・LLM推定

| 関数/クラス | ファイル | 役割 |
|------------|---------|------|
| `RankEstimator` | [src/rank_estimator.py](src/rank_estimator.py) | クラス本体 |
| `RankEstimator.normalize_rank(raw_rank)` | [src/rank_estimator.py](src/rank_estimator.py) | ★5/数字 → A-E 変換（LLM不使用） |
| `RankEstimator.estimate_from_lead(lead, enable_llm)` | [src/rank_estimator.py](src/rank_estimator.py) | 正規化 → 失敗時LLM推定 → デフォルト"C" |
| `RankEstimator._llm_estimate(lead)` | [src/rank_estimator.py](src/rank_estimator.py) | ChatOpenAI でランク推定 |

### 機能 3: ベクトルDB構築（RAG）

| 関数/クラス | ファイル | 役割 |
|------------|---------|------|
| `VectorDBManager` | [src/vectordb.py](src/vectordb.py) | ChromaDB管理クラス本体 |
| `VectorDBManager.build_index(tech_docs_dir, crm_records_dir)` | [src/vectordb.py](src/vectordb.py) | Markdownを親子チャンクに分割してChromaDB + parent_store.json に格納 |
| `VectorDBManager._load_markdown_files(directory, source_type)` | [src/vectordb.py](src/vectordb.py) | .mdファイル読み込み・メタデータ推定 |
| `VectorDBManager._get_splitters_for_doc_type(source_type)` | [src/vectordb.py](src/vectordb.py) | ドキュメント種別ごとの適応チャンクサイズ返却 |
| `_build_context_prefix(title, source_type, metadata)` | [src/vectordb.py](src/vectordb.py) | 各チャンクに文脈プレフィックスを付与（Contextual Retrieval） |

### 機能 4: ハイブリッド検索

| 関数/クラス | ファイル | 役割 |
|------------|---------|------|
| `VectorDBManager.search(query, top_k, filter_metadata, hybrid)` | [src/vectordb.py](src/vectordb.py) | メインエントリ（ベクトル/ハイブリッドを切り替え） |
| `VectorDBManager._hybrid_search(query, top_k, filter_metadata)` | [src/vectordb.py](src/vectordb.py) | ベクトル + BM25 → RRF統合 |
| `VectorDBManager._vector_search(query, top_k, filter_metadata)` | [src/vectordb.py](src/vectordb.py) | ChromaDB similarity search + 親チャンク取得 |
| `VectorDBManager._bm25_search(query, corpus, top_k)` | [src/vectordb.py](src/vectordb.py) | BM25キーワード検索（日本語トークナイズ） |
| `_reciprocal_rank_fusion(vector_results, bm25_results, k=60)` | [src/vectordb.py](src/vectordb.py) | RRFスコア計算 |
| `VectorDBManager.search_tech_docs(query, top_k, hybrid)` | [src/vectordb.py](src/vectordb.py) | tech_doc / pdf_upload フィルタ済み検索 |
| `VectorDBManager.search_crm(query, top_k, hybrid)` | [src/vectordb.py](src/vectordb.py) | crm_record フィルタ済み検索 |

### 機能 5: CRM名寄せ

| 関数/クラス | ファイル | 役割 |
|------------|---------|------|
| `CRMMatcher` | [src/crm_matcher.py](src/crm_matcher.py) | クラス本体 |
| `CRMMatcher.match(lead, crm_df)` | [src/crm_matcher.py](src/crm_matcher.py) | メール完全一致 → 会社名ファジーマッチの2段階照合 |
| `CRMMatcher._normalize_company_name(name)` | [src/crm_matcher.py](src/crm_matcher.py) | NFKC正規化 + 株式会社等の除去 |
| `CRMMatcher.match_all(leads_df, crm_df, ...)` | [src/crm_matcher.py](src/crm_matcher.py) | バッチ照合（DataFrame 全行） |

### 機能 6: Web検索（企業情報）

| 関数/クラス | ファイル | 役割 |
|------------|---------|------|
| `WebSearcher` | [src/web_searcher.py](src/web_searcher.py) | クラス本体 |
| `WebSearcher.search_company(company_name, max_results_per_query)` | [src/web_searcher.py](src/web_searcher.py) | プロフィール + ニュースの2クエリを並走 |
| `WebSearcher._run_query(ddgs_cls, query, section, ...)` | [src/web_searcher.py](src/web_searcher.py) | DuckDuckGo（ddgs）実行・重複除去 |
| `WebSearcher._build_summary(company_name, ...)` | [src/web_searcher.py](src/web_searcher.py) | LLMへ渡す要約文字列を整形 |
| `WebSearcher._normalize_for_search(company_name)` | [src/web_searcher.py](src/web_searcher.py) | 株式会社等のサフィックスを除去 |

### 機能 7: メール生成（LLM）

| 関数/クラス | ファイル | 役割 |
|------------|---------|------|
| `EmailGenerator` | [src/email_generator.py](src/email_generator.py) | クラス本体 |
| `EmailGenerator.generate(lead, tech_context, crm_context, ...)` | [src/email_generator.py](src/email_generator.py) | システム+ユーザープロンプト組み立て → LLM呼び出し → パース |
| `EmailGenerator._build_system_prompt(sender_company)` | [src/email_generator.py](src/email_generator.py) | 営業担当ペルソナのシステムプロンプト |
| `EmailGenerator._build_human_prompt(lead, policy, ...)` | [src/email_generator.py](src/email_generator.py) | 顧客情報・CRM・技術文書・Web・音声コンテキストを結合 |
| `EmailGenerator._parse_llm_response(response)` | [src/email_generator.py](src/email_generator.py) | 【件名】【本文】【CTA】を分割抽出 |
| `RANK_POLICY` | [src/email_generator.py](src/email_generator.py) | A-E ランク別トーン・指示の定数辞書 |
| `FollowUpAgent.process_lead(lead, ...)` | [src/agent.py](src/agent.py) | 全ステップを統括するオーケストレーター |
| `_build_audio_context(transcript, needs)` | [src/agent.py](src/agent.py) | 音声コンテキスト文字列を構築（最優先情報として挿入） |

### 機能 8: 音声文字起こし

| 関数/クラス | ファイル | 役割 |
|------------|---------|------|
| `AudioProcessor` | [src/audio_processor.py](src/audio_processor.py) | クラス本体 |
| `AudioProcessor.get_audio_metadata(file_bytes, filename)` | [src/audio_processor.py](src/audio_processor.py) | mutagen で再生時間・録音日時を取得 |
| `AudioProcessor.transcribe(file_bytes, filename, language)` | [src/audio_processor.py](src/audio_processor.py) | Whisper API 呼び出し（拡張子→MIMEマップ付き） |
| `AudioProcessor.estimate_cost(duration_sec)` | [src/audio_processor.py](src/audio_processor.py) | Whisper コスト推定 |
| `_extract_recording_time(audio)` | [src/audio_processor.py](src/audio_processor.py) | TDRC/©day 等の複数タグから録音日時を抽出 |

### 機能 9: 音声ニーズ抽出

| 関数/クラス | ファイル | 役割 |
|------------|---------|------|
| `AudioProcessor.extract_needs(transcript)` | [src/audio_processor.py](src/audio_processor.py) | 文字起こしテキストから `{summary, issues, needs, budget, decision_maker, temperature}` を構造化抽出 |
| `_NEEDS_EXTRACTION_SYSTEM` | [src/audio_processor.py](src/audio_processor.py) | ニーズ抽出用システムプロンプト定数 |

### 機能 10: 音声リード紐づけ

| 関数/クラス | ファイル | 役割 |
|------------|---------|------|
| `AudioMatcher` | [src/audio_matcher.py](src/audio_matcher.py) | クラス本体 |
| `AudioMatcher.parse_rep_from_filename(filename)` | [src/audio_matcher.py](src/audio_matcher.py) | `YYYYMMDD_担当者名_連番.ext` から担当者名を抽出 |
| `AudioMatcher.match(audio_meta_list, leads_df, ...)` | [src/audio_matcher.py](src/audio_matcher.py) | タイムスタンプ照合 → 順番フォールバック → 信頼度（green/yellow/red）を付与 |
| `AudioMatcher.match_with_csv(mapping_df, ...)` | [src/audio_matcher.py](src/audio_matcher.py) | 手動マッピングCSVによる照合 |
| `AudioMatcher.detect_timestamp_col(leads_df, llm)` | [src/audio_matcher.py](src/audio_matcher.py) | scan_time列を自動検出（完全一致 → キーワード → LLM） |
| `AudioMatcher.detect_gaps(audio_results, leads_df, ...)` | [src/audio_matcher.py](src/audio_matcher.py) | 録音なしリードを担当者別に検出 |
| `AudioMatcher.get_red_flag_rate(results)` | [src/audio_matcher.py](src/audio_matcher.py) | 赤フラグ率（命名規則違反の割合）を計算 |
| `MatchResult` (dataclass) | [src/audio_matcher.py](src/audio_matcher.py) | 紐づけ結果の構造体 |

### 機能 11: PDF取り込み

| 関数/クラス | ファイル | 役割 |
|------------|---------|------|
| `VectorDBManager.add_pdf(pdf_file, source_name)` | [src/vectordb.py](src/vectordb.py) | pypdf でテキスト抽出 → 親子チャンク生成 → ChromaDB + BM25 + parent_store に追加 |
| `VectorDBManager.remove_document(source_name)` | [src/vectordb.py](src/vectordb.py) | 対象PDFの全チャンクを削除・BM25再構築 |
| `VectorDBManager._get_pdf_chunks()` | [src/vectordb.py](src/vectordb.py) | pdf_upload チャンクをChromaDBから取得（build_index 時の保護用） |

### 機能 12: コスト見積もり

| 定数/ロジック | ファイル | 役割 |
|-------------|---------|------|
| `Config.LLM_PRICE_INPUT_PER_1M` / `LLM_PRICE_OUTPUT_PER_1M` | [src/config.py](src/config.py) | LLMトークン単価（$/1Mトークン） |
| `Config.EST_INPUT_TOKENS_PER_LEAD` / `EST_OUTPUT_TOKENS_PER_LEAD` | [src/config.py](src/config.py) | メール生成1件あたりの推定トークン数 |
| `Config.EST_SECONDS_PER_LEAD_BASE` / `EST_SECONDS_PER_LEAD_WEB` | [src/config.py](src/config.py) | 推定所要秒数 |
| `Config.WHISPER_PRICE_PER_MIN` | [src/config.py](src/config.py) | Whisper 文字起こし単価 |
| `AudioProcessor.estimate_cost(duration_sec)` | [src/audio_processor.py](src/audio_processor.py) | 音声ファイルのWhisperコスト計算 |
| コスト計算ロジック | [app.py](app.py) | `_render_tab_email()` 内でバッチ実行前に見積もり表示（UI専用） |

### 機能 13: CLIインターフェース

| 関数/クラス | ファイル | 役割 |
|------------|---------|------|
| `run_check()` | [src/cli_runner.py](src/cli_runner.py) | 環境・ファイル・KB状態を検証してdictで返す |
| `run_build_kb(tech_docs_dir, crm_records_dir)` | [src/cli_runner.py](src/cli_runner.py) | KB構築を実行してdictで返す |
| `run_load_leads(csv_path, ranks)` | [src/cli_runner.py](src/cli_runner.py) | リードCSVを読み込みdictで返す |
| `run_generate(csv_path, ranks, ..., on_progress)` | [src/cli_runner.py](src/cli_runner.py) | メール一括生成のメインビジネスロジック（UI非依存） |
| `load_cli_config()` / `save_cli_config(config)` | [src/cli_runner.py](src/cli_runner.py) | cli_config.yaml の読み書き |
| `cli.py` (Click commands) | [cli.py](cli.py) | `check` / `build-kb` / `generate` / `config` コマンドのCUI定義 |

### 機能 14: Streamlit UI

| 関数 | ファイル | 役割 |
|------|---------|------|
| `main()` | [app.py](app.py) | エントリポイント。セッション初期化 → サイドバー → 各タブ描画 |
| `_init_session_state()` | [app.py](app.py) | 20以上のsession_stateキーを初期化 |
| `_initialize_components()` | [app.py](app.py) | VectorDBManager / EmailGenerator / FollowUpAgent をsession_stateにキャッシュ |
| `_render_sidebar()` | [app.py](app.py) | ステップインジケーター・CSVアップロード・KB構築・ランクフィルター |
| `_render_column_mapping()` | [app.py](app.py) | カラムマッピング確認UI（2段階: リード + CRM） |
| `_render_tab_leads()` | [app.py](app.py) | リード一覧・品質スコア・単件プレビュー |
| `_do_single_generate(lead_data)` | [app.py](app.py) | 単件メール生成・ステップ表示・デバッグ情報 |
| `_render_tab_email()` | [app.py](app.py) | バッチ生成（コスト見積もり → 確認ダイアログ → 進捗バー → ダウンロード） |
| `_render_tab_audio()` | [app.py](app.py) | 音声アップロード・文字起こし・紐づけ・ニーズ抽出 |
| `_render_tab_knowledge()` | [app.py](app.py) | KB概要・PDFアップロード・ドキュメント削除・検索テスト |

---

## 3. Streamlit/UI依存 vs 純粋ロジックの切り分け

**結論: `src/` 配下の全モジュール（11ファイル）はStreamlitに完全無依存。**  
Streamlit依存コードは `app.py` の1ファイルにのみ存在する。

| レイヤー | ファイル | Streamlit依存 | session_state依存 | 移植容易性 |
|----------|----------|:---:|:---:|:---:|
| Web UI | [app.py](app.py) | ✅ | ✅ | 全書き換え必要 |
| CLI UI | [cli.py](cli.py) | ❌ | ❌ | Click/Rich依存のみ |
| ビジネスロジックファサード | [src/cli_runner.py](src/cli_runner.py) | ❌ | ❌ | そのまま使える |
| オーケストレーター | [src/agent.py](src/agent.py) | ❌ | ❌ | そのまま使える |
| ベクトルDB | [src/vectordb.py](src/vectordb.py) | ❌ | ❌ | そのまま使える |
| メール生成 | [src/email_generator.py](src/email_generator.py) | ❌ | ❌ | そのまま使える |
| Web検索 | [src/web_searcher.py](src/web_searcher.py) | ❌ | ❌ | そのまま使える |
| CRM名寄せ | [src/crm_matcher.py](src/crm_matcher.py) | ❌ | ❌ | そのまま使える |
| 音声処理 | [src/audio_processor.py](src/audio_processor.py) | ❌ | ❌ | そのまま使える |
| 音声紐づけ | [src/audio_matcher.py](src/audio_matcher.py) | ❌ | ❌ | そのまま使える |
| ランク推定 | [src/rank_estimator.py](src/rank_estimator.py) | ❌ | ❌ | そのまま使える |
| ユーティリティ | [src/utils.py](src/utils.py) | ❌ | ❌ | そのまま使える |
| 設定・定数 | [src/config.py](src/config.py) | ❌ | ❌ | そのまま使える |

---

## 4. 流用価値の判定

### 判定基準

- **A: そのまま使える** — モジュールが完全独立。コピーするだけで動く
- **B: 抽出して整理が必要** — 動作するが、依存関係・列構造・定数に軽微な調整が必要
- **C: 捨てる** — フレームワーク固有のUIコード。他環境への移植は全書き換えに相当

| # | 機能名 | 判定 | 移植時のメモ |
|---|--------|:---:|-------------|
| 1 | CSVインポート＆カラムマッピング | **A** | `src/utils.py` の関数群がそのまま使える。`Config.REQUIRED_FIELDS` / `OPTIONAL_FIELDS` の候補カラム名リストだけ新プロジェクト用に調整 |
| 2 | ランク正規化・LLM推定 | **A** | `src/rank_estimator.py` が完全独立。`enable_llm=False` にするとLLMなしで変換のみ動作 |
| 3 | ベクトルDB構築（RAG） | **A** | `src/vectordb.py` はChromaDB設定込みで完結。`persist_dir` パスだけ変更すればOK |
| 4 | ハイブリッド検索 | **A** | `VectorDBManager.search()` のインターフェースが安定。BM25は `rank_bm25` パッケージが必要 |
| 5 | CRM名寄せ | **A** | `src/crm_matcher.py` が完全独立。HubSpot以外のCRM形式でも列名を `Config.CRM_OPTIONAL_FIELDS` で調整するだけ |
| 6 | Web検索（企業情報） | **A** | `src/web_searcher.py` が完全独立。`ddgs` パッケージ（旧 `duckduckgo-search`）が必要。APIキー不要 |
| 7 | メール生成（LLM） | **A** | `src/email_generator.py` が完全独立。`RANK_POLICY` の文言は業種・用途に合わせて変更推奨 |
| 8 | 音声文字起こし | **A** | `src/audio_processor.py` が独立。`mutagen`（オプション）・`openai` パッケージが必要 |
| 9 | 音声ニーズ抽出 | **A** | `AudioProcessor.extract_needs()` が独立。`_NEEDS_EXTRACTION_SYSTEM` プロンプトを業種に合わせて調整推奨 |
| 10 | 音声リード紐づけ | **B** | `src/audio_matcher.py` 自体は独立だが、`leads_df` の列名（`visitor_name`, `rep_name`, `scan_time`等）に依存。新プロジェクトの列名に合わせて `match()` の引数を調整する |
| 11 | PDF取り込み | **A** | `VectorDBManager.add_pdf()` が独立。`pypdf` パッケージが必要 |
| 12 | コスト見積もり | **B** | `src/config.py` のトークン単価定数（`EST_*`, `LLM_PRICE_*`）はモデル変更時に更新が必要。見積もり計算ロジック自体は `app.py` に組み込まれているためUI非依存での流用は要抽出 |
| 13 | CLIインターフェース | **B** | `src/cli_runner.py` はそのまま使える。`cli.py` の Click コマンド定義は新プロジェクトのコマンド体系に合わせて軽微な改造が必要 |
| 14 | Streamlit UI | **C** | `app.py` 全体が Streamlit の session_state・ウィジェット・タブに強依存。FastAPI + React / Next.js 等への移植は実質的な全書き換え。ビジネスロジックは `src/` から再利用し、UIだけを作り直す方針が現実的 |

---

## 5. 依存関係グラフ（簡略）

```
app.py (Streamlit UI)
  └→ src/agent.py (FollowUpAgent)
       ├→ src/vectordb.py
       ├→ src/email_generator.py
       ├→ src/web_searcher.py
       ├→ src/rank_estimator.py
       ├→ src/crm_matcher.py
       └→ src/utils.py

cli.py (Click CLI)
  └→ src/cli_runner.py (ビジネスロジックファサード)
       ├→ src/agent.py
       ├→ src/vectordb.py
       ├→ src/email_generator.py
       └→ src/utils.py

src/* (コアモジュール)
  └→ src/config.py (共通設定)
  └→ langchain-openai, langchain-chroma, openai, pandas, rapidfuzz, ddgs, mutagen, pypdf
```

---

## 6. 移植時のパッケージ最小構成

### 必須

```
langchain
langchain-openai
langchain-chroma
langchain-text-splitters
chromadb
rank_bm25
rapidfuzz
pandas
python-dotenv
ddgs
```

### オプション（機能別）

| パッケージ | 依存機能 |
|-----------|---------|
| `pypdf` | PDF取り込み（機能11） |
| `mutagen` | 音声メタデータ取得（機能8） |
| `openai` | Whisper文字起こし（機能8） |
| `click` + `rich` | CLIインターフェース（機能13） |
| `streamlit` | Web UI（機能14、捨てる場合は不要） |
