# プロジェクト構成ガイド

**バージョン**: 2026年5月  
**対象読者**: このリポジトリを初めて読む人・将来の自分

---

## 全体ツリー（要約）

```
exhibition-followup-agent/
├── .claude/commands/       # Claude Code Skills（対話フロー定義）
├── src/                    # Pythonビジネスロジック（コアモジュール群）
├── tests/                  # 自動テスト（pytest）
├── data/                   # 入力データ（リードCSV・製品資料・CRMデモ）
├── docs/                   # 設計ドキュメント
├── output/                 # 生成済みメールCSV（実行結果）
├── credentials/            # Gmail OAuth認証情報（コミット対象外）
├── chroma_db/              # ChromaDB永続化データ（実行時自動生成）
├── .venv/                  # Python仮想環境（Gitでは管理しない）
├── cli_config.yaml         # 実行設定（会社名・ランク・パス等）
├── requirements.txt        # 依存パッケージ一覧
├── CLAUDE.md               # Claude Codeへの指示書（AIが読む設定ファイル）
├── README.md               # プロジェクト概要（外部公開向け）
└── system_overview.md      # システム全体アーキテクチャ解説
```

---

## .claude/commands/ ── Claude Code Skills

Claude Codeのスラッシュコマンド（`/xxx`）の対話フローをMarkdownで定義するディレクトリ。  
ここにファイルを置くと Claude Code が自動的にコマンドとして認識する。  
UIコードを一切書かずに対話型ワークフローを実現するのがこの設計の肝。

| ファイル | コマンド | 役割 |
|---|---|---|
| `setup.md` | `/setup` | 初回セットアップガイド（9ステップ。APIキー設定→KB構築まで）|
| `email-workflow.md` | `/email-workflow` | メール生成の全体フロー（Step 0〜Gmail下書きまで） |
| `csv-mapping.md` | `/csv-mapping` | CSVカラムのマッピング確認・名前表記ゆれの対話的修正 |
| `audio-matching.md` | `/audio-matching` | 音声ファイルのリードへの紐づけと文字起こし |
| `inspect-data.md` | `/inspect-data` | リードCSVのデータ品質チェック（読み取り専用） |
| `match-records.md` | `/match-records` | リード ↔ CRMレコードの照合確認 |
| `kb-status.md` | `/kb-status` | ナレッジベース状態確認（登録ドキュメント・チャンク数・最終更新日時） |
| `rank-mapping.md` | `/rank-mapping` | ランク値の正規化マッピング確認・`cli_config.yaml` への保存（カラムの「値」の揺れに対応） |

---

## src/ ── Pythonビジネスロジック

Skillsから呼び出されるバックエンドロジック。Skills側はUIとフロー制御を担い、このディレクトリのモジュールが実際の処理を担う。

### エントリーポイント

| ファイル | 役割 |
|---|---|
| `cli_runner.py` | **Skillsと各モジュールをつなぐ唯一の橋渡し層**。`run_check()` / `run_build_kb()` / `run_generate()` / `run_draft_to_gmail()` の4関数を提供。すべての戻り値は `dict` に統一されている |
| `agent.py` | **オーケストレーター**（`FollowUpAgent`）。1リード分の処理フロー全体（ランク推定→RAG→CRM照合→Web検索→メール生成）を調整する。LangChainのAgentは使わず、シンプルな関数呼び出しで実装 |
| `config.py` | **設定・定数の一元管理**。モデル名・料金・CSVカラム候補名（`REQUIRED_FIELDS` / `OPTIONAL_FIELDS`。後者には `rep_name`〈スキャン担当者：音声紐づけ用〉と `follow_person`〈フォロー担当：将来拡張用〉を分離して保持）・ランク定義等。コードに直書きせずここに集約することで変更を一箇所で済ませる |

### RAG（製品知識の検索）

| ファイル | 役割 |
|---|---|
| `vectordb.py` | **ChromaDB + BM25 ハイブリッド検索 + 親子チャンク**の実装。小さな子チャンクで高精度に検索し、ヒットした周辺の親チャンクをLLMに渡す設計。`build_index()` でMarkdown/PDFを取り込み、`hybrid_search()` でリードの関心に合う資料を返す |
| `pdf_processor.py` | **VLM による PDF テキスト化**。PyMuPDF でページを PNG 画像化し、gpt-5.4-nano vision API で Markdown テキストとして抽出する。`build_index()` から呼ばれ、`data/tech_documents/` の `.pdf` ファイルを自動処理 |

### メール生成・品質保証（3層アーキテクチャ）

| ファイル | 役割 | 検証レイヤー |
|---|---|---|
| `email_generator.py` | リードの情報（ランク・関心製品・RAG文脈・CRM・Web・音声）を統合してGPTでメールを生成。ランク（A〜E）ごとにプロンプトテンプレートを切り替える | — |
| `email_validator.py` | **Layer 1**: LLM不使用・APIコストゼロの機械チェック。「プレースホルダ漏れ」「example.com混入」「関心外製品への言及」等4ルールを正規表現で検査 | Layer 1 |
| `email_judge.py` | **Layer 2**: LLM-as-a-Judge。Layer 1通過後のメールをLLMがトーン・内容・整合性を審査。`cli_config.yaml` の `enable_llm_judge` で有効/無効を切り替え可能（コスト管理） | Layer 2 |
| `date_validator.py` | 商談候補日の妥当性チェック（過去日・土日・遠すぎる日付を排除）。メール本文に「来週ご都合はいかがでしょうか」等の日程提案が含まれる場合に使用 | Layer 1の補助 |

### リード情報の補完

| ファイル | 役割 |
|---|---|
| `rank_estimator.py` | 商談ランクの正規化と推定。「★5→A」等の変換はルールベース、変換できない場合はLLMが担当者コメント等からA〜Eを推定。`infer_rank_mapping_with_llm()` でランクフィールドのユニーク値リストを A〜E にマッピング推定 |
| `crm_matcher.py` | 既存CRM CSVとリードをメール完全一致 + 社名ファジーマッチ（rapidfuzz）で照合。過去商談記録をメール生成のコンテキストに追加する |
| `web_searcher.py` | DuckDuckGo（`ddgs`）で企業情報を2クエリ並走取得。クエリA: 事業内容・製品（期間制限なし）、クエリB: 最新ニュース（直近1ヶ月）|

### 音声処理

| ファイル | 役割 |
|---|---|
| `audio_processor.py` | Whisper APIで音声ファイルを文字起こしし、LLMでニーズを構造化抽出（`summary` / `issues` / `needs` / `budget` / `decision_maker` / `temperature`）。`mutagen` で音声メタデータも取得 |
| `audio_matcher.py` | ファイル名から担当者名・日時を解析し、リードCSVと照合。信頼度を🟢green/🟡yellow/🔴redで分類し、確認が必要なものをユーザーに提示 |

### 出力

| ファイル | 役割 |
|---|---|
| `gmail_drafter.py` | Gmail API（OAuth 2.0）で生成済みメールを下書き保存。初回のみブラウザ認証が必要。その後は `credentials/token.json` を自動利用 |
| `calendar_client.py` | Google Calendar API で空き時間を取得する。`fetch_free_slots()` で平日・稼働時間内・予定重複なしの枠を返し、`format_slots_for_email()` でメール挿入用テキストに整形。Gmail と同じ `credentials/token.json` を使用 |
| `utils.py` | CSV読み込み・カラムマッピング・ロガー・品質チェック等の汎用ヘルパー。各モジュールが共通で使う処理をここに集約 |
| `pdf_processor.py` | **VLM PDF テキスト化**。`extract_text_from_pdf_vlm()` で PyMuPDF + gpt-5.4-nano vision により PDF をページ画像化→Markdown テキスト抽出。`is_pdf()` で拡張子判定。`build_index()` から呼ばれる |

### KB 状態確認

`cli_runner.run_kb_status()` が ChromaDB から登録済みドキュメント・チャンク数・最終更新を取得して dict を返す。`/kb-status` Skill から呼ばれる読み取り専用 API。

---

## tests/ ── 自動テスト（pytest）

全テストはLLMをモック化しているためAPIコストゼロで実行できる。

| ファイル | テスト対象 | 内容 |
|---|---|---|
| `test_email_validator.py` | `email_validator.py` | Layer 1の4ルール（プレースホルダ・ダミーURL・関心外製品等）の正常系・異常系 |
| `test_email_judge.py` | `email_judge.py` | Layer 2（LLM-as-a-Judge）のスコアリング・合否判定ロジック |
| `test_3layer_integration.py` | `agent.py` 経由の統合 | Layer 1→Layer 2→出力の連鎖動作。ハルシネーション検出の回帰防止 |
| `test_email_generator_prompt_routing.py` | `email_generator.py` | ランク別プロンプトテンプレートの切り替えロジック |
| `test_column_mapping.py` | `utils.py` | 異なるCSVフォーマット（RX Japan/Sansan/HubSpot等）のカラム自動マッピング |
| `test_rank_normalizer.py` | `utils.py` / `rank_estimator.py` | ランク値の正規化・マッピング推定（`extract_unique_rank_values` / `normalize_rank_values` / `infer_rank_mapping_with_llm`） |
| `test_field_labels.py` | `config.py` | フィールド表示名（日本語ラベル）の正確性 |
| `test_rank_estimator.py` ※ | `rank_estimator.py` | ランク正規化ルール（★5→A等の変換） |
| `test_date_validator.py` | `date_validator.py` | 過去日・土日・遠すぎる日付の排除ロジック（freezegunで日時固定） |
| `test_validate_candidate_dates.py` | `cli_runner.py` | 商談候補日のバリデーション（日付フォーマット・範囲チェック） |
| `test_agent_schedule_logic.py` | `agent.py` | メール本文に日程提案を含めるか判断するロジック |
| `test_audio_processor_whisper_prompt.py` | `audio_processor.py` | Whisper APIへのプロンプトパラメータが正しく渡されているか |
| `test_temperature_calibration.py` | `audio_processor.py` | 音声から抽出するtemperature（購買意欲）の判定閾値 |
| `test_output_management.py` | `cli_runner.py` | 出力ファイルのパス管理（タイムスタンプ付きファイル名生成等） |
| `test_calendar_client.py` | `calendar_client.py` | Google Calendar API のモック化テスト（空き枠検索・土日除外・working_hours 境界・busy 重複・2営業日ルール・format_slots_for_email） |

---

## data/ ── 入力データ

| パス | 役割 |
|---|---|
| `data/leads.csv` | **実行で使うリードCSV**。LeadManager 実形式（22列：`No.` / `フォロー担当` / `対応状況` / `氏名` / `会社名` / `商談確度`〈`B：担当者フォロー` 等〉 / `興味のある商材` / `スキャン担当者` / `来場日` / `時間` / …）に準拠したデモデータ（18件）。商談確度は `/rank-mapping` で A〜E に正規化して使用する |
| `data/tech_documents/*.md` | 製品技術資料（Markdown形式）。RAGのナレッジベースソース |
| `data/tech_documents/*.pdf` | 技術資料のPDF版。pypdfでテキスト抽出しRAGに取り込む |
| `data/crm_demo.csv` | CRMデータのデモ用サンプル（独自フォーマット） |
| `data/crm_hubspot_demo.csv` | HubSpot形式のCRMデモ用サンプル |
| `data/leads_rx_demo.csv` | RX Japan Lead Manager形式のデモ用リードCSV |
| `data/test/*.csv` | 自動テスト専用のフィクスチャCSV（英語列・日本語列・欠損列・音声テスト用等） |

> **運用**: `data/tech_documents/` と `data/crm_records/` は「ユーザーが自社ファイルを置く場所」。  
> デモ用ファイルと自社ファイルを混在させないこと。

---

## docs/ ── 設計ドキュメント

| ファイル | 内容 |
|---|---|
| `hallucination-mitigation.md` | Phase 3.3.5で発見した2種類のハルシネーション事案の記録。真因・対策設計判断・3層検証アーキテクチャへの到達過程 |
| `improvement-roadmap.md` | フェーズ別の改善履歴とバックログ。「現在地」と「次に何があるか」を1ファイルで把握するため |
| `future-rag-design.md` | VLM・Reranking・Contextual Retrieval等、次フェーズのRAG拡張構想メモ |
| `project-structure.md` | **このファイル**。フォルダ・ファイル構成と各ファイルの意味を説明 |

---

## output/ ── 生成結果

| パス | 役割 |
|---|---|
| `output/emails.csv` | 最新の生成済みメールCSV（デフォルト出力先） |
| `output/emails_YYYYMMDD_HHMMSS.csv` | タイムスタンプ付きバックアップ（`output_naming: timestamp` 設定時） |
| `output/legacy/` | 旧バージョンの出力ファイル保管場所 |
| `output/.gitkeep` | 空ディレクトリをGitで管理するためのプレースホルダ |

---

## credentials/ ── Gmail認証情報

| ファイル | 役割 |
|---|---|
| `credentials/credentials.json` | Google Cloud ConsoleからダウンロードするOAuthクライアントID。**コミット禁止（.gitignore対象）** |
| `credentials/token.json` | 初回ブラウザ認証後に自動生成されるアクセストークン。**コミット禁止** |
| `credentials/.gitkeep` | ディレクトリ自体はGitで管理するためのプレースホルダ |

---

## ルートファイル群

| ファイル | 役割 |
|---|---|
| `cli_config.yaml` | **実行時設定ファイル**。`sender_company`（送信元会社名）・`sender_name`・`leads_csv_path`・`default_ranks`・`enable_web_search`・`enable_llm_judge`・`output_naming`・`calendar`（Google Calendar連携設定）等を管理。Skillsの対話で自動更新される |
| `requirements.txt` | pip依存パッケージの一覧。`.venv/Scripts/pip install -r requirements.txt` でインストール |
| `CLAUDE.md` | **Claude Codeへの指示書**。Skillsの呼び出し対応表・ビジネスロジックの呼び出し方・守るべきルール等をAIに伝える。Claude Codeはこのファイルを自動的に読む |
| `README.md` | プロジェクト概要（外部公開向け）。採用担当者・初見の開発者が最初に読む |
| `system_overview.md` | アーキテクチャ・処理フロー・コスト設計の詳細解説。READMEより技術的な読み物 |
| `.env` | `OPENAI_API_KEY` 等の秘匿情報。**コミット禁止** |
| `.env.example` | `.env` のサンプル（キー名だけ書いて値は空）。リポジトリに含める |
| `.gitignore` | Git管理対象外のファイルを指定（`.env` / `credentials/*.json` / `.venv/` / `chroma_db/` 等） |

---

## 実行時に自動生成されるディレクトリ

| パス | 役割 |
|---|---|
| `chroma_db/` | ChromaDBの永続化ストレージ。`run_build_kb()` 実行時に生成される |
| `chroma_db/parent_store.json` | 親子チャンクの親テキストを永続保存するJSONファイル |
| `.venv/` | Python仮想環境。`python -m venv .venv` で作成。依存ライブラリはすべてここに入る |

---

## モジュール間の依存関係（概略）

```
Claude Code Skills (.claude/commands/)
    │
    ▼
cli_runner.py  ← Skills唯一の入口
    │
    ├─► agent.py（オーケストレーター）
    │       │
    │       ├─► rank_estimator.py（ランク推定）
    │       ├─► vectordb.py（RAG検索）
    │       ├─► crm_matcher.py（CRM照合）
    │       ├─► web_searcher.py（Web検索）
    │       ├─► audio_processor.py（音声文字起こし）
    │       ├─► email_generator.py（メール生成）
    │       ├─► email_validator.py（Layer 1検証）
    │       └─► email_judge.py（Layer 2検証）
    │
    ├─► audio_matcher.py（音声紐づけ）
    ├─► gmail_drafter.py（Gmail下書き）
    ├─► calendar_client.py（Google Calendar空き時間取得）
    └─► utils.py（共通ヘルパー）
            │
            └─► config.py（設定・定数）
```
