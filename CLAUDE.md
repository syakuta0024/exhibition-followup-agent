# CLAUDE.md — 展示会フォローアップAIエージェント

---

## Claude Code との連携（Skills モード）

### ユーザーの発言 → 呼ぶ Skill

| ユーザーの発言 | 呼ぶ Skill |
|---|---|
| 「メールを作って」「フォローアップしたい」「来場者にメール送りたい」 | `/email-workflow` |
| 「CSVのカラムを確認して」「名前のゆれがある」「マッピングを見て」 | `/csv-mapping` |
| 「音声ファイルを紐づけて」「録音を紐づけたい」 | `/audio-matching` |
| 「データの状態を確認して」「CSVを点検して」 | `/inspect-data` |
| 「CRMと照合して」「名寄せしたい」 | `/match-records` |

### Skills 一覧（.claude/commands/）

| Skill | ファイル | 役割 |
|---|---|---|
| `/email-workflow` | [.claude/commands/email-workflow.md](.claude/commands/email-workflow.md) | メール生成全体フロー（Step 0〜Gmail下書きまで） |
| `/csv-mapping` | [.claude/commands/csv-mapping.md](.claude/commands/csv-mapping.md) | カラムマッピング確認 + 名前表記ゆれの対話確認 |
| `/audio-matching` | [.claude/commands/audio-matching.md](.claude/commands/audio-matching.md) | 音声ファイル紐づけ・文字起こし |
| `/inspect-data` | [.claude/commands/inspect-data.md](.claude/commands/inspect-data.md) | データ品質チェック（読み取り専用） |
| `/match-records` | [.claude/commands/match-records.md](.claude/commands/match-records.md) | リード ↔ CRM の紐づけ確認 |

### ビジネスロジックの呼び出し方（Skills 内での標準パターン）

Skills は `cli.py` ではなく `src/cli_runner.py` の関数を直接呼ぶ：

```bash
# 環境チェック
.venv/Scripts/python -c "
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
from src.cli_runner import run_check
result = run_check()
for item in result['items']:
    icon = {'ok': '[OK]', 'warning': '[!!]', 'error': '[NG]'}[item['status']]
    print(f\"{icon} {item['label']}: {item['detail']}\")
"

# ナレッジベース構築
.venv/Scripts/python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.cli_runner import run_build_kb
result = run_build_kb()
print(result['message'])
"

# メール生成
.venv/Scripts/python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.cli_runner import run_generate
result = run_generate(
    sender_company='株式会社XXX',
    exhibition_name='展示会名',
    exhibition_date='2026年4月10日〜12日',
    exhibition_venue='東京ビッグサイト',
)
print(result['message'])
"

# Gmail 下書き保存
.venv/Scripts/python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.cli_runner import run_draft_to_gmail
result = run_draft_to_gmail()
print(result['message'])
"

# 設定の確認・変更
.venv/Scripts/python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.cli_runner import load_cli_config, save_cli_config
cfg = load_cli_config()
print(cfg)
# 変更する場合:
# cfg['sender_company'] = '株式会社XXX'
# save_cli_config(cfg)
"
```

### 注意事項（Claude Code が守るルール）

- **プロジェクトルートから実行**すること（相対パスが壊れる）
- KB 未構築の状態で `run_generate()` を実行しない（`run_check()` で確認してから）
- 出力ファイルは `output/emails.csv`（デフォルト）に保存される
- Gmail 下書き機能は `credentials/credentials.json` が必要（初回のみブラウザ認証）

### Skills 共通の対話ルール

全 Skill の確認ステップでのユーザー応答の解釈:

| 応答の種類 | 例 | 対応 |
|---|---|---|
| 明確な肯定 → 次のステップへ進む | 「OK」「はい」「いいよ」「進めて」「了解」「お願いします」 | 次のステップへ即時進む |
| 曖昧な応答 → 確認を取る | 「うん」「大丈夫」「いいです」「まあ」 | 「進めてもよろしいですか？」と再確認してから進む |
| 否定・追加要望 → 対応 | 「いや」「待って」「変更したい」など | ユーザーの意図を汲み取り対応してから再確認 |

---

## プロジェクト概要

展示会で収集したリードCSVをもとに、来場者1人ひとりに最適化したフォローアップメールをAIで生成するClaude Code Skills ベースのエージェント。

- **想定ユーザー**: 営業担当（展示会翌日にフォローメールを一括送信したい）
- **LLM**: OpenAI `gpt-5.4-nano`（APIキー必要）
- **会社非依存**: 特定企業名はコードに含まない。送信元会社名はUIで入力し `sender_company` として全処理に伝播する

---

## 環境セットアップ

`.env` ファイルに `OPENAI_API_KEY=sk-...` を設定すること。

### ⚠️ パッケージインストール時の注意

このプロジェクトは `.venv` 仮想環境で動作する。新しいパッケージは必ず `.venv` 内に入れること。

```bash
# NG: グローバルPythonにインストールしても .venv からは見えない
pip install some-package

# OK: .venv のpipを直接指定する
.venv/Scripts/pip install some-package
# または .venv をアクティブにしてから
.venv/Scripts/activate && pip install some-package
```

---

## ディレクトリ構成

```
exhibition-followup-agent/
├── cli_config.yaml         # 設定（sender_company・ランク・パスなど）
├── output/                 # メール生成結果CSV（output/emails.csv）
├── credentials/            # Gmail OAuth 認証情報（.gitignore 対象）
│   ├── credentials.json    # Google Cloud からダウンロード（コミット禁止）
│   └── token.json          # 初回認証後に自動生成（コミット禁止）
├── .venv/                  # 仮想環境（依存ライブラリはここに入れる）
├── .claude/
│   └── commands/           # Skills（スラッシュコマンド）
│       ├── email-workflow.md   # /email-workflow: メール生成全体フロー
│       ├── csv-mapping.md      # /csv-mapping: カラムマッピング確認
│       ├── audio-matching.md   # /audio-matching: 音声ファイル紐づけ
│       ├── inspect-data.md     # /inspect-data: データ品質チェック
│       └── match-records.md    # /match-records: CRM紐づけ確認
├── src/
│   ├── config.py           # 設定・定数（モデル名・料金・フィールド定義）
│   ├── agent.py            # オーケストレーター（FollowUpAgent）
│   ├── vectordb.py         # ChromaDB + BM25 ハイブリッド検索 + 親子チャンク
│   ├── email_generator.py  # LLMメール生成（EmailGenerator）
│   ├── rank_estimator.py   # ランク正規化 + LLM推定（RankEstimator）
│   ├── web_searcher.py     # DuckDuckGo企業情報検索（WebSearcher）
│   ├── crm_matcher.py      # CRM CSV紐付け（メール完全一致 + 社名ファジーマッチ）
│   ├── audio_processor.py  # Whisper文字起こし + LLMニーズ抽出（AudioProcessor）
│   ├── audio_matcher.py    # ファイル名解析 + リード紐づけエンジン（AudioMatcher）
│   ├── gmail_drafter.py    # Gmail API で下書き作成（GmailDrafter）
│   ├── cli_runner.py       # Skills共通ビジネスロジック（run_check/build_kb/generate/draft）
│   └── utils.py            # ロガー・品質チェック等ユーティリティ
├── data/
│   ├── leads.csv           # リードデータ（入力）
│   ├── tech_documents/     # 製品技術資料 Markdown（空ディレクトリ。自社MDを配置する）
│   └── crm_records/        # 商談記録 Markdown（空ディレクトリ。自社MDを配置する）
├── chroma_db/              # ChromaDB永続化ディレクトリ
│   └── parent_store.json   # 親チャンクのテキスト保存（親子チャンク用）
├── system_overview.md      # システム全体説明資料（背景・アーキテクチャ・使い方）
└── requirements.txt
```

---

## 処理フロー（FollowUpAgent.process_lead）

1. **ランク正規化/推定** — `★5→A` 等の変換。失敗時はLLMで推定
2. **RAG検索（技術資料）** — 親子チャンク：子チャンクで検索し親チャンクをLLMに渡す
3. **CRM紐付け** — CRM CSVがあればファジーマッチ、なければベクトル検索
4. **Web検索** — 企業の事業内容・製品・最新動向を2クエリで取得（APIキー不要）
5. **メール生成** — RAG文脈 + CRM + Web情報 + **音声コンテキスト（最優先）** をもとにLLMでメール生成

`audio_context` が存在する場合、`EmailGenerator` がプロンプト最上部に `## ★最優先情報（録音音声より）` セクションを挿入する。音声なしリードはデフォルト値（空文字）で従来通りに動作。

---

## Web検索設計（src/web_searcher.py）

`WebSearcher.search_company()` は2クエリを並走して企業情報を収集する。

| クエリ | 内容 | 期間制限 |
|---|---|---|
| クエリA | `{社名} 事業内容 製品 サービス` | なし（会社プロフィール） |
| クエリB | `{社名} ニュース プレスリリース 新製品` | 直近1ヶ月 |

- 各クエリ最大3件取得、URL重複は除去（profileを優先）
- 結果は `section: "profile" | "news"` で分類
- LLMへは `【事業内容・製品・サービス】` と `【最新動向】` の2セクション構造で渡す
- 使用パッケージ: `ddgs`（`duckduckgo-search` から改名。`pip install ddgs` で入れること）

---

## RAG設計（親子チャンク）

| ドキュメント種別 | 子チャンク（検索用） | 親チャンク（LLM文脈） |
|---|---|---|
| tech_doc | 250文字 / overlap 50 | 1000文字 / overlap 100 |
| crm_record | 350文字 / overlap 50 | 1200文字 / overlap 100 |
| pdf_upload | 400文字 / overlap 50 | 1400文字 / overlap 100 |

- **ハイブリッド検索**: ChromaDBベクトル検索 + BM25 → RRF統合スコア
- 親チャンクは `chroma_db/parent_store.json` に永続保存
- `build_index()` 実行時、PDFの親チャンクは消えないよう保護される

---

## ランク推定（RankEstimator）

- `normalize_rank()`: A-E そのまま / ★5・5・5.0→A 等をLLMなしで変換
- `estimate_from_lead()`: 変換できない場合にLLM推定（`enable_llm=False` で無効化可）
- method フィールド: `existing` / `normalized` / `llm_estimated` / `default`

---

## コスト設定（src/config.py）

```python
LLM_MODEL = "gpt-5.4-nano"
LLM_PRICE_INPUT_PER_1M  = 0.20   # $/1M 入力トークン
LLM_PRICE_OUTPUT_PER_1M = 1.25   # $/1M 出力トークン
EST_INPUT_TOKENS_PER_LEAD   = 2200  # メール生成1件あたり推定入力
EST_OUTPUT_TOKENS_PER_LEAD  = 400   # メール生成1件あたり推定出力
EST_RANK_EXTRA_INPUT_TOKENS = 300   # ランク推定有効時の追加入力
EST_SECONDS_PER_LEAD_BASE   = 8.0   # メール生成の推定秒数/件
EST_SECONDS_PER_LEAD_WEB    = 3.0   # Web検索追加分の推定秒数/件

# 音声文字起こし（Whisper API）
WHISPER_MODEL: str = "whisper-1"
WHISPER_PRICE_PER_MIN: float = 0.006          # $/分
WHISPER_MAX_FILE_MB: int = 25                 # Whisper API上限
AUDIO_TIMESTAMP_TOLERANCE_MINUTES: int = 5   # タイムスタンプ許容誤差（分）
AUDIO_RED_FLAG_WARNING_THRESHOLD: float = 0.30  # 赤フラグ率の警告閾値
```

---

## CSVカラムマッピング

リードCSVはRX Japan Lead Manager / Q-PASS / Sansan 等の異なるカラム名に対応。
`Config.REQUIRED_FIELDS` と `Config.OPTIONAL_FIELDS` で候補カラム名を定義。
マッピング外のカスタム質問列は `extra_` プレフィックスで自動保持され、メール生成のコンテキストに使用される。

---

## 送信元会社名（会社非依存設計）

アプリは特定の企業名をコードに持たない。送信元会社名は `cli_config.yaml` の `sender_company` または Skills の対話で入力し、`FollowUpAgent.process_lead(sender_company=...)` → `EmailGenerator.generate(sender_company=...)` → `_build_system_prompt(sender_company=...)` の順で挨拶・署名に使用される。

- 未入力の場合は `"弊社"` で代替（エラーにならない）
- システムプロンプトには会社名のみ渡す。製品・サービスの説明は RAG（tech_documents）から取得するため、プロンプトへのハードコードは禁止

### ナレッジベースディレクトリ

`data/tech_documents/` と `data/crm_records/` は **空ディレクトリ**として管理する。
利用者が自社のMarkdownファイルを配置し、「🔨 ナレッジベース構築」で取り込む運用。
デモ用の架空企業データをこれらのディレクトリにコミットしないこと。

---

## 音声紐づけ・文字起こし機能（src/audio_processor.py / src/audio_matcher.py）

展示会当日の録音音声をWhisperで文字起こしし、LLMでニーズを構造化抽出して、メール生成の最優先コンテキストとして活用する機能。

### ファイル命名規則

`YYYYMMDD_担当者名_連番.{mp3/m4a/wav}` を推奨（担当者名の自動解析精度が上がる）。

```
例: 20260424_営業A_001.m4a   → 担当者「営業A」として自動認識
    営業A_001.mp3            → 日付なしでも可
    rec001.m4a               → 担当者名不明 → 🔴 red 扱い
```

### AudioMatcher 紐づけロジック

| 条件 | 信頼度 | 手動対応 |
|---|---|---|
| タイムスタンプ + 担当者名 一致（±10分） | 🟢 green（自動確定） | 不要 |
| 担当者名一致・タイムスタンプなし | 🟡 yellow（順番で仮紐づけ） | 要確認 |
| 担当者名不明 | 🔴 red（未紐づけ） | 手動選択 |

- 赤フラグ率 > 30% → 命名ルール違反警告バナーを表示（運用改善フィードバックループ）
- 同一リードに複数ファイルが紐づく場合、再生時間が長い方を優先採用

### AudioProcessor

- `mutagen` でメタデータ（再生時間・録音日時 ID3 TDRCタグ）を取得
- `openai.OpenAI` で Whisper API 文字起こし（25MB超は `ValueError`）
- LLMで構造化ニーズ抽出: `{summary, issues, needs, budget, decision_maker, temperature}`

### AudioProcessor 実装上の注意（ハマりポイント）

**mutagen の BytesIO 渡し方**:
```python
# NG: 位置引数で渡すと None が返る（フォーマット判別に失敗）
audio = MutagenFile(bio)

# OK: fileobj= + filename= をキーワード引数で渡す
audio = MutagenFile(fileobj=bio, filename=filename)
```

**mutagen の bool 判定**:
```python
# NG: タグが空の m4a は bool(audio) == False になる（self.tags が空dict）
if audio and audio.info:

# OK: None チェックで判定する
if audio is not None and audio.info is not None:
```

**Whisper API への m4a 渡し方**:
```python
# NG: mimetypes.guess_type(".m4a") が環境によって video/mp4 や None を返しWhisperが失敗
response = client.audio.transcriptions.create(model="whisper-1", file=bio)

# OK: 拡張子→MIMEマップを持ちタプル形式で明示指定
_MIME = {"mp3": "audio/mpeg", "m4a": "audio/mp4", "wav": "audio/wav", ...}
ext = filename.rsplit(".", 1)[-1].lower()
response = client.audio.transcriptions.create(
    model="whisper-1",
    file=(filename, bio, _MIME.get(ext, "audio/mpeg")),
)
```

---

## ナレッジベース管理（PDFの削除）

`VectorDBManager.remove_document(source_name: str) -> int` でPDFを削除できる。

- ChromaDB の該当チャンクを `where={"source_file": source_name}` で全削除
- `_parent_store` から該当エントリも削除し `parent_store.json` を更新
- BM25コーパスを自動再構築
- 戻り値: 削除したチャンク数

---

## 主要な依存ライブラリ

| ライブラリ | 用途 |
|---|---|
| `langchain` + `langchain-openai` | LLM・Embedding |
| `langchain-chroma` | ベクトルDB |
| `rank_bm25` | BM25ハイブリッド検索 |
| `rapidfuzz` | CRM社名ファジーマッチ |
| `ddgs` | Web検索（旧 `duckduckgo-search`、APIキー不要） |
| `pypdf` | PDFテキスト抽出（オプション、未インストール時はPDFアップロードUI非表示） |
| `mutagen` | 音声ファイルメタデータ取得（再生時間・録音日時抽出、オプション） |
| `pandas` | CSV操作 |
| `google-api-python-client` | Gmail API（下書き作成） |
| `google-auth-httplib2` | Gmail OAuth 2.0 認証 |
| `google-auth-oauthlib` | Gmail OAuth 2.0 認証フロー |
