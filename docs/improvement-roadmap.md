# 改善ロードマップ

**バージョン**: 2026年5月  
**現在地**: Phase 3.3.6 完了 → Phase 4（Gmail 下書き）待ち

---

## 1. このドキュメントの目的

「いま何をして、次に何があるか」をひとつの場所で確認できるようにするためのドキュメント。
完了済みフェーズの成果・残フェーズの計画・Phase 6 の改善バックログを集約する。

**想定読者**

- 将来の自分：再開時の文脈復元、優先度判断の参照
- 面接官：プロジェクトの継続的設計改善・段階的開発計画の確認素材

**関連ドキュメント**

- [system_overview.md](../system_overview.md) — システム全体のアーキテクチャ・処理フロー
  （フェーズ一覧は同資料 §10 にも記載あり。本ドキュメントはより詳細な開発ログとして共存）
- [docs/hallucination-mitigation.md](hallucination-mitigation.md) — Phase 3.3.5 の
  ハルシネーション対策詳細（Phase 6 の §3.2〜3.4 の背景資料）
- [docs/future-rag-design.md](future-rag-design.md) — RAG 設計の将来計画メモ

---

## 2. プロジェクト全体ロードマップ

### 2.1 完了済みフェーズ

| フェーズ | 概要 | 成果 |
|---|---|---|
| **Step A** | v1（Streamlit版）を `legacy-streamlit-v1` ブランチに退避 | v1 コードを保持しつつ v2 開発を開始できる状態 |
| **Step B** | `app.py` / `cli.py` 削除、`CLAUDE.md` 大掃除 | v2 の Claude Code Skills ベース構造を確立 |
| **Step D** | Skill 内ハードコード値のプレースホルダー化 | 企業名・設定値をコードから分離 |
| **Step E** | Gmail 件名の `Header()` エンコード対応 | 日本語件名が Gmail で文字化けしなくなった |
| **Step F** | `system_overview.md` 章立て合意・刷新、`MIGRATION.md` 削除 | 移行完了ドキュメントの整理 |
| **Phase 1** | 正常系の環境チェック（`run_check()`） | API キー・KB・CSV・Gmail 認証の一括確認 |
| **Phase 1.5** | 異常系の環境チェック | エラーケースの網羅的な検出・メッセージ整備 |
| **Phase 1.6** | P0: KB 破壊バグ修正 / P1: KB ヘルスチェック / P2: tech_documents 状態別エラー / P3: credentials.json メッセージ修正 | KB 関連の主要バグをまとめて解消 |
| **Phase 2** | リード CSV 確認（カラムマッピング動作検証） | Lead Manager 形式での自動マッピング確認 |
| **Phase 2.5** | マッピング実力テスト（日本語・英語の表記ゆれ実証） | 11件中10件成功。未対応カラム候補（LeadID, VisitDate 等）を特定 |
| **Phase 3.1** | メール生成1件テスト | パイプライン全体（RAG + CRM + Web + 生成）の疎通確認 |
| **Phase 3.1.5** | 送信元担当者（`sender_name`）の設定機能追加 | 署名・名乗りに担当者名を自動反映 |
| **Phase 3.2** | 14件の一括生成 | `run_generate()` の量産動作確認 |
| **Phase 3.2.5** | `OUTPUT_FORMAT_INSTRUCTION` の f-string 変数展開漏れバグ修正 | `{会社名}` リテラルがメール本文に混入するバグを解消 |
| **Phase 3.3** | 全15件の品質レビュー | ランク別トーン・構成・文体の品質確認。ハルシネーション事案を発見 |
| **Phase 3.3.5** | ハルシネーション対策（lead_id 漏洩 + URL 自己生成） | 禁止指示の二重防衛 + `product_urls` 設計で問題を抑制。詳細は [hallucination-mitigation.md](hallucination-mitigation.md) |
| **Phase 3.3.6** | 日程機能 + プロンプト系まとめ修正 | 候補日入力・検証（`date_validator`）、曜日 Python 計算、ランク別分岐（A/B=候補日提示・C=情報提供型）、関心外製品禁止・主要1〜2絞り込みルールを実装。52件テスト全パス、代表4リード実機検証で全観点○ |

### 2.2 ドキュメント整備（Phase 3.3.6 と並行）

| タスク | 内容 | 状態 |
|---|---|---|
| **B-0** | `AUDIO_TIMESTAMP_TOLERANCE_MINUTES` の不一致解消（`config.py` と `CLAUDE.md` と `audio_matcher.py` を `5` に統一） | ✅ 完了（コミット `91c910e`） |
| **B-1** | `docs/hallucination-mitigation.md` 作成 | ✅ 完了 |
| **B-2** | `docs/improvement-roadmap.md` 作成（このドキュメント） | ✅ 完了 |
| **B-3** | `docs/onboarding-design.md` 作成 | 🔲 未着手 |
| **B-4** | `docs/vector-db-design.md` 作成 | 🔲 未着手 |

### 2.3 残フェーズ

**Phase 4: Gmail 下書き保存**

OAuth 初回認証（ブラウザ認証フロー）を通して `run_draft_to_gmail()` を実行し、15件を
Gmail 下書きに保存。生成されたメールを1件ずつ目視確認する。

**Phase 5: 個別 Skill 動作確認**

`/inspect-data` / `/csv-mapping` / `/audio-matching` / `/match-records` の各 Skill を
単独で起動し対話フローを検証する。音声ファイルと CSV の紐づけテスト、CRM 情報がある
ケースの動作確認も含む。

**音声テストデータ補足（Phase 5.4 調査結果）**: テストファイル `20260425_営業A_001.m4a` は
埋め込みタグなし（`start_time=None`）のため 🟢 green・🔴 red は未確認。
完全確認には以下が必要：
- 🟢 green 確認: mutagen で録音日時タグ付き m4a を作成
  （例: `from mutagen.mp4 import MP4; f["©day"] = "2026-04-25T10:30:00"`）
- 🔴 red 確認: 担当者名なしファイル（例: `20260425_001.m4a`）を作成

**Phase 6: 全体改善まとめ**

各フェーズで積み上がった改善項目バックログをまとめて実装するフェーズ。詳細は §3 参照。

**Phase 7: `/setup` Skill 新設（オンボーディング）**

初回利用者が最短でメール生成まで辿り着けるガイド Skill を新設する。必須・推奨・任意の
3分類で対話的にヒアリングする9ステップ構成（API キー / CSV / 送信元 / 展示会 / 技術資料 /
音声 / CRM / Gmail / ランク）。最終的に `cli_config.yaml` を自動更新する。
設計詳細は `docs/onboarding-design.md`（B-3）で先行整備予定。

**Phase 8: クリーン環境一気通貫テスト**

設定を全消しした状態から `/setup` → 技術資料投入 → メール生成 → Gmail 下書き保存まで
を一連で通すエンドツーエンドテスト。v2 の完成確認フェーズ。

---

## 3. Phase 6: 改善項目バックログ

各フェーズの開発・テスト中に発見・議論した改善候補をここに集約する。
Phase 6 で集中実装するが、急を要する場合は前倒しも可。

### 3.1 モデル切り替え機能

**背景**: 現在 `Config.LLM_MODEL` は `gpt-5.4-nano` 固定。高品質が必要なケースと
低コスト優先のケースで異なるモデルを選びたいニーズが想定される。

**想定実装**: `cli_config.yaml` に `model` キーを追加し、`run_generate()` が
`config.get("model", Config.LLM_MODEL)` で読み取る。選択肢は `gpt-5.4-nano` /
`GPT-4.1` / `claude-opus-4-7` 等。`/email-workflow` で用途別モードを選べるようにする。

**優先度**: 中（Phase 6 の中盤以降）

### 3.2 LLM-as-a-Judge による品質検証層

**背景**: Phase 3.3.5 で実装した禁止指示はハルシネーション抑制に有効だが、100%
保証できない。生成後の事後検証層が必要。詳細背景は
[hallucination-mitigation.md §6](hallucination-mitigation.md#6-禁止指示の効果と限界)。

**想定実装**: メール生成後に別 LLM 呼び出しで「ビジネスメールとして適切か」「内部 ID や
架空 URL が含まれないか」を自動評価。不合格の場合はリトライまたはフラグ付きで出力。

**優先度**: 高（§3.4 の3層構造の Layer 2 を担う）

### 3.3 Python による機械チェック層

**背景**: LLM による検証の前段に、コストゼロの正規表現チェックで弾ける違反を先に処理
したい。LLM-as-a-Judge の呼び出し頻度を減らすコスト最適化にもなる。

**想定実装**: 生成メールに以下を regex で検査。

```python
patterns = [
    r"L\d{3,}",                      # 内部ID（L007 等）
    r"example\.(com|org|net)",        # サンプルドメイン
    r"https?://\S+",                  # URL全般（product_urls 未設定時）
]
```

日付関連は `past_date_check()` で過去日・不正曜日も検証（Phase 3.3.6 の日程機能と連動）。

**優先度**: 高（実装コストが低い割に効果が高い）

### 3.4 3層検証アーキテクチャ

**背景**: §3.2 と §3.3 をまとめて設計として定義する。現在は Layer 3（人間確認）のみ
稼働しており、Layer 1・2 が欠けている。

```
Layer 1: Python 機械チェック（§3.3）← Phase 6 で実装
Layer 2: LLM-as-a-Judge（§3.2）    ← Phase 6 で実装
Layer 3: 人間確認（Gmail 下書き）   ← 現在稼働中
```

Layer 1 → Layer 2 → Layer 3 の順で処理し、Layer 1 で弾けたものは Layer 2 を呼ばない
設計にする（コスト効率）。

**優先度**: 高（§3.2 と §3.3 の組み合わせとして同時実装）

### 3.5 マッピング辞書拡充

**背景**: Phase 2.5 のマッピングテストで11件中10件成功だったが、`LeadID` /
`VisitDate` 等の標準外カラムが未対応として残った。`Config.REQUIRED_FIELDS` と
`Config.OPTIONAL_FIELDS` の候補リスト追加で対応できる。

**想定実装**: `src/config.py` の `REQUIRED_FIELDS` / `OPTIONAL_FIELDS` に未対応
カラム名を追記。追加の都度テスト CSV で再検証する。

**優先度**: 低（現行でも `extra_` プレフィックスで保持されるため実害は限定的）

### 3.6 エラーメッセージ日本語化

**背景**: 環境チェック（`run_check()`）の出力に、標準フィールド名（`visitor_name` 等）が
そのまま表示されるケースがある。営業担当者が見たときに意味が伝わらない。

**想定実装**: 標準名 → 表示用ラベルの変換辞書を `src/config.py` または `src/utils.py` に
追加する。エラーメッセージ生成時に変換して出力。

**優先度**: 低（影響範囲は UI 上のメッセージのみ）

### 3.7 引数肥大化対応（リファクタ候補）

**背景**: `agent.process_lead()` と `email_generator.generate()` の引数は現在10個
以上あり、Phase 3.3.6 でさらに増えた。可読性・テスタビリティが低下している。

**想定実装**: `ProcessContext` / `GenerationContext` 等のデータクラスに引数をまとめ、
呼び出し元をシンプルにする。既存テストへの影響があるため Phase 6 以降でまとめて実施。

**優先度**: 中（機能追加が止まった後にまとめて整理）

### 3.8 BM25 relevance score 負値 UserWarning

**背景**: Phase 3.3.6 実機検証（2026-05-07）で `UserWarning: Relevance scores must be
between 0 and 1` が頻出。`VectorDBManager.hybrid_search()` の BM25 スコアが負になるケース
があり、ChromaDB が処理途中で警告を出す。処理自体は継続され生成品質への影響は未確認。

**想定対処**: BM25 スコアを `max(score, 0.0)` でクランプするか、RRF 統合後にスコアを
正規化する。`src/vectordb.py` の `similarity_search_with_relevance_scores()` 呼び出し箇所
または `hybrid_search()` 内のスコア合算ロジックを修正する。

**優先度**: 低（警告のみで動作継続。品質への実害が確認されてから対処でよい）

### 3.9 多製品リード（3〜4製品）での関心製品省略問題

**背景**: Phase 3.3.6 実機検証で L013（関心製品 Sorani/DigiMA/EdgeGuard/FactoryBrain の
4製品）に対して生成されたメールが FactoryBrain を省略。「主要1〜2製品に絞る」ルールの
LLM 判断によるもので、絞り込み自体は正常動作。ただし意図せず重要製品が毎回抜けるリスクが
ある（再現性はランダム）。

**想定対処**: Phase 3.3.6.4 のランク別プロンプトで「絞り込む場合は関心製品リストの
先頭から優先せよ」という順序ヒントを追加するか、絞り込み選択の理由をログに残す。

**優先度**: 低（現状の絞り込みはメール品質として自然。省略リスクの定量評価が先）

### 3.10 Google Calendar 連携（候補日自動抽出）

**背景**: Phase 3.3.6 で実装した候補日提案（Step 5.5）は、ユーザーが手動で空き時間を
入力する前提。実運用では「自分の空き時間がわからない」「全リードに同じ日程を使い回す
しかない」という構造的課題がある。

**想定実装**: Google Calendar API で自分のカレンダーを参照し、指定期間内の空き時間を
自動抽出して `candidate_dates` に変換する。`/email-workflow` Step 5.5 を「自動取得 /
手動入力」の2モードに拡張する。認証は Gmail OAuth と同じトークンにスコープ追加
（`https://www.googleapis.com/auth/calendar.readonly`）で対応可能。

**優先度**: 低（Phase 8 以降。基本機能完成後の上乗せ）

### 3.11 Gmail セットアップヘルスチェック＋手順ガイド機能

**背景**: Phase 4 で `google-auth` 系パッケージの未インストール・credentials.json の
配置が前提条件となった。初回ユーザーは Google Cloud Console での OAuth クライアント作成
から始める必要があり、`run_check()` では「credentials.json 配置済み」の存在確認しか
していない（中身の妥当性は未検証）。

**想定実装**: `run_check()` の Gmail credentials チェックを以下に強化する。
- credentials.json の `installed` / `web` キー存在確認（OAuth クライアント形式検証）
- token.json の有効期限（`expiry` フィールド）を読み取り、期限切れを事前警告
- 必要スコープ（`gmail.compose`）が token に含まれるか確認
- 不足・不正時は Google Cloud Console の設定手順を日本語テキストで案内

**優先度**: 中（Phase 7 の `/setup` Skill Step 8 強化として実装。初回ユーザー体験に直結）

### 3.12 output/ 世代管理ルール

**背景**: Phase 4 で output/emails.csv（Phase 3.2 旧版・15件）と
output/test_phase336_results.json（Phase 3.3.6 新版・4件）が混在し、どちらを
Gmail ドラフトに使うかの判断コストが発生した。プロンプト更新のたびに旧出力が
残るとドラフト対象の取り違えリスクが生じる。

**想定実装**: 以下のいずれか（どちらかを選択または併用）。
- 命名規則の統一: `output/{YYYYMMDD}_{phase}_{exhibition}.csv` 形式で世代を付与
- レガシー自動退避: `run_generate()` 実行時に既存 emails.csv を `output/legacy/` へ移動
- `run_generate()` に `--overwrite / --new-file` オプション追加

**優先度**: 中（Phase 6。品質改善フェーズで output/ 運用ルール策定と同時に対応）

### 3.13 lead_id 自動上書き問題

**背景**: Phase 5.4 で `data/test/leads_audio_test.csv` の `AT001/AT002/AT003` という
独自 lead_id が `auto_map_columns` によって `L001/L002/L003` に置き換えられた。
元の lead_id は `extra_lead_id` に退避されたが、外部 CSV に独自 ID が存在するケースでは
データ整合性リスクになる。

**想定実装**: `auto_map_columns()` に lead_id 特例処理を追加。既存の lead_id 列がある場合は
`extra_original_id` として退避し、マッピングによる上書きを行わない。

**優先度**: 中（Phase 6）

### 3.14 Whisper プロンプト補正による認識精度向上

**背景**: Phase 5.4 の文字起こしで「プレスキュー」が「プレス機の」に誤認識された可能性。
業界用語・固有名詞は Whisper のデフォルト設定では精度が落ちる。

**想定実装**: `openai.audio.transcriptions.create()` の `prompt` パラメータに展示会名・
製品名・技術用語を渡す。`cli_config.yaml` に `whisper_prompt` フィールドを追加し、
ユーザーが現場用語を登録できるようにする。

**優先度**: 中（Phase 6）

### 3.15 temperature 判定キャリブレーション

**背景**: Phase 5.4 で「デモ希望・予算確定（約400万円）・自己承認権限あり」という情報が
`temperature=中` と判定された。このケースは「高」相当であり、判定ロジックの精度向上が必要。

**想定実装**: `extract_needs()` の temperature 判定プロンプトに明示的なルールを追加。
「予算金額確認済み + 意思決定権あり + デモ要求あり → 高」のような複合条件を
ルールとして列挙する。

**優先度**: 中（Phase 6）

---

## 4. 開発中に発見した Improvement Examples

開発・ドキュメント作業中に偶発的に発見した改善候補の記録。同種の発見があった場合は
同形式でここに追記する。

### 4.1 AUDIO_TIMESTAMP_TOLERANCE_MINUTES 死に変数問題（B-0 で解消）

**発見の経緯**

docs 整備（B-1 の hallucination-mitigation.md 執筆準備）のために `CLAUDE.md` を精読
していたところ、`AUDIO_TIMESTAMP_TOLERANCE_MINUTES: int = 10` という記載を発見。
`src/config.py` を確認すると `= 5` と異なっていた。さらに調査すると
`src/audio_matcher.py:94` で `tolerance_minutes: int = 10` とハードコードされており、
`Config` 定数が実際には参照されていない（死に変数）状態だった。

**対処**

`src/audio_matcher.py` に `from src.config import Config` を追加し、デフォルト引数を
`Config.AUDIO_TIMESTAMP_TOLERANCE_MINUTES` に変更。`CLAUDE.md` の記載も `5` に統一。
（コミット `91c910e`）

**追記テンプレート**

同種の発見があった場合は以下の形式でここに追記する。

```
### 4.N タイトル（対応状況）

**発見の経緯**: どの作業中に、どのファイルを見て気づいたか

**対処**: 何を変更したか、コミットハッシュ

**残課題（あれば）**: 今回対応できなかった部分
```

---

## 5. MVP 原則と優先度

**基本方針**: 動くシステムを最短で完成させてから品質改善する。

Phase 8（一気通貫テスト）が完了するまでは、機能の追加・修正を優先し、リファクタや
品質改善は Phase 6 に先送りする。ただし以下の条件に該当する場合は前倒しして対応する。

- バグが発生しているか、将来バグの原因になる確度が高い（§4 の死に変数ケースが該当）
- 修正コストが小さく、Phase 6 まで放置するリスクの方が大きい

**各フェーズの品質改善タイミング**

| 対象 | 改善タイミング | 備考 |
|---|---|---|
| プロンプト内容・禁止指示 | Phase 3.3.6 | 機能追加に伴い同時修正 |
| 検証層（Python + LLM-as-a-Judge） | Phase 6 | §3.2〜3.4 |
| リファクタ（引数肥大化） | Phase 6 | §3.7 |
| マッピング辞書・エラーメッセージ | Phase 6 | §3.5〜3.6 |
| モデル切り替え | Phase 6 | §3.1 |
| オンボーディング Skill | Phase 7 | `/setup` 新設 |

---

*更新日: 2026年5月10日（Phase 5.4 完了・§3.13〜3.15 追記・Phase 5 音声テストデータ補足追記）*
