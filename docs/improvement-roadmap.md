# 改善ロードマップ

**バージョン**: 2026年5月  
**現在地**: Phase 3.3.6 準備（docs/ 整備中）

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

### 2.2 進行中（Phase 3.3.6 準備・docs/ 整備）

Phase 3.3.6 に着手する前に、ノウハウをドキュメントとして集約している段階。

| タスク | 内容 | 状態 |
|---|---|---|
| **B-0** | `AUDIO_TIMESTAMP_TOLERANCE_MINUTES` の不一致解消（`config.py` と `CLAUDE.md` と `audio_matcher.py` を `5` に統一） | ✅ 完了（コミット `91c910e`） |
| **B-1** | `docs/hallucination-mitigation.md` 作成 | ✅ 完了 |
| **B-2** | `docs/improvement-roadmap.md` 作成（このドキュメント） | 🔄 作成中 |
| **B-3** | `docs/onboarding-design.md` 作成 | 🔲 未着手 |
| **B-4** | `docs/vector-db-design.md` 作成 | 🔲 未着手 |

### 2.3 残フェーズ

**Phase 3.3.6: 日程機能 + プロンプト系まとめ修正**

`/email-workflow` に Step 5.5（候補日入力 + 対象ランク選択）を新設する。Python 側で
日付検証（過去日 NG）と曜日整形を実装。ランク別分岐で A/B は候補日提示、C は情報提供型
に切り分ける。プロンプトには「関心外製品への誘導禁止 + 主要1〜2製品に絞る」を追加する。

**Phase 4: Gmail 下書き保存**

OAuth 初回認証（ブラウザ認証フロー）を通して `run_draft_to_gmail()` を実行し、15件を
Gmail 下書きに保存。生成されたメールを1件ずつ目視確認する。

**Phase 5: 個別 Skill 動作確認**

`/inspect-data` / `/csv-mapping` / `/audio-matching` / `/match-records` の各 Skill を
単独で起動し対話フローを検証する。音声ファイルと CSV の紐づけテスト、CRM 情報がある
ケースの動作確認も含む。

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
以上あり、Phase 3.3.6 でさらに増える予定。可読性・テスタビリティが低下している。

**想定実装**: `ProcessContext` / `GenerationContext` 等のデータクラスに引数をまとめ、
呼び出し元をシンプルにする。既存テストへの影響があるため Phase 6 以降でまとめて実施。

**優先度**: 中（機能追加が止まった後にまとめて整理）

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

*更新日: 2026年5月*
