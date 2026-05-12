# 展示会フォローアップ AI エージェント

展示会で集めたリード情報から、AI が個別最適化したフォローアップメールを
自動生成し、Gmail の下書きに保存するシステム。

## 特徴

- **複数ツール対応**: Lead Manager / Q-PASS / Sansan など、異なる CSV 形式を自動マッピング
- **個別最適化メール**: 製品資料 RAG × 音声文字起こし × Web 検索で一人ひとりに最適な文面を生成
- **ランク別対応**: A/B ランク(候補日提示型) / C ランク(情報提供型) の自動分岐
- **3層品質検証**: Python 機械チェック → LLM-as-a-Judge → 人間確認
- **対話型操作**: Claude Code Skills による自然な操作感

## セットアップ

```bash
# 1. 依存パッケージをインストール
pip install -r requirements.txt

# 2. .env を作成して OPENAI_API_KEY を設定
echo "OPENAI_API_KEY=sk-..." > .env

# 3. Claude Code で /setup を実行
/setup
```

## 使い方

初回: セットアップガイドを起動

```
/setup
```

2回目以降: メール生成フロー

```
/email-workflow
```

## 主な Skill

| Skill | 機能 |
|---|---|
| `/setup` | 初回セットアップガイド（9ステップ） |
| `/email-workflow` | メール生成 → Gmail 下書き保存 |
| `/inspect-data` | リード CSV データ品質チェック |
| `/csv-mapping` | カラムマッピング確認 |
| `/audio-matching` | 音声録音 → リード紐づけ |
| `/match-records` | リード ↔ CRM 照合 |

## 技術スタック

| 分類 | 技術 |
|---|---|
| LLM | gpt-5.4-nano (OpenAI) |
| Embedding | text-embedding-3-small |
| RAG | ChromaDB + BM25 ハイブリッド + 親子チャンク |
| 音声 | Whisper |
| Gmail | Gmail API (OAuth 2.0 / `gmail.compose` スコープ) |
| UI | Claude Code Skills (Markdown) |

## ドキュメント

- [`system_overview.md`](system_overview.md) — システム全体のアーキテクチャ・設計判断
- [`docs/improvement-roadmap.md`](docs/improvement-roadmap.md) — 改善ロードマップ・バックログ
- [`docs/hallucination-mitigation.md`](docs/hallucination-mitigation.md) — ハルシネーション対策ノウハウ
