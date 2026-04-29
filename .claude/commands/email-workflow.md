# email-workflow — メール生成ワークフロー（全体）

展示会フォローアップメールを生成して Gmail 下書きに保存するまでの
全体フローを案内するスキル。Step 1〜7 を順番に実行する。

---

## 実行フロー

### Step 1: 環境チェック

```bash
.venv/Scripts/python -c "
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
from src.cli_runner import run_check
result = run_check()
for item in result['items']:
    icon = {'ok': '[OK]', 'warning': '[!!]', 'error': '[NG]'}[item['status']]
    print(f'{icon} {item[\"label\"]}: {item[\"detail\"]}')
"
```

- `[NG]` が1件でもあれば原因を説明し、対処法を案内してから止まる
- `[!!]` は内容を説明した上で続行可能

---

### Step 2: リードCSVの確認

「leads.csv はどこにありますか？（デフォルト: `data/leads.csv`）」と確認する。

パスが確認できたら `/inspect-data` を案内し、データ品質を先に確認することを推奨する。
ユーザーが「すでに確認済み」「急いでいる」と言った場合はスキップして Step 3 へ進む。

---

### Step 3: ナレッジベース確認・構築

```bash
.venv/Scripts/python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from src.config import Config

chroma_sqlite = Path(Config.CHROMA_DB_DIR) / 'chroma.sqlite3'
if chroma_sqlite.exists() and chroma_sqlite.stat().st_size > 1000:
    print('KB: 構築済み')
else:
    print('KB: 未構築')
"
```

KB が未構築の場合：「ナレッジベースを構築しますか？（`data/tech_documents/` に Markdown を置いてから実行）」と確認してから：

```bash
.venv/Scripts/python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.cli_runner import run_build_kb
result = run_build_kb()
print(result['message'])
if not result['ok']:
    print('ERROR:', result['message'])
"
```

---

### Step 4: 送信元会社名の確認

```bash
.venv/Scripts/python -c "
import sys, yaml
sys.stdout.reconfigure(encoding='utf-8')
from src.cli_runner import load_cli_config
cfg = load_cli_config()
print('sender_company:', cfg.get('sender_company', ''))
"
```

`sender_company` が空なら「送信元会社名を教えてください」と確認し、設定を更新する：

```bash
.venv/Scripts/python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.cli_runner import load_cli_config, save_cli_config
cfg = load_cli_config()
cfg['sender_company'] = '株式会社XXX'  # ユーザーが入力した会社名に置き換える
save_cli_config(cfg)
print('送信元会社名を設定しました:', cfg['sender_company'])
"
```

---

### Step 5: 展示会情報の確認

「展示会名・開催日・会場を教えてください（任意ですが、メールの品質が向上します）」と確認する。

例：
- 展示会名: 「製造業 DX 展 2026」
- 開催日: 「2026年4月24日〜26日」
- 会場: 「東京ビッグサイト」

---

### Step 6: メール生成

確認した情報を使ってメールを一括生成する。対象ランクも確認する（デフォルト: A,B,C）：

```bash
.venv/Scripts/python -c "
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
from src.cli_runner import run_generate

result = run_generate(
    csv_path='data/leads.csv',
    ranks=['A', 'B', 'C'],
    sender_company='株式会社XXX',
    enable_web_search=True,
    enable_rank_estimation=True,
    output_path='output/emails.csv',
    exhibition_name='製造業 DX 展 2026',
    exhibition_date='2026年4月24日〜26日',
    exhibition_venue='東京ビッグサイト',
)
print(result['message'])
if result['errors'] > 0:
    print(f'[!!] エラーあり: output/emails.csv の subject=ERROR 行を確認してください')
"
```

生成中は進捗を `on_progress` で逐次表示する（件数・会社名・氏名）。

---

### Step 7: Gmail 下書きへの保存

生成が完了したら「Gmail の下書きフォルダに保存しますか？」と確認する。

「はい」の場合：

```bash
.venv/Scripts/python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.cli_runner import run_draft_to_gmail

result = run_draft_to_gmail(
    output_csv_path='output/emails.csv',
)
print(result['message'])
if result['errors'] > 0:
    print('[!!] エラー詳細:')
    for d in result['error_details']:
        print(f'  - {d}')
"
```

- **初回実行時**: ブラウザが開き Google アカウントの認証を求める。  
  `credentials/credentials.json` が必要（Google Cloud Console からダウンロード）。
- **2回目以降**: `credentials/token.json` が自動利用されるため認証不要。

---

### 完了メッセージ

```
━━━ 展示会フォローアップ完了 ━━━

■ メール生成: XX件成功 / Y件エラー
  出力ファイル: output/emails.csv

■ Gmail 下書き: XX件作成
  Gmail を開いて「下書き」フォルダを確認してください。

エラーがある場合: output/emails.csv の subject='ERROR' 行を確認してください。

━━━ お疲れ様でした ━━━
```

---

## ショートカット（よく使うバリエーション）

### 特定ランクのみ生成
`/email-workflow` の引数に「Aランクだけ」などと添えると ranks を変更する。

### Web検索を無効化
通信環境が悪い場合は `enable_web_search=False` に変更して実行。

### 音声コンテキストを含める
事前に `/audio-matching` で紐づけと文字起こしを完了させておくこと。
文字起こしが完了していると `run_generate()` がそれを自動参照する（Streamlit 経由の場合のみ。CLI 経由では手動で `transcript` を渡す必要がある）。

---

## 注意事項

- KB が未構築の状態で generate を実行しない
- プロジェクトルートから実行すること（相対パスが壊れる）
- Gmail 下書き機能は `credentials/credentials.json` が必要（詳細は README 参照）
- Streamlit UI（`app.py`）は起動しない（ユーザーから明示的に求められない限り）
