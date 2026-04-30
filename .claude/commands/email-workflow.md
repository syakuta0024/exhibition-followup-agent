# email-workflow — メール生成ワークフロー（全体）

展示会フォローアップメールを生成して Gmail 下書きに保存するまでの
全体フローを案内するスキル。Step 0〜7 を順番に実行する。

---

## 実行フロー

### Step 0: 現在の設定確認

```bash
.venv/Scripts/python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.cli_runner import load_cli_config, mask_api_key
from src.config import Config

cfg = load_cli_config()

print('━━━ 現在の設定 ━━━')
sender = cfg.get('sender_company', '')
if sender:
    print(f'■ 送信元会社名: {sender}（保存済み）')
else:
    print('■ 送信元会社名: 未設定 → 入力してください')

leads_path = cfg.get('leads_csv_path', 'data/leads.csv')
label = 'カスタム' if leads_path != 'data/leads.csv' else 'デフォルト'
print(f'■ リードCSV:    {leads_path}（{label}）')

output_path = cfg.get('output_path', 'output/emails.csv')
label2 = 'カスタム' if output_path != 'output/emails.csv' else 'デフォルト'
print(f'■ 出力先:       {output_path}（{label2}）')

api_msg = mask_api_key(Config.OPENAI_API_KEY)
print(f'■ APIキー:      {api_msg}')
print()
print('■ 今回のイベント情報（毎回入力）:')
print('  展示会名 / 開催日 / 会場 → これから確認します')
print()
print('変更がある場合は教えてください。なければ「OK」と返してください。')
"
```

- APIキー未設定 → 対処法を案内して止まる
- 送信元会社名が未設定 → Step 4 でのオーバーライド入力を案内する
- それ以外は「変更がある場合は教えてください。なければ「OK」と返してください。」と確認してから Step 1 へ進む

**設定を変更する場合のフロー**:

ユーザーが `sender_company` / `leads_csv_path` / `output_path` の変更を希望したら、
値を受け取り以下のコードで保存してから設定を再表示する:

```bash
.venv/Scripts/python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.cli_runner import load_cli_config, save_cli_config
cfg = load_cli_config()
cfg['sender_company'] = '（ユーザーが入力した値）'  # 変更するキーと値を更新
save_cli_config(cfg)
print('設定を保存しました')
"
```

保存後は Step 0 の確認コードを再実行して「この設定でよろしいですか？」と確認してから Step 1 へ進む。

---

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

Step 0 の設定と Step 5 の展示会情報をもとにメールを一括生成する。
送信元会社名・ランク・CSV パス等は `cli_config.yaml` から自動参照する：

```bash
.venv/Scripts/python -c "
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
from src.cli_runner import run_generate

result = run_generate(
    # 展示会情報（毎回入力）— Step 5 で確認した値を入力
    exhibition_name='',   # ← Step 5 で確認した展示会名（空欄可）
    exhibition_date='',   # ← Step 5 で確認した開催日（空欄可）
    exhibition_venue='',  # ← Step 5 で確認した会場（空欄可）
    # sender_company / ranks / csv_path 等は cli_config.yaml から自動参照
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
文字起こしが完了していると `run_generate()` がそれを自動参照する（Skills 版では `transcript` を手動で引数に渡す必要がある）。

---

## 注意事項

- KB が未構築の状態で generate を実行しない
- プロジェクトルートから実行すること（相対パスが壊れる）
- Gmail 下書き機能は `credentials/credentials.json` が必要（詳細は README 参照）
