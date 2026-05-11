# /setup — 初回セットアップガイド

展示会フォローアップエージェントを初めて使う方向けのオンボーディングスキル。
Step 1〜9 を順に進めることで、約 30 分でメール生成の準備が整います。

再実行時は既存の設定を表示した上で「変更しますか？」と確認します。

---

## 前提

- Python 環境（3.10 以上）のインストール済み
- `.venv` 仮想環境に `pip install -r requirements.txt` 完了済み
- プロジェクトルートで実行すること

---

## 実行フロー

### Step 1: OPENAI_API_KEY の設定 [必須]

まず現在の API キー設定状態を確認する：

```bash
.venv/Scripts/python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.config import Config
from src.cli_runner import mask_api_key
from pathlib import Path

key_msg = mask_api_key(Config.OPENAI_API_KEY)
env_exists = Path('.env').exists()
print('OPENAI_API_KEY:', key_msg)
print('.env ファイル:', '存在' if env_exists else '未作成')
"
```

**APIキーが未設定の場合：**

APIキーはセキュリティ上の理由から対話では受け取りません。
ターミナルに入力した内容は会話ログに残るためです。
代わりに `.env` ファイルへの直接記載を案内する：

```
プロジェクトルートに .env ファイルを作成して、以下の1行を追加してください:

    OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

保存後「設定しました」と教えてください。
```

設定完了の報告を受けたら有効性を確認する：

```bash
.venv/Scripts/python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.cli_runner import run_check
result = run_check()
for item in result['items']:
    if item['label'] == 'OPENAI_API_KEY':
        icon = {'ok': '[OK]', 'warning': '[!!]', 'error': '[NG]'}[item['status']]
        print(f'{icon} {item[\"label\"]}: {item[\"detail\"]}')
"
```

`[NG]` の場合は `.env` の記載内容を再確認するよう案内して止まる。

---

### Step 2: リードCSV の配置 [必須]

以下の3択を提示する：

```
リードCSVはどこにありますか？

  A) data/leads.csv に配置済み（確認するだけ）
  B) 別の場所にある（パスを教えてください）
  C) サンプルデータを使いたい（テスト用）
```

**A を選んだ場合：**

```bash
.venv/Scripts/python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

path = Path('data/leads.csv')
if path.exists():
    count = sum(1 for _ in open(path, encoding='utf-8', errors='replace')) - 1
    print(f'[OK] data/leads.csv が見つかりました（{max(count,0)}件）')
else:
    print('[NG] data/leads.csv が見つかりません。ファイルを配置してください。')
"
```

**B を選んだ場合：**

パスを受け取り、確認してから cli_config.yaml を更新する：

```bash
.venv/Scripts/python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from src.cli_runner import load_cli_config, save_cli_config

csv_path = '<ユーザーが指定したパス>'
p = Path(csv_path)
if p.exists():
    count = sum(1 for _ in open(p, encoding='utf-8', errors='replace')) - 1
    print(f'[OK] {csv_path} が見つかりました（{max(count,0)}件）')
    cfg = load_cli_config()
    cfg['leads_csv_path'] = csv_path
    save_cli_config(cfg)
    print('cli_config.yaml を更新しました')
else:
    print(f'[NG] {csv_path} が見つかりません。パスを確認してください。')
"
```

**C を選んだ場合：**

```
data/test/ に以下のサンプルCSVがあります:

  - data/test/leads_japanese.csv  （日本語形式）
  - data/test/leads_english.csv   （英語形式）
  - data/test/leads_missing.csv   （欠損データのテスト用）

どれを使いますか？ 選択後、leads_csv_path として設定します。
```

CSV が確定したら簡易品質チェックを実行する：

```bash
.venv/Scripts/python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.config import Config
from src.utils import load_leads, auto_map_columns
from src.cli_runner import load_cli_config

cfg = load_cli_config()
df = load_leads(cfg.get('leads_csv_path', 'data/leads.csv'))
mapping = auto_map_columns(df.columns.tolist(), {**Config.REQUIRED_FIELDS, **Config.OPTIONAL_FIELDS})

print(f'総件数: {len(df)}')
print()
print('=== 必須フィールドマッピング ===')
for field in Config.REQUIRED_FIELDS:
    orig = mapping.get(field)
    status = '[OK]' if orig else '[NG]'
    print(f'  {status} {field} <- {orig}')
"
```

必須フィールド（`visitor_name` / `company_name` / `email`）に `[NG]` がある場合は CSV を修正するよう案内して止まる。

---

### Step 3: 送信者情報の設定 [必須]

現在の設定を表示する：

```bash
.venv/Scripts/python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.cli_runner import load_cli_config

cfg = load_cli_config()
company = cfg.get('sender_company', '')
name    = cfg.get('sender_name', '')
print('送信元会社名:', company if company else '（未設定）')
print('送信元担当者:', name    if name    else '（未設定）')
"
```

- 設定済みの場合：「前回の設定: {会社名} / {担当者名}。変更しますか？」と確認する
- 未設定の場合：「送信元会社名を教えてください」と入力を求める

入力を受けたら更新する：

```bash
.venv/Scripts/python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.cli_runner import load_cli_config, save_cli_config

cfg = load_cli_config()
cfg['sender_company'] = '<ユーザーが入力した会社名>'
cfg['sender_name']    = '<ユーザーが入力した担当者名>'
save_cli_config(cfg)
print('送信者情報を保存しました')
print('  会社名:', cfg['sender_company'])
print('  担当者:', cfg['sender_name'])
"
```

---

### Step 4: 展示会情報の設定 [推奨]

「展示会名・開催日・会場を入力してください。推奨ですが、後からでも設定できます。
スキップする場合は「s」と入力してください。」と確認する。

入力を受けた場合は cli_config.yaml に保存する：

```bash
.venv/Scripts/python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.cli_runner import load_cli_config, save_cli_config

cfg = load_cli_config()
cfg['exhibition_name']  = '<ユーザーが入力した展示会名>'
cfg['exhibition_date']  = '<ユーザーが入力した開催日>'
cfg['exhibition_venue'] = '<ユーザーが入力した会場>'
save_cli_config(cfg)
print('展示会情報を保存しました')
print('  展示会名:', cfg['exhibition_name'])
print('  開催日:  ', cfg['exhibition_date'])
print('  会場:    ', cfg['exhibition_venue'])
"
```

スキップした場合は「/email-workflow 実行時に入力できます」と案内して Step 5 へ進む。

---

### Step 5: 製品技術資料の確認・KB 構築 [必須]

現状を確認する：

```bash
.venv/Scripts/python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from src.config import Config

tech_dir = Path(Config.TECH_DOCS_DIR)
if tech_dir.exists():
    md_files  = list(tech_dir.glob('*.md'))
    pdf_files = list(tech_dir.glob('*.pdf'))
    if md_files:
        print(f'[OK] Markdown ファイルが {len(md_files)} 件あります:')
        for f in md_files:
            print(f'  - {f.name}')
    elif pdf_files:
        print(f'[!!] PDF ファイルが {len(pdf_files)} 件あります（直接読み込み不可）:')
        for f in pdf_files:
            print(f'  - {f.name}')
        print('→ Markdown に変換してから配置してください（例: DigiMA.pdf → DigiMA.md）')
    else:
        print('[!!] data/tech_documents/ が空です。')
        print('→ 製品技術資料を .md 形式で配置してください。')
else:
    print('[NG] data/tech_documents/ ディレクトリが見つかりません。')
"
```

**Markdown が1件以上ある場合：**

「ナレッジベースを構築します。Embedding API の費用が発生します（目安: 100KB の Markdown で約 ¥0.03）。
実行しますか？」と確認してから実行する：

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

**Markdown がない場合：**

ファイルを配置してから `/setup` の Step 5 を再実行するよう案内する。
技術資料がなくてもメール生成は可能だが、製品への言及がないメールになる旨を補足する。

---

### Step 6: 音声ファイルの確認 [任意]

「展示会当日の録音ファイルはありますか？（任意。なければスキップ可）」と確認する。

音声データなしでもメール生成は可能。音声がある場合はメールの精度が大幅に向上する。

音声ファイルがある場合はフォルダを確認する：

```bash
.venv/Scripts/python -c "
import sys, os
sys.stdout.reconfigure(encoding='utf-8')

audio_dir = '<ユーザーが指定したフォルダパス>'
exts = ('.m4a', '.mp3', '.wav')
try:
    files = sorted(f for f in os.listdir(audio_dir) if f.lower().endswith(exts))
    print(f'音声ファイル数: {len(files)}')
    for f in files:
        print(f'  {f}')
except FileNotFoundError:
    print(f'[NG] フォルダが見つかりません: {audio_dir}')
"
```

以下の命名規則を案内する：

```
推奨ファイル名: YYYYMMDD_担当者名_連番.m4a
例: 20260424_営業A_001.m4a
    20260424_営業B_002.mp3

・日付とタイムスタンプがあると自動紐づけ精度が向上します
・担当者名が含まれていないファイルは手動紐づけが必要になります
```

音声の紐づけ・文字起こしは `/audio-matching` スキルで実行できると案内する。

---

### Step 7: CRM 情報の確認 [任意]

「HubSpot・Salesforce 等の CRM データはありますか？（任意。なければスキップ可）」と確認する。

CRM データなしでもメール生成は可能。CRM があると過去商談の内容をメールに反映できる。

**CRM CSV がある場合：**

```bash
.venv/Scripts/python -c "
import sys, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
from src.config import Config
from src.utils import auto_map_columns

crm_path = '<ユーザーが指定した CRM CSV パス>'
try:
    df = pd.read_csv(crm_path, encoding='utf-8-sig')
    mapping = auto_map_columns(df.columns.tolist(), {**Config.CRM_REQUIRED_FIELDS, **Config.CRM_OPTIONAL_FIELDS})
    print(f'CRM CSV: {len(df)}件')
    for std, orig in mapping.items():
        status = '[OK]' if orig else '[--]'
        print(f'  {status} {std} <- {orig}')
except Exception as e:
    print(f'[NG] 読み込みエラー: {e}')
"
```

**CRM が Markdown 形式の場合：** `data/crm_records/` に配置するよう案内する。
CRM は `/match-records` スキルでさらに詳細な照合確認ができると補足する。

---

### Step 8: Gmail 下書き機能の設定 [推奨]

「生成したメールを Gmail の下書きに自動保存する機能を使いますか？」と確認する。

使わない場合はスキップ。`output/emails.csv` でメール内容を確認できる旨を案内する。

**使う場合 — 認証状態を確認する：**

```bash
.venv/Scripts/python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

creds = Path('credentials/credentials.json')
token = Path('credentials/token.json')

if creds.exists():
    print('[OK] credentials.json が見つかりました')
    if token.exists():
        print('[OK] token.json も存在します（認証済み）')
    else:
        print('[!!] token.json がありません（/email-workflow 初回実行時にブラウザ認証が必要です）')
else:
    print('[NG] credentials/credentials.json が見つかりません')
    print('  Google Cloud Console で OAuth クライアントを作成してください（手順は下記）')
"
```

**credentials.json がない場合：** 以下の手順を案内する：

```
Google Cloud Console での OAuth クライアント作成手順:

1. https://console.cloud.google.com/ にアクセス
2. プロジェクトを作成（または既存を選択）
3. 「APIとサービス」→「ライブラリ」→「Gmail API」を有効化
4. 「認証情報」→「認証情報を作成」→「OAuth 2.0 クライアント ID」
5. アプリケーションの種類: 「デスクトップ アプリ」を選択
6. 作成後「JSON をダウンロード」
7. ダウンロードした JSON を credentials/credentials.json に配置

配置後、/email-workflow の Step 7 で初回のブラウザ認証が実行されます。
```

---

### Step 9: メール対象ランクの設定 [必須]

現在の設定を表示する：

```bash
.venv/Scripts/python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.cli_runner import load_cli_config

cfg = load_cli_config()
ranks = cfg.get('default_ranks', ['A', 'B', 'C'])
print('現在の対象ランク:', ranks)
"
```

- 設定済みの場合：「前回の設定: {ranks}。変更しますか？」と確認する
- 変更する場合は以下の選択肢を提示する：

```
メール生成の対象ランクを選んでください（A が最上位）:

  1) A・B・C（デフォルト推奨）
  2) A・B のみ（ホットリードのみ）
  3) A のみ
  4) A・B・C・D・E（全件）
  5) カスタム指定（例: A,C）
```

選択を受けたら更新する：

```bash
.venv/Scripts/python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.cli_runner import load_cli_config, save_cli_config

cfg = load_cli_config()
cfg['default_ranks'] = <ユーザーが選択したランクリスト>  # 例: ['A', 'B', 'C']
save_cli_config(cfg)
print('対象ランクを更新しました:', cfg['default_ranks'])
"
```

---

### 完了報告

すべてのステップが完了したら設定サマリーを表示する：

```bash
.venv/Scripts/python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.cli_runner import run_check, load_cli_config

cfg   = load_cli_config()
check = run_check()

def _find(label):
    for item in check['items']:
        if item['label'] == label:
            return item['detail']
    return '不明'

print('━━━ セットアップ完了サマリー ━━━')
print()
print('■ APIキー:       ' + _find('OPENAI_API_KEY'))
print('■ リードCSV:     ' + cfg.get('leads_csv_path', '未設定'))
print('■ 送信元会社名:  ' + (cfg.get('sender_company') or '（未設定）'))
print('■ 送信元担当者:  ' + (cfg.get('sender_name') or '（未設定）'))
print('■ 展示会名:      ' + (cfg.get('exhibition_name') or '（未設定）'))
print('■ 開催日:        ' + (cfg.get('exhibition_date') or '（未設定）'))
print('■ 会場:          ' + (cfg.get('exhibition_venue') or '（未設定）'))
print('■ 対象ランク:    ' + str(cfg.get('default_ranks', [])))
print('■ KB 状態:       ' + _find('ナレッジベース (KB)'))
print('■ Gmail:         ' + _find('Gmail credentials'))
print()
print('設定は cli_config.yaml に保存されています。')
"
```

最後に次のアクションを案内する：

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
セットアップが完了しました。

■ 次のステップ:
  → /email-workflow を実行してメール生成を開始できます

■ 詳細確認が必要な場合:
  → /inspect-data    — リードCSVのデータ品質を詳細確認
  → /csv-mapping     — カラムマッピングを詳細確認
  → /audio-matching  — 音声ファイルの紐づけ（音声データがある場合）
  → /match-records   — CRM とリードの照合（CRM データがある場合）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 注意事項

- APIキーは対話では入力しないこと（セキュリティ上の理由: 会話ログに残るため）
- ナレッジベース構築（Step 5）は Embedding API の費用が発生する（目安: 100KB あたり約 ¥0.03）
- Step 4・6・7 はスキップ可能（後から `/setup` を再実行して設定できる）
- 設定はすべて `cli_config.yaml` に保存される。テキストエディタで直接編集することも可能
