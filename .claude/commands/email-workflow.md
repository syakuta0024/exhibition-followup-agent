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

sender_name = cfg.get('sender_name', '')
if sender_name:
    print(f'■ 送信元担当者: {sender_name}（保存済み）')
else:
    print('■ 送信元担当者: 未設定 → /email-workflow 内で設定してください（未設定の場合「●●」になります）')

leads_path = cfg.get('leads_csv_path', 'data/leads.csv')
label = 'カスタム' if leads_path != 'data/leads.csv' else 'デフォルト'
print(f'■ リードCSV:    {leads_path}（{label}）')

crm_path = cfg.get('crm_csv_path', '')
if crm_path:
    from pathlib import Path
    exists = Path(crm_path).exists()
    state = '存在' if exists else '※ ファイル未発見'
    print(f'■ CRM CSV:      {crm_path}（{state}）')
else:
    print('■ CRM CSV:      未設定（vectordb の Markdown CRM 検索にフォールバック）')

from pathlib import Path as _P
audio_ctx = _P('output/audio_context.json')
if audio_ctx.exists():
    print(f'■ 音声コンテキスト: {audio_ctx}（読み込み対象）')
else:
    print('■ 音声コンテキスト: なし（/audio-matching を実行すると自動生成されます）')

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

まず前回の処理プロファイルが存在するか確認する:

```bash
.venv/Scripts/python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.cli_runner import load_last_run_profile
profile = load_last_run_profile()
if profile:
    print('profile_found: true')
    print(f'exhibition_name: {profile.get(\"exhibition_name\", \"\")}')
    print(f'exhibition_date: {profile.get(\"exhibition_date\", \"\")}')
    print(f'exhibition_venue: {profile.get(\"exhibition_venue\", \"\")}')
    print(f'ranks: {profile.get(\"ranks\", [])}')
    print(f'schedule_policy: {profile.get(\"schedule_policy\", \"\")}')
    print(f'saved_at: {profile.get(\"saved_at\", \"\")}')
else:
    print('profile_found: false')
"
```

**profile_found が true の場合**:

```
前回の設定が見つかりました:

  展示会名: {exhibition_name}
  開催日:   {exhibition_date}
  会場:     {exhibition_venue}
  対象ランク: {ranks}
  候補日ポリシー: {schedule_policy}
  保存日時: {saved_at}

この設定を引き継ぎますか？

[1] 引き継ぐ（候補日だけ再入力）
[2] 最初から入力する
```

- **[1] を選んだ場合**: プロファイルの展示会名・ランク・ポリシーをそのまま使用し、Step 5.5 の候補日入力（質問3）のみ行ってから Step 6 へ進む（質問1・質問2はスキップ）
- **[2] を選んだ場合**: 通常の入力フロー（下記）へ進む

**profile_found が false の場合（またはユーザーが [2] を選んだ場合）**:

「展示会名・開催日・会場を教えてください（任意ですが、メールの品質が向上します）」と確認する。

例：
- 展示会名: 「製造業 DX 展 2026」
- 開催日: 「2026年4月24日〜26日」
- 会場: 「東京ビッグサイト」

---

### Step 5.5: 対象ランクと候補日の収集

#### 質問1: メール送信対象のランク

以下の選択肢を提示し、選んでもらう:

| 選択肢 | ランク | target_ranks の値 |
|---|---|---|
| A) A・B・Cランク全員（デフォルト） | A / B / C | ["A", "B", "C"] |
| B) A・Bランクのみ（ホットリードのみ） | A / B | ["A", "B"] |
| C) Aランクのみ | A | ["A"] |
| D) 全ランク（D・Eも含む） | A〜E | ["A", "B", "C", "D", "E"] |
| E) カスタム指定（例: "A,C"） | 任意 | ユーザー入力をカンマ分割 |

→ `target_ranks: List[str]` として確定する

#### 質問2: 候補日提示ポリシー

以下を提示して選んでもらう:

| 選択肢 | 内容 | schedule_policy の値 |
|---|---|---|
| A) A/Bのみ候補日提示、Cは情報提供型（業界推奨） | ホットリードに絞る | "ab_only" |
| B) 全ランクで候補日を提示 | 全員に日程提案 | "all" |
| C) 全ランクで候補日を提示しない | 日程提案なし | "none" |

→ `schedule_policy: str` として確定する

`"none"` を選んだ場合は **質問3をスキップして Step 6 へ進む**（`candidate_dates = None`）。

#### 質問3: 候補日の入力（Q2 が "all" または "ab_only" の場合のみ）

まず cli_config.yaml の calendar.enabled を確認し、Google Calendar 連携が有効な場合は自動取得を優先して案内する:

```bash
.venv/Scripts/python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.cli_runner import load_cli_config
cfg = load_cli_config()
cal = cfg.get('calendar', {})
print('calendar_enabled:', cal.get('enabled', True))
"
```

**calendar_enabled が true の場合：** 以下の選択肢を提示する:

```
候補日の入力方法を選んでください:
  [1] Google Calendar から空き時間を自動取得（推奨）
  [2] 手動で入力する
```

**[1] を選択した場合（Google Calendar 自動取得）:**

```bash
.venv/Scripts/python -c "
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
from src.cli_runner import run_fetch_calendar_slots
result = run_fetch_calendar_slots()
if result['error']:
    print('ERROR:', result['error'])
else:
    print(f'空き枠が {len(result[\"slots\"])} 件見つかりました:')
    for i, s in enumerate(result['slots'], 1):
        print(f'  {i}. {s[\"display\"]}')
    print()
    print('formatted:', result['formatted'])
"
```

取得成功時（slots が1件以上）の場合:

```
以下の空き枠が見つかりました:
  1. 5月20日（水）14:00〜15:00
  2. 5月21日（木）10:00〜11:00
  3. 5月22日（金）13:00〜14:00
  4. 5月26日（月）15:00〜16:00
  5. 5月27日（火）11:00〜12:00

使用する枠を選んでください（例: 1,2,3 / すべて使う場合は Enter）:
```

ユーザーが番号を選択したら、選択された枠の `display` 値を結合してカスタム候補日テキストを組み立て、
validate_candidate_dates() に渡せる形式（`YYYY/M/D HH:MM-HH:MM`）に変換してから検証する。
変換できない場合は手動入力フローへフォールバックする。

slots が0件の場合:

```
指定期間内に空き枠が見つかりませんでした。手動入力に切り替えます。
```

→ [2] 手動入力フローへ進む。

エラーが発生した場合（Google Calendar API 未有効化など）:

```
Google Calendar への接続でエラーが発生しました:
  （result["error"] の内容）

手動入力に切り替えます。
```

→ [2] 手動入力フローへ進む。

**[2] を選択した場合（手動入力）または calendar_enabled が false の場合:**

以下のフォーマットで候補日を入力してもらう（改行区切り、時間帯はカンマ区切り）:

```
2026/5/8 10:00-12:00, 14:00-17:00
2026/5/12 9:00-12:00
2026/5/13 13:00-17:00
```

入力を受けたら以下のコードで検証する:

```bash
.venv/Scripts/python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.cli_runner import validate_candidate_dates

user_input = '''2026/5/8 10:00-12:00, 14:00-17:00
2026/5/12 9:00-12:00
2026/5/13 13:00-17:00'''  # ← ユーザー入力に置き換える

result = validate_candidate_dates(user_input)
print('is_valid:', result['is_valid'])
if result['errors']:
    for e in result['errors']:
        print('エラー:', e)
else:
    print('件数:', len(result['parsed']))
"
```

**検証結果の処理**:

| 結果 | 対応 |
|---|---|
| `is_valid = True` | `result["parsed"]` を `candidate_dates` として確定し、Step 6 へ進む |
| `is_valid = False` | エラー内容を表示し、**3つすべてを再入力**してもらう（部分修正不可） |

**エラー時の案内文（必ず以下の形式で伝える）**:

```
以下のエラーがありました:
  ・（result["errors"] の内容）

候補日を3つすべて入力し直してください。
書式例: 2026/5/8 10:00-12:00, 14:00-17:00
※ 過去の日付が1件でも含まれると全部やり直しになります。
```

#### Step 5.5 完了時の確定変数

| 変数 | 型 | 値の例 |
|---|---|---|
| `target_ranks` | `List[str]` | `["A", "B", "C"]` |
| `schedule_policy` | `str` | `"ab_only"` |
| `candidate_dates` | `List[Dict]` または `None` | `[{"date": "2026/5/8", "time_slots": ["10:00-12:00"]}]` |

---

### Step 6: メール生成

Step 0 の設定、Step 5 の展示会情報、Step 5.5 の収集値をもとにメールを一括生成する。
送信元会社名・CSV パス等は `cli_config.yaml` から自動参照する：

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
    # Step 5.5 で収集した値
    ranks=[],             # ← Step 5.5 の target_ranks（例: ['A', 'B', 'C']）
    schedule_policy='',   # ← Step 5.5 のポリシー（例: 'ab_only'）
    candidate_dates=[],   # ← Step 5.5 の候補日リスト（"none" 時は None）
    # sender_company / csv_path 等は cli_config.yaml から自動参照
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
事前に `/audio-matching` を実行しておくと、結果が `output/audio_context.json` に
保存され、`/email-workflow`（`run_generate()`）が自動的に読み込んでメール生成の
最優先コンテキストとして反映する。手動で引数を渡す必要はない。

`lead_key` は `src/cli_runner.build_lead_key(lead)` で算出される（lead_id 優先、
無ければ `visitor_name_company_name` の複合キー）。`/audio-matching` 側の保存
スクリプトも同じ関数を使うため、両者の紐づけは自動的に整合する。

### CRM CSV を併用する
`cli_config.yaml` の `crm_csv_path` に CRM CSV のパスを設定すると、
`run_generate()` が起動時に `load_crm_csv()` で読み込み、各リードに対して
メール完全一致 + 社名ファジーマッチ（rapidfuzz）で紐づける。
未設定（空文字）の場合は `data/crm_records/*.md` を対象にした vectordb 検索に
フォールバックする。

---

## 注意事項

- KB が未構築の状態で generate を実行しない
- プロジェクトルートから実行すること（相対パスが壊れる）
- Gmail 下書き機能は `credentials/credentials.json` が必要（詳細は README 参照）
