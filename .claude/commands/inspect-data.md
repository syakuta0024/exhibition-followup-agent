# inspect-data — データ品質確認スキル

リードCSV（および任意でCRM CSV）を読み込み、**カラムの意味・マッピング状況・データの過不足**を
わかりやすく整理して報告する。実際のメール生成前の「データ確認」フェーズで使う。

---

## 実行フロー

### Step 0: 対象ファイルの確認

1. `cli_config.yaml` の `leads_csv_path` を読む。なければ `data/leads.csv`。
2. 引数でパスが渡された場合（例: `/inspect-data data/my_leads.csv`）はそのまま使う。
3. CRM CSV があれば「CRM CSVのパスはありますか？」と確認する（任意）。

---

### Step 1: CSVを読み込んでカラム一覧を取得

```bash
.venv/Scripts/python -c "
import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')
from src.config import Config
from src.utils import load_leads, auto_map_columns, apply_column_mapping

from src.cli_runner import load_cli_config
cfg = load_cli_config()
df = load_leads(cfg.get('leads_csv_path', 'data/leads.csv'))
mapping = auto_map_columns(df.columns.tolist(), {**Config.REQUIRED_FIELDS, **Config.OPTIONAL_FIELDS})
df_mapped = apply_column_mapping(df, mapping)

print('=== 元カラム一覧 ===')
for col in df.columns:
    print(f'  {col}')

print()
print('=== マッピング結果 ===')
print('REQUIRED:')
for std, orig in mapping.items():
    if std in Config.REQUIRED_FIELDS:
        status = '[OK]' if orig else '[NG]'
        print(f'  {status} {std} <- {orig}')
print('OPTIONAL:')
for std, orig in mapping.items():
    if std in Config.OPTIONAL_FIELDS:
        status = '[OK]' if orig else '[--]'
        print(f'  {status} {std} <- {orig}')

# extra_ カラムを検出
extra_cols = [c for c in df_mapped.columns if c.startswith('extra_') and c != 'extra_lead_id']
if extra_cols:
    print()
    print('=== カスタム質問列（extra_）===')
    for c in extra_cols:
        print(f'  {c}')

print()
print('件数:', len(df))
"
```

---

### Step 2: カラムマッピング解説を表示

取得結果をもとに、以下の形式で表を作って説明する:

```
=== カラムマッピング結果 ===

【必須フィールド】
  状態  標準名           元カラム名       説明
  ✅    visitor_name     氏名             来場者の氏名（姓名1列）
  ✅    company_name     会社名           所属企業名
  ✅    email            メールアドレス   連絡先メール
  ❌    （未マッピング） —                → 手動確認が必要

【任意フィールド】（あれば品質向上）
  状態  標準名              元カラム名
  ✅    lead_rank           評価
  ✅    memo                メモ
  —     department          未検出
  —     scan_time           未検出（音声紐づけに使用）
  —     rep_name            未検出（音声紐づけに使用）

【カスタム質問列】（extra_ 付きで自動保持）
  extra_future_requests   今後のご要望
  extra_interest          関心分野
```

`scan_time` と `rep_name` が未検出の場合は「音声ファイルの自動紐づけができません」と補足する。

---

### Step 3: データ品質チェック

```bash
.venv/Scripts/python -c "
import pandas as pd, sys, re
sys.stdout.reconfigure(encoding='utf-8')
from src.config import Config
from src.utils import load_leads, auto_map_columns, apply_column_mapping

from src.cli_runner import load_cli_config
cfg = load_cli_config()
df = load_leads(cfg.get('leads_csv_path', 'data/leads.csv'))
mapping = auto_map_columns(df.columns.tolist(), {**Config.REQUIRED_FIELDS, **Config.OPTIONAL_FIELDS})
df = apply_column_mapping(df, mapping)

total = len(df)
print(f'総件数: {total}')
print()

# 必須フィールドの欠損チェック
print('=== 必須フィールド 欠損チェック ===')
for field in ['visitor_name', 'company_name', 'email']:
    if field in df.columns:
        missing = df[field].isna().sum() + (df[field].astype(str).str.strip() == '').sum()
        pct = missing / total * 100
        flag = '[NG]' if missing > 0 else '[OK]'
        print(f'  {flag} {field}: {missing}件 欠損 ({pct:.0f}%)')
    else:
        print(f'  [NG] {field}: カラム未検出')

# メールアドレス形式チェック
print()
print('=== メールアドレス 形式チェック ===')
if 'email' in df.columns:
    email_pattern = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')
    invalid = df[df['email'].notna() & (df['email'].astype(str).str.strip() != '')].copy()
    invalid = invalid[~invalid['email'].astype(str).apply(lambda x: bool(email_pattern.match(x.strip())))]
    print(f'  形式不正: {len(invalid)}件')
    if len(invalid) > 0:
        for _, row in invalid.iterrows():
            print(f'    {row.get(\"visitor_name\",\"\")} ({row.get(\"company_name\",\"\")}) -> {row.get(\"email\",\"\")}')

# ランク分布チェック
print()
print('=== ランク分布 ===')
if 'lead_rank' in df.columns:
    rank_counts = df['lead_rank'].value_counts(dropna=False)
    for rank, cnt in rank_counts.items():
        print(f'  {rank}: {cnt}件')
    # 未設定チェック
    unknown = df[df['lead_rank'].isna() | (df['lead_rank'].astype(str).str.strip() == '')].shape[0]
    if unknown > 0:
        print(f'  [!!] ランク未設定: {unknown}件 → LLM推定または手動設定を推奨')
else:
    print('  [!!] lead_rank カラム未検出 → 全件 LLM推定になります')

# 任意フィールドの充足率
print()
print('=== 任意フィールド 充足率 ===')
for field in ['department', 'job_title', 'memo', 'interested_products', 'future_requests', 'scan_time', 'rep_name']:
    if field in df.columns:
        filled = df[field].notna().sum() - (df[field].astype(str).str.strip() == '').sum()
        pct = max(filled, 0) / total * 100
        print(f'  {field}: {max(filled,0)}/{total}件 ({pct:.0f}%)')
    else:
        print(f'  {field}: 未検出 (--)')
"
```

---

### Step 4: 品質レポートを表示

チェック結果を整理して以下のような形式で出力する:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
データ品質レポート  data/leads.csv  （全18件）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【必須フィールド】
  ✅ 氏名 (visitor_name)        全18件 入力済み
  ✅ 会社名 (company_name)      全18件 入力済み
  ✅ メール (email)             全18件 入力済み / 形式不正 0件

【ランク分布】
  A: 3件  B: 7件  C: 5件  D: 2件  未設定: 1件
  ⚠️ ランク未設定が1件あります。LLM推定が有効な場合は自動補完されます。

【任意フィールド 充足率】
  ✅ 部署 (department)        18/18件 (100%)
  ✅ 役職 (job_title)         18/18件 (100%)
  ✅ メモ (memo)              15/18件 ( 83%)
  ✅ 関心製品 (interested_products) 12/18件 ( 67%)
  ⚠️ スキャン時刻 (scan_time)  0/18件 (  0%) → 音声紐づけ不可
  ⚠️ 担当営業 (rep_name)       0/18件 (  0%) → 音声紐づけ不可

【カスタム質問列】
  ✅ extra_future_requests    今後のご要望（全件保持済み）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
総合判定: メール生成可能 ✅
  推奨アクション:
  1. メモ未入力3件（L009・L014・L016）は内容が薄いメールになります
  2. scan_time・rep_name がないため音声紐づけは手動対応が必要です
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

総合判定の基準:
- 必須フィールドが全件揃っている → `メール生成可能 ✅`
- 必須フィールドに欠損あり → `メール生成前に修正が必要 ❌`
- 必須フィールドは揃っているが品質懸念あり → `メール生成可能（品質注意） ⚠️`

---

### Step 5: CRM CSV の確認（任意）

ユーザーが CRM CSV を指定した場合、同様に読み込んで:

```bash
.venv/Scripts/python -c "
import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')
from src.config import Config
from src.utils import auto_map_columns, apply_column_mapping

df = pd.read_csv('<Step5で確認したCRM CSVパス>', encoding='utf-8-sig')
mapping = auto_map_columns(df.columns.tolist(), {**Config.CRM_REQUIRED_FIELDS, **Config.CRM_OPTIONAL_FIELDS})

print('=== CRM カラムマッピング ===')
for std, orig in mapping.items():
    if std in Config.CRM_REQUIRED_FIELDS:
        status = '[OK]' if orig else '[NG]'
        print(f'  {status} {std} <- {orig}')
    else:
        status = '[OK]' if orig else '[--]'
        print(f'  {status} {std} <- {orig}')
print()
print('件数:', len(df))
"
```

CRM の必須フィールド（`email` または `company_name`）が検出されない場合は「紐づけに使えません」と案内する。

---

## 注意事項

- このスキルは**読み取りのみ**で、ファイルの変更は行わない
- 必須フィールドが欠損している場合は生成前に CSV を修正するよう案内する
- scan_time・rep_name が空でもメール生成自体は可能（音声機能が使えないだけ）
- 品質確認が完了したら `/match-records` でCRM紐づけ、または `/email-workflow` でメール生成へ進むよう案内する
