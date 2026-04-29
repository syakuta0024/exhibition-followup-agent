# csv-mapping — カラムマッピング確認スキル

リードCSVとCRM CSVのカラムを自動推定し、名前の表記ゆれをLLMが検出して
**ユーザーに1件ずつ確認**する。確認が完了したら次のステップへ案内する。

---

## 実行フロー

### Step 0: 対象ファイルの確認

1. `cli_config.yaml` の `leads_csv_path` を読む。なければ `data/leads.csv`。
2. 引数でパスが渡された場合（例: `/csv-mapping data/my_leads.csv`）はそのまま使う。
3. 「CRM CSVはありますか？あれば場所を教えてください（任意）」と確認する。

---

### Step 1: リードCSVのカラムマッピングを確認

```bash
.venv/Scripts/python -c "
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
from src.config import Config
from src.utils import load_leads, auto_map_columns, apply_column_mapping

df = load_leads('data/leads.csv')
all_fields = {**Config.REQUIRED_FIELDS, **Config.OPTIONAL_FIELDS}
mapping = auto_map_columns(df.columns.tolist(), all_fields)

print('=== 元カラム一覧 ===')
for col in df.columns:
    print(f'  {col}')
print()
print('=== 自動マッピング結果 ===')
for std, orig in mapping.items():
    if std in Config.REQUIRED_FIELDS:
        status = '[必須][OK]' if orig else '[必須][NG]'
    else:
        status = '[任意][OK]' if orig else '[任意][--]'
    print(f'  {status} {std} <- {orig}')
"
```

結果を以下の形式で整理して表示する：

```
=== リードCSV カラムマッピング ===

【必須フィールド】
  ✅ visitor_name  ← 「氏名」
  ✅ company_name  ← 「会社名」
  ✅ email         ← 「メールアドレス」

【任意フィールド（品質向上に効果あり）】
  ✅ lead_rank        ← 「評価」
  ✅ memo             ← 「メモ」
  ✅ scan_time        ← 「スキャン時刻」 （音声紐づけに使用）
  ✅ rep_name         ← 「担当営業」     （音声紐づけに使用）
  — department        未検出
  — interested_products 未検出
```

マッピングが間違っていると思われる箇所は「このマッピングは正しいですか？」と確認する。
必須フィールドが未マッピングの場合は「メール生成前に CSV を修正してください」と案内する。

---

### Step 2: 名前の表記ゆれ検出

```bash
.venv/Scripts/python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.config import Config
from src.utils import load_leads, auto_map_columns, apply_column_mapping

df = load_leads('data/leads.csv')
mapping = auto_map_columns(df.columns.tolist(), {**Config.REQUIRED_FIELDS, **Config.OPTIONAL_FIELDS})
df = apply_column_mapping(df, mapping)

if 'visitor_name' in df.columns:
    names = df[['lead_id', 'visitor_name', 'company_name']].dropna(subset=['visitor_name'])
    print('=== 氏名サンプル（全件）===')
    for _, row in names.iterrows():
        print(f'  {row[\"lead_id\"]}  {row[\"visitor_name\"]}  ({row[\"company_name\"]})')
"
```

取得した氏名リストをLLMとして以下の観点で分析する：
- **漢字 ↔ ひらがな/カタカナ**：「田中 誠」と「田中 まこと」
- **苗字のみ ↔ フルネーム**：「鈴木」と「鈴木 一郎」
- **旧字体/新字体**：「斎藤」と「齋藤」
- **スペース有無**：「山田太郎」と「山田 太郎」

表記ゆれの候補ペアが見つかった場合は1件ずつ以下の形式で確認する：

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[表記ゆれ確認 1/3]

候補A: L003  田中 誠  (ABC製造株式会社)
候補B: L011  田中 まこと  (ABC製造)

→ この2件は同一人物ですか？
  [y] 同一人物（L003 を代表とする）
  [n] 別人
  [s] スキップ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

同一人物と判定した場合はメモとして記録する（実際のCSV変更は行わない）。

---

### Step 3: CRM CSVのマッピング確認（任意）

CRM CSVがある場合：

```bash
.venv/Scripts/python -c "
import sys, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
from src.config import Config
from src.utils import auto_map_columns

df = pd.read_csv('data/crm.csv', encoding='utf-8-sig')  # パスはユーザーから取得
mapping = auto_map_columns(df.columns.tolist(), {**Config.CRM_REQUIRED_FIELDS, **Config.CRM_OPTIONAL_FIELDS})

print('=== CRM カラムマッピング ===')
for std, orig in mapping.items():
    flag = '[必須]' if std in Config.CRM_REQUIRED_FIELDS else '[任意]'
    status = '[OK]' if orig else '[--]'
    print(f'  {flag}{status} {std} <- {orig}')
print('件数:', len(df))
"
```

CRM の必須フィールド（`email` または `company_name`）が検出されない場合は「紐づけには使えません」と案内する。

---

### Step 4: 結果サマリー

```
━━━ カラムマッピング確認 完了 ━━━

■ リードCSV
  必須フィールド: 全3件 マッピング済み ✅
  任意フィールド: 5/12件 マッピング済み

■ 表記ゆれ
  候補 3ペア → 同一人物 2件 / 別人 1件 / スキップ 0件

■ CRM CSV
  マッピング済み ✅（email一致で紐づけ可能）

━━━ 次のアクション ━━━
  → /email-workflow でメール生成フロー全体を開始できます
  → /inspect-data でデータ品質の詳細を確認できます
```

---

## 注意事項

- このスキルは**確認のみ**で、実際のCSVファイルの変更は行わない
- 表記ゆれの同一人物判定結果はメモとして会話内に保持する（永続化しない）
- 必須フィールドが未マッピングの場合はメール生成をブロックするよう案内する
