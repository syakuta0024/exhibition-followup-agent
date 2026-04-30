# match-records — データ紐づけ確認スキル

リードCSVと CRM CSV を横断的に紐づけ、**曖昧なケースだけ**ユーザーに確認を求める。
明らかな一致は自動確定し、スキップする。

音声ファイルの紐づけは `/audio-matching` を使ってください。

---

## 実行フロー

### Step 0: データソースの確認

まずどのソースを紐づけるか確認する。

1. **リードCSV**（必須）: `cli_config.yaml` の `leads_csv_path` を読む。なければ `data/leads.csv`。
2. **CRM CSV**（任意）: ユーザーに「CRM CSVはありますか？あれば場所を教えてください」と確認。

引数でパスが渡された場合（例: `/match-records data/crm.csv`）はそのまま使う。

---

### Step 1: 各データソースを読み込む

```bash
# リードCSV
.venv/Scripts/python -c "
import pandas as pd
from src.config import Config
from src.utils import load_leads, auto_map_columns, apply_column_mapping
from src.cli_runner import load_cli_config
cfg = load_cli_config()

df = load_leads(cfg.get('leads_csv_path', 'data/leads.csv'))
mapping = auto_map_columns(df.columns.tolist(), {**Config.REQUIRED_FIELDS, **Config.OPTIONAL_FIELDS})
df = apply_column_mapping(df, mapping)

print('=== LEADS: カラムとサンプル ===')
key_cols = [c for c in ['lead_id','visitor_name','company_name','email','scan_time','rep_name'] if c in df.columns]
print(df[key_cols].to_string(index=True))
"
```

CRM CSVも同様に読み込む（パスはユーザーから取得）。

---

### Step 2: カラムマッピングの解説

各ソース間で **同じ情報を指すが名前が違うカラム** を一覧にして説明する：

```
| 情報       | リードCSV       | CRM CSV         |
|-----------|----------------|----------------|
| 氏名       | visitor_name   | name / 氏名     |
| 会社名     | company_name   | company / 会社名 |
| メール     | email          | email / Email   |
| 担当営業   | rep_name       | contact_owner  |
```

実際のカラム名は読み込んだデータに基づいて修正する。

---

### Step 3: マッチング実行（リード ↔ CRM）

```bash
.venv/Scripts/python -c "
import sys, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
from src.config import Config
from src.utils import load_leads, auto_map_columns, apply_column_mapping
from src.crm_matcher import CRMMatcher

# リードCSV
from src.cli_runner import load_cli_config
cfg = load_cli_config()
df = load_leads(cfg.get('leads_csv_path', 'data/leads.csv'))
mapping = auto_map_columns(df.columns.tolist(), {**Config.REQUIRED_FIELDS, **Config.OPTIONAL_FIELDS})
df = apply_column_mapping(df, mapping)

# CRM CSV（パスはユーザーから取得）
crm_df = pd.read_csv('data/crm.csv', encoding='utf-8-sig')
crm_mapping = auto_map_columns(crm_df.columns.tolist(), {**Config.CRM_REQUIRED_FIELDS, **Config.CRM_OPTIONAL_FIELDS})
crm_df = apply_column_mapping(crm_df, crm_mapping)

matcher = CRMMatcher()

print('=== マッチング結果 ===')
auto_matched = []
need_confirm = []
unmatched = []

for _, lead_row in df.iterrows():
    lead = lead_row.to_dict()
    match = matcher.match(lead, crm_df)
    if match is None:
        unmatched.append(lead)
    elif match.get('_crm_match_score', 0) == 100:
        auto_matched.append((lead, match))
    else:
        need_confirm.append((lead, match))

print(f'自動確定: {len(auto_matched)}件 / 要確認: {len(need_confirm)}件 / 未紐づけ: {len(unmatched)}件')
for lead, match in auto_matched:
    print(f'  [確定] {lead[\"visitor_name\"]} <- CRM: {match.get(\"visitor_name\",\"\")} [{match[\"_crm_match_method\"]}]')
"
```

優先順位：

| 優先順 | 条件 | 判定 |
|---|---|---|
| 1 | メールアドレスが完全一致 | 自動確定（score=100） |
| 2 | 会社名コアが一致 + 姓が一致 | 自動確定 |
| 3 | 姓名が似ているが確信が持てない | **要確認** |
| 4 | 候補なし | 未紐づけ |

---

### Step 4: 名前の読みの判定ルール（LLMとして判断する）

以下を参考に、漢字とひらがな/カタカナの対応を推論する:

- `田中 誠` vs `田中 まこと` → **誠＝まこと** → 自動確定
- `山田 健` vs `山田 けん` → **健＝けん** → 自動確定
- `鈴木 和夫` vs `鈴木 かずお` → **和夫＝かずお** → 自動確定
- `田中 健` vs `田中 たかし` → **健はケン/タケシ両方あり得る** → 要確認
- `佐藤 実` vs `佐藤 みのる` → **実＝みのる** → 自動確定

**判断基準**: その読みが唯一である、または非常に一般的な場合のみ自動確定。
複数の読みがある漢字は要確認にする。

---

### Step 5: 要確認ケースの提示方法

1件ずつ以下の形式で提示し、ユーザーの回答を待つ:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[要確認 2/5]

リード:  田中 健 (株式会社山田製作所)
CRM:     田中 たかし (ヤマダ製作所)   email: t.tanaka@yamada.co.jp

類似度: 姓一致 / 名は「健」→「たかし」複数の読みあり / 会社名ほぼ一致

→ この2件は同一人物ですか？
  [y] 同一人物として確定
  [n] 別人
  [s] スキップ（後で確認）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

ユーザーが `y/n/s` を入力したら次のケースへ進む。
途中で止めたい場合は `done` または `q` で終了する。

---

### Step 6: 結果サマリーの表示

全ケース処理後（または `done` 入力後）、以下の形式で結果をまとめる:

```
━━━ 紐づけ結果サマリー ━━━

■ 自動確定 (8件)
  - 山田 太郎 (ABC製造) ← CRM: 山田 太郎 (ABC製造株式会社)  [メール一致]
  ...

■ ユーザー確定 (3件)
  - 鈴木 健  (東京機器) ← CRM: 鈴木 たけし (東京機器)  [y]
  ...

■ 別人と判断 (1件)
  - 佐藤 誠  (大阪製作) ← CRM候補: 佐藤 誠 (大阪製鉄)  [n]

■ 未紐づけ (2件)
  - 伊藤 花子  (千葉電機)  → CRMに対応候補なし

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
合計: 全15件 → 確定11件 / 未紐づけ2件 / 別人1件 / スキップ1件

━━━ 次のアクション ━━━
  → /email-workflow でメール生成へ進めます
  → /audio-matching で音声ファイルの紐づけを行えます
```

未紐づけがある場合は「手動で確認してください」と伝える。

---

## 注意事項

- `output/emails.csv` がある場合は既に処理済みの可能性があることをユーザーに伝える
- このスキルは**確認のみ**で、実際のデータ変更は行わない
- 音声ファイルの紐づけは `/audio-matching` を使ってください
