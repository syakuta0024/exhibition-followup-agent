# rank-mapping — ランク値正規化スキル

リードCSVの商談確度フィールドのユニーク値を LLM が A〜E にマッピングし、
ユーザーに確認後 `cli_config.yaml` へ保存して次回から自動適用する。

---

## 実行フロー

### Step 0: 準備

1. `cli_config.yaml` の `leads_csv_path` と `rank_value_mapping` を読む。
2. `rank_value_mapping` がすでに存在する場合は「保存済みのマッピングがあります」と表示し、
   再実行するか確認する（上書きしたい場合のみ続行）。

```bash
.venv/Scripts/python -c "
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
from src.cli_runner import load_cli_config
cfg = load_cli_config()
mapping = cfg.get('rank_value_mapping', {})
if mapping:
    print('=== 保存済みのランク値マッピング ===')
    for k, v in mapping.items():
        print(f'  {k!r} -> {v}')
else:
    print('(未保存)')
"
```

---

### Step 1: ランク値を分析

`cli_config.yaml` の `leads_csv_path` を使いランクフィールドのユニーク値を取得し、
LLM が A〜E へのマッピングを推定する。

```bash
.venv/Scripts/python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.config import Config
from src.cli_runner import load_cli_config, run_rank_mapping
from langchain_openai import ChatOpenAI

cfg = load_cli_config()
client = ChatOpenAI(model=Config.LLM_MODEL, temperature=0, openai_api_key=Config.OPENAI_API_KEY)
result = run_rank_mapping(
    leads_csv_path=cfg.get('leads_csv_path', 'data/leads.csv'),
    rank_field='lead_rank',
    client=client,
)
import json
print(json.dumps(result, ensure_ascii=False, indent=2))
"
```

結果に応じて以下のいずれかを表示する：

**already_clean: true の場合**（すべての値がすでに A〜E）:
```
ランク値はすでに標準形式（A〜E）です。正規化は不要です。
```
→ ここで終了。

**already_clean: false の場合**:

```
🔍 ランク値マッピングを検出しました
─────────────────────────────────────
  B：担当者フォロー  →  B
  A：決裁者商談      →  A
  C：情報収集        →  C
─────────────────────────────────────
このマッピングで確定しますか？[Y / 修正する]
```

---

### Step 2: ユーザー確認

CLAUDE.md の「Skills 共通の対話ルール」に従いユーザー応答を解釈する。

**Y（確定）の場合** → Step 3 へ

**「修正する」の場合**:
- どの値のマッピングを変えたいか確認する（例: 「★5 は A ではなく B にしたい」）
- 修正後に再度 Step 2 の確認画面を表示する

---

### Step 3: cli_config.yaml へ保存

```bash
.venv/Scripts/python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.cli_runner import load_cli_config, save_cli_config

cfg = load_cli_config()
cfg['rank_value_mapping'] = {
    # ← ユーザーが確定したマッピングを dict リテラルで埋める
}
save_cli_config(cfg)
print('保存しました')
"
```

保存後に表示するメッセージ:
```
✅ ランク値マッピングを cli_config.yaml に保存しました。
   次回から LLM 推定をスキップして自動適用されます。

→ /email-workflow でメール生成フローを続けられます。
```

---

## 注意事項

- このスキルはランクフィールドの**値**の正規化のみを担う。
  カラム**名**の揺れには `/csv-mapping` を使うこと。
- `normalize_rank_values()` を使う実際の正規化は `/email-workflow` の CSV 読み込みステップで行われる。
  このスキルはマッピングの確認・保存のみ。
- `already_clean: true` の場合は `rank_value_mapping` を保存しない（上書きしない）。
