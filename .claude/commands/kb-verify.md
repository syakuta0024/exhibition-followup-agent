# kb-verify — 製品ナレッジ確認・保存スキル

KBに登録された技術資料をLLMで読み込み、製品カードを1件ずつ確認・修正して
`data/product_knowledge.yaml` に保存する。

---

## 実行フロー

### Step 0: KB構築チェック

```bash
.venv/Scripts/python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.cli_runner import run_kb_status
kb = run_kb_status()
print('is_empty:', kb['is_empty'])
print('total_chunks:', kb['total_chunks'])
"
```

- `is_empty: True` の場合は以下を表示して終了:
  ```
  ナレッジベースが空です。
  先に /setup Step 5 または /email-workflow Step 3 でKBを構築してから実行してください。
  ```

---

### Step 1: 製品カード下書き生成

```bash
.venv/Scripts/python -c "
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
from src.cli_runner import run_kb_summary
result = run_kb_summary()
print('ok:', result['ok'])
print('message:', result['message'])
print(json.dumps(result['products'], ensure_ascii=False, indent=2))
"
```

- `ok: False` の場合はエラーメッセージを表示して終了。
- 成功時: 「{N}製品のカード下書きを生成しました。1件ずつ確認します。」と表示。

---

### Step 2: 製品を1件ずつ提示して確認

製品リストをインデックス順にループし、1製品ずつ以下の形式で表示する:

```
━━━━━━━━━━━━━━━━
[製品 {i}/{N}] {製品名}

{カード本文}

→ この理解で合っていますか？
  [y] OK  /  [修正] 違う点を教えてください
━━━━━━━━━━━━━━━━
```

CLAUDE.md の「Skills 共通の対話ルール」に従いユーザー応答を解釈する:

| 応答 | 対応 |
|---|---|
| 明確な肯定（y / ok / はい / 進めて 等） | そのカードを確定し次の製品へ |
| 曖昧（大丈夫・いいです 等） | 「進めてもよろしいですか？」と再確認 |
| 修正要望 | Step 3 へ |

---

### Step 3: カード本文の修正（「修正」の場合のみ）

ユーザーの指摘（例: 「価格が違う」「機能の説明が不足」）を受けて、
そのカード本文を書き換え、再度 Step 2 の確認画面を表示する。

修正は会話の中でテキストとして行う（外部ファイルは不要）。
確認が取れたら確定して次の製品へ。

---

### Step 4: 全製品確認後に保存

全製品の確認が済んだら `save_product_knowledge()` で保存する:

```bash
.venv/Scripts/python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.cli_runner import save_product_knowledge

products = {
    # ← 確定した製品名: カード本文 の dict をここに埋める
}
save_product_knowledge(products)
print('保存しました')
"
```

---

### Step 5: 完了サマリ

```
✅ {N}製品の製品ナレッジを data/product_knowledge.yaml に保存しました。

→ /email-workflow でメール生成時に反映されます（Phase 4 実装後）。
```

---

## 注意事項

- Step 0 で KB が空の場合はカード生成を実行しない（LLM API コストを無駄にしない）。
- 修正は何度でも繰り返してよい。「y」が確認できるまで次の製品へ進まない。
- 保存は全製品の確認が完了してから一括で行う（途中保存しない）。
- `load_product_knowledge()` で既存ファイルを確認したい場合は別途手動で確認すること。
  このスキルは常に LLM の新規下書きから始め、既存ファイルを上書きする設計。
