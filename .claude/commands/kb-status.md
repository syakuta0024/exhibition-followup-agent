# /kb-status — ナレッジベース状態確認

現在の ChromaDB ナレッジベースに登録されているドキュメント一覧・チャンク数・最終更新日時を表示する読み取り専用 Skill。

---

## 実行フロー

### Step 1: ナレッジベース状態を取得する

```bash
.venv/Scripts/python -c "
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
from src.cli_runner import run_kb_status
result = run_kb_status()
print(json.dumps(result, ensure_ascii=False))
"
```

### Step 2: 結果を表示する

**KB にドキュメントが存在する場合**、以下の形式で表示する：

```
📚 ナレッジベース状態
─────────────────────────────────
登録ドキュメント: {len(documents)} 件
総チャンク数    : {total_chunks} 件

  ✅ {source}   {chunk_count} chunks  ({source_type})
  ...

最終更新: {last_updated}
─────────────────────────────────
```

**KB が空（is_empty=true）の場合**、以下を表示して終了する：

```
⚠️  ナレッジベースが空です。
/setup の Step 5 でナレッジベースを構築してください。
```

---

## 表示ルール

- `source_type` は `markdown` または `pdf` または `unknown` を括弧付きで表示する
- `last_updated` が null の場合は「最終更新: 不明」と表示する
- ドキュメントはファイル名アルファベット順で表示する（戻り値がすでにソート済み）
- この Skill は **読み取り専用**。データの変更は行わない

---

## 注意事項

- KB 未構築の場合は `/setup` の Step 5 を案内する
- チャンク数が 0 件でも is_empty=false になる場合は ChromaDB の状態に問題がある可能性がある。
  その場合は「KB ファイルはあるが空です。/setup Step 5 で再構築してください」と案内する
