# audio-matching — 音声ファイル紐づけスキル

展示会当日の録音音声ファイルをリードに紐づけるスキル。
ファイル名解析 + タイムスタンプ照合を行い、曖昧なケースだけユーザーに確認する。

---

## 実行フロー

### Step 0: 前提確認

まず以下を確認する：

1. 「音声ファイルのフォルダはどこですか？（例: test_m4a/音声データ）」
2. 「ファイル名に顧客名や担当者名は含まれていますか？  
   含まれている場合、フォーマットを教えてください。  
   例: `YYYYMMDD_担当者名_連番.m4a`（推奨形式）」
3. リードCSV に `rep_name`（担当営業）や `scan_time`（スキャン時刻）はあるか
   → `/inspect-data` の結果がある場合はそれを参照

---

### Step 1: 音声ファイル一覧の取得

```bash
.venv/Scripts/python -c "
import sys, os
sys.stdout.reconfigure(encoding='utf-8')

audio_dir = '<AUDIO_DIR>'  # Step0 で確認したフォルダパスに置き換える
exts = ('.m4a', '.mp3', '.wav')

files = [f for f in os.listdir(audio_dir) if f.lower().endswith(exts)]
files.sort()
print(f'音声ファイル数: {len(files)}')
for f in files:
    print(f'  {f}')
"
```

ファイル一覧を表示し、ユーザーが教えてくれたフォーマットと照合する。

---

### Step 2: AudioMatcher で紐づけ実行

```bash
.venv/Scripts/python -c "
import sys, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
from src.config import Config
from src.utils import load_leads, auto_map_columns, apply_column_mapping
from src.audio_matcher import AudioMatcher

# リードCSV読み込み
from src.cli_runner import load_cli_config
cfg = load_cli_config()
df = load_leads(cfg.get('leads_csv_path', 'data/leads.csv'))
mapping = auto_map_columns(df.columns.tolist(), {**Config.REQUIRED_FIELDS, **Config.OPTIONAL_FIELDS})
df = apply_column_mapping(df, mapping)

matcher = AudioMatcher()

# 音声ファイルのメタデータ取得（mutagen が使える場合）
import os
audio_dir = '<AUDIO_DIR>'  # Step0 で確認したフォルダパスに置き換える
audio_meta_list = []
for fname in os.listdir(audio_dir):
    if fname.lower().endswith(('.m4a', '.mp3', '.wav')):
        fpath = os.path.join(audio_dir, fname)
        with open(fpath, 'rb') as f:
            file_bytes = f.read()
        rep_name = matcher.parse_rep_from_filename(fname)
        audio_meta_list.append({
            'filename': fname,
            'file_bytes': file_bytes,
            'rep_name': rep_name,
            'start_time': None,  # AudioProcessor.get_audio_metadata() で取得
            'duration_sec': 0,
        })

# 紐づけ実行
rep_col = 'rep_name' if 'rep_name' in df.columns else None
ts_col = 'scan_time' if 'scan_time' in df.columns else None
results = matcher.match(audio_meta_list, df, rep_col=rep_col, timestamp_col=ts_col)

print()
print('=== 紐づけ結果 ===')
for r in results:
    conf_icon = {'green': '[確定]', 'yellow': '[要確認]', 'red': '[未紐づけ]'}.get(r.confidence, '[?]')
    lead_info = ''
    if r.lead_idx is not None and r.lead_idx in df.index:
        row = df.loc[r.lead_idx]
        lead_info = f'{row.get(\"visitor_name\",\"\")} ({row.get(\"company_name\",\"\")})'
    print(f'  {conf_icon} {r.audio_filename} -> {lead_info or \"(未紐づけ)\"}  [{r.method}]')

# 赤フラグ率チェック
red_rate = matcher.get_red_flag_rate(results)
if red_rate > 0.30:
    print()
    print(f'[警告] 命名規則違反率が高いです（{red_rate:.0%}）。')
    print('  推奨形式: YYYYMMDD_担当者名_連番.m4a')
"
```

---

### Step 3: 要確認ケースの対話確認

信頼度が `yellow`（担当者名一致・タイムスタンプなし）のファイルを1件ずつ確認する：

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[要確認 1/3]

音声ファイル: 営業A_002.mp3（担当: 営業A）
紐づけ候補:
  1. L004  山田 花子  (XYZ株式会社)   — 営業A担当・タイムスタンプ不明
  2. L007  田中 一郎  (ABC製造)       — 営業A担当・タイムスタンプ不明

→ この音声ファイルはどのリードの録音ですか？
  [1] L004 山田 花子
  [2] L007 田中 一郎
  [n] なし（未紐づけ）
  [s] スキップ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

信頼度が `red`（担当者名不明）のファイルは、リード一覧から手動選択を促す。

---

### Step 4: 文字起こしの実行確認

紐づけが完了したファイルについて：

「紐づけた音声ファイルを文字起こし（Whisper API）しますか？  
文字起こしするとメールの品質が大幅に向上します。  
コスト: 約 ¥0.9/分（Whisper料金）」

「はい」の場合：

```bash
.venv/Scripts/python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.audio_processor import AudioProcessor
from src.config import Config

processor = AudioProcessor(api_key=Config.OPENAI_API_KEY, llm=None)

audio_path = '<AUDIO_DIR>/営業A_002.mp3'  # Step0 で確認したフォルダパス + ファイル名に置き換える
with open(audio_path, 'rb') as f:
    file_bytes = f.read()

import os
filename = os.path.basename(audio_path)
transcript = processor.transcribe(file_bytes, filename)
print('=== 文字起こし結果 ===')
print(transcript[:500], '...' if len(transcript) > 500 else '')
"
```

---

### Step 4.5: 音声コンテキストの永続化（自動）

文字起こしが完了したリードについて、`output/audio_context.json` に追記する。
このファイルは `/email-workflow` の `run_generate()` が自動で読み込み、
各リードのメール生成時に最優先コンテキストとして使う。

**lead_key の生成ロジックは `src/cli_runner.build_lead_key()` を必ず使うこと**
（run_generate 側と完全一致させる必要があるため、ここでロジックを書き直さない）。

```bash
.venv/Scripts/python -c "
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from src.cli_runner import build_lead_key, DEFAULT_AUDIO_CONTEXT_PATH

# このスキル内で蓄積したリード辞書 → transcript/needs の対応を渡す
# 例:
#   entries = [
#       {'lead': {'lead_id': 'L001', 'visitor_name': '田中 太郎', 'company_name': 'XYZ'},
#        'transcript': '...全文...',
#        'needs': {'summary': '...', 'issues': '...', 'needs': '...',
#                  'budget': '...', 'decision_maker': '...', 'temperature': 'high'}},
#   ]
entries = []  # ← Step 2〜4 で得たリード+文字起こし結果に差し替える

out_path = Path(DEFAULT_AUDIO_CONTEXT_PATH)
out_path.parent.mkdir(parents=True, exist_ok=True)

# 既存ファイルがあればマージ（複数回実行で消えないように）
existing = {}
if out_path.exists():
    try:
        existing = json.loads(out_path.read_text(encoding='utf-8')) or {}
    except json.JSONDecodeError:
        existing = {}

for e in entries:
    key = build_lead_key(e['lead'])
    existing[key] = {
        'transcript': e.get('transcript', '') or '',
        'needs': e.get('needs') or {},
    }

out_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'audio_context.json を更新しました: {out_path} ({len(existing)} 件)')
"
```

`needs` には `AudioProcessor.extract_needs()` の戻り値（`summary` / `issues` /
`needs` / `budget` / `decision_maker` / `temperature` を含む dict）をそのまま入れる。
`extract_needs` を呼ばずに文字起こしのみ取得した場合は `needs={}` で問題ない。

---

### Step 5: 結果サマリー

```
━━━ 音声紐づけ完了 ━━━

■ 自動確定 (green): 5件
  - 20260424_営業A_001.m4a → L001 山田 太郎 (ABC製造) [タイムスタンプ一致]
  - 20260424_営業B_001.m4a → L005 鈴木 花子 (XYZ工業) [タイムスタンプ一致]
  ...

■ 手動確定 (yellow→確定): 2件
  - 営業A_002.mp3 → L004 山田 花子 [手動選択]

■ 未紐づけ (red): 1件
  - rec001.m4a → 担当者名不明のため未紐づけ

━━━ 次のアクション ━━━
  → /email-workflow で文字起こし情報を含めてメール生成できます
```

---

## 注意事項

- `mutagen` がインストールされていない場合、タイムスタンプ取得はスキップ（ファイル名解析のみ）
- 文字起こしは Whisper API を使用するため OPENAI_API_KEY が必要
- 紐づけ結果は会話内のメモとして保持する（/email-workflow 実行時に参照）
- 赤フラグ率 > 30% の場合は命名規則違反の警告を表示する
