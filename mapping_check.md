# カラムマッピング 自動推定チェックシート

## チェック方法
アプリを起動して各CSVをアップロードし、
自動推定結果が以下の「期待値」と一致するか確認する。

---

## ① test_leads_rx_standard.csv（日本語・標準形式）

| 標準フィールド | CSVのカラム名 | 期待：自動推定される？ |
|---|---|---|
| visitor_name | 氏名 | ✅ ほぼ確実 |
| company_name | 会社名 | ✅ ほぼ確実 |
| email | メールアドレス | ✅ ほぼ確実 |
| lead_rank | 評価 | ✅ たぶん大丈夫 |
| memo | メモ | ✅ ほぼ確実 |
| interest_product | 関心製品 | ✅ たぶん大丈夫 |
| rep_name | 担当者名 | ⚠️ 要確認（新規追加カラム） |
| scan_time | スキャン時刻 | ⚠️ 要確認（新規追加カラム） |

**新規追加した「担当者名」「スキャン時刻」がconfig.pyの候補リストに入っているか確認必須。**

---

## ② test_leads_with_custom.csv（日本語・カスタムカラムあり）

標準カラムは①と同じ。追加カラムの扱いを確認。

| CSVのカラム名 | 期待：extra_プレフィックスで保持される？ |
|---|---|
| 導入予定時期 | ✅ extra_導入予定時期 として保持されるはず |
| 現在の課題 | ✅ extra_現在の課題 として保持されるはず |
| 年間予算規模 | ✅ extra_年間予算規模 として保持されるはず |
| 従業員数規模 | ✅ extra_従業員数規模 として保持されるはず |
| 最終意思決定者 | ✅ extra_最終意思決定者 として保持されるはず |
| 担当者名 | ⚠️ 上記と同じ |
| スキャン時刻 | ⚠️ 上記と同じ |

**カスタムカラムがメール生成のコンテキストに使われているか、生成されたメールを見て確認する。**

---

## ③ test_leads_messy.csv（英語カラム名・一番ブレている）

| 標準フィールド | CSVのカラム名 | 期待 | リスク |
|---|---|---|---|
| visitor_name | Full Name | ⚠️ 要確認 | "name"候補にないかも |
| company_name | Company | ⚠️ 要確認 | "company"候補にないかも |
| email | Email | ✅ たぶん通る | emailは汎用的 |
| lead_rank | Rank | ⚠️ 要確認 | "rank"が候補にあるか |
| memo | Notes | ⚠️ 要確認 | "notes"が候補にあるか |
| interest_product | Interest | ❌ 厳しいかも | "interest"は意味が広い |
| rep_name | Sales_Rep | ❌ ほぼ厳しい | 英語表記は候補にないはず |
| scan_time | Scan_Time | ❌ ほぼ厳しい | 英語表記は候補にないはず |

**英語カラムは手動マッピングが必要になる可能性が高い。**
**→ これはバグではなく仕様（手動マッピングUIが存在するため）。**
**→ ただしconfig.pyに英語候補を追加すれば自動化できる。**

---

## config.py に追加すべき候補（確認後に追加）

```python
# OPTIONAL_FIELDS の rep_name 候補に追加
"rep_name": [
    "担当者名", "担当者", "営業担当", "担当営業",
    "ブース担当者", "対応者",
    # 英語対応（messyCSV用）
    "Sales_Rep", "sales_rep", "Rep", "rep",
    "Salesperson", "Staff", "Assigned_To"
],

# OPTIONAL_FIELDS の scan_time 候補に追加
"scan_time": [
    "スキャン時刻", "スキャン日時", "来場時刻", "受付時刻",
    "スキャン時間", "QRスキャン時刻",
    # 英語対応
    "Scan_Time", "scan_time", "Visit_Time",
    "Timestamp", "timestamp", "Scanned_At"
],
```

---

## 確認後の記録欄（テスト時に書き込む）

| ファイル | rep_name マッピング結果 | scan_time マッピング結果 | その他気づき |
|---|---|---|---|
| rx_standard | | | |
| with_custom | | | |
| messy | | | |
