from src.vectordb import VectorDBManager

db = VectorDBManager()

# ベクトルDBの総件数を確認
count = db.vectorstore._collection.count()
print(f"ベクトルDB総チャンク数: {count}")
print()

# PDFからの内容が入っているか確認
print("=== PDF由来のチャンク確認 ===")
results = db.search_tech_docs('導入事例 プレス 不良品削減', top_k=5)
for i, r in enumerate(results, 1):
    source = r["metadata"].get("source_file", "不明")
    print(f'[{i}] ソース: {source}')
    print(f'    内容冒頭: {r["text"][:150]}')
    print()

print("=== EdgeGuard PDF確認 ===")
results2 = db.search_tech_docs('エッジAI 振動センサー 月間47件', top_k=3)
for i, r in enumerate(results2, 1):
    source = r["metadata"].get("source_file", "不明")
    print(f'[{i}] ソース: {source}')
    print(f'    内容冒頭: {r["text"][:150]}')
    print()