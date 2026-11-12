from rag import load_docs, split_docs

KEYWORD = "72시간 전까지"
SIZES = [100, 400, 1500]

docs = load_docs()

print("== 문서별 글자 수 ==")
for d in docs:
    print(f"{d['source']}: {len(d['text'])}자")
print(f"전체: {sum(len(d['text']) for d in docs)}자\n")

print("== 조각 크기별 조각 수 ==")
results = {}
for size in SIZES:
    results[size] = split_docs(docs, size)
    print(f"크기 {size}: {len(results[size])}개")

for size in SIZES:
    hits = [c for c in results[size] if KEYWORD in c["text"]]
    print(f"\n== 크기 {size} · '{KEYWORD}' 포함 조각 {len(hits)}개 ==")
    for c in hits:
        print(f"[출처: {c['source']} / {len(c['text'])}자]")
        print(c["text"])
