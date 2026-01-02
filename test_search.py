# This is a test page to verify if the semantic search works
from app.search import load_pdf, search

pages = load_pdf("sample.pdf")
results = search(pages, "What is this document about?", 1)

for r in results:
    print(f"Page {r['page']} | score={r['score']:.3f}")
    print(r["snippet"])
    print("-" * 60)
