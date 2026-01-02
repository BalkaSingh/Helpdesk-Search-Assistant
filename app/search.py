import fitz  # pymupdf that reads PDFs and extracts text + page numbers
# SetenceTransforner turns text into vectors (numbers) and Util measures how similar two vectors are
from sentence_transformers import SentenceTransformer, util 

# Pretrained model that converts sentences to vectors where same meaning = closer vectors
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_pdf(path):
    doc = fitz.open(path)
    # Each page is a dict that containes a page number 
    pages = []
    # Loop through the doc and retrieve each index (starting at 0) & page (Page object)
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages.append({
                "page": i + 1,
                "text": text
            })
    return pages

def search(pages, question, top_k=3):
    texts = [p["text"] for p in pages]
    embeddings = model.encode(texts, convert_to_tensor=True)
    query_embedding = model.encode(question, convert_to_tensor=True)

    # Computes cosine similarity between vectors. Where 1 means very similar to 0 meaning not similar and negative meaning not close at all
    # Index 0 because we want one score per page and we take the first row of the matrix because cos_sim returns 2d tensor
    scores = util.cos_sim(query_embedding, embeddings)[0]
    # gives the top "k"(number) scores meaning they are the closest values.
    best = scores.topk(k=min(top_k, len(texts)))

    results = []
    # Zip together each tensor and it returns an iterator of tuples 
    for score, idx in zip(best.values, best.indices):
        idx = int(idx)
        results.append({
            "page": pages[idx]["page"],
            "score": float(score),
            "snippet": pages[idx]["text"]
        })
    return results
