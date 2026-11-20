import fitz  # pymupdf that reads PDFs and extracts text + page numbers
# SetenceTransforner turns text into vectors (numbers) and Util measures how similar two vectors are
from sentence_transformers import SentenceTransformer, util 

# Pretrained model that converts sentences to vectors where same meaning = closer vectors
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_pdf(path):
    try:
        # A context manager closes the document even if text extraction fails.
        with fitz.open(path) as doc:
            if not doc.is_pdf:
                raise ValueError("The uploaded file is not a PDF.")
            if doc.needs_pass:
                raise ValueError("Password-protected PDFs are not supported.")

            pages = []
            # enumerate starts at zero; PDF page numbers start at one.
            for i, page in enumerate(doc):
                text = page.get_text("text")
                if text.strip():
                    pages.append({"page": i + 1, "text": text})
    except (fitz.FileDataError, fitz.EmptyFileError) as exc:
        raise ValueError("The uploaded file is not a readable PDF.") from exc
    return pages

def _chunk_text(text, chunk_size=600, overlap=120):
    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end].strip())
        if end == text_length:
            break
        start = max(end - overlap, start + 1)

    return chunks

def search(pages, question, top_k=3):
    snippets = []
    for page in pages:
        for chunk in _chunk_text(page["text"]):
            snippets.append({
                "page": page["page"],
                "text": chunk,
            })

    if not snippets:
        return []

    texts = [snippet["text"] for snippet in snippets]
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
            "page": snippets[idx]["page"],
            "score": float(score),
            "snippet": snippets[idx]["text"]
        })
    return results
