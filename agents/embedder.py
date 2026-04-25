from typing import List

try:
    from sentence_transformers import SentenceTransformer, util
except Exception:  # pragma: no cover - runtime dependency failure path
    SentenceTransformer = None
    util = None


_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBEDDER = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None and SentenceTransformer is not None:
        try:
            _EMBEDDER = SentenceTransformer(_MODEL_NAME)
        except Exception:
            _EMBEDDER = None
    return _EMBEDDER


def _chunk_text(text, chunk_words=200, overlap_words=50):
    words = (text or "").split()
    if not words:
        return []

    chunks: List[str] = []
    step = max(1, chunk_words - overlap_words)
    for start in range(0, len(words), step):
        chunk = words[start: start + chunk_words]
        if not chunk:
            continue
        chunks.append(" ".join(chunk))
        if start + chunk_words >= len(words):
            break
    return chunks


def _best_chunk_and_similarity(claim, evidence_text):
    if not claim or not evidence_text:
        return "", 0.0

    model = _get_embedder()
    chunks = _chunk_text(evidence_text)
    if model is None or not chunks or util is None:
        return "", 0.0

    try:
        claim_emb = model.encode(claim, convert_to_tensor=True)
        chunk_embs = model.encode(chunks, convert_to_tensor=True)
        sims = util.cos_sim(claim_emb, chunk_embs)[0]
        best_idx = int(sims.argmax().item())
        best_score = float(sims[best_idx].item())
        best_score = max(0.0, min(1.0, best_score))
        return chunks[best_idx], best_score
    except Exception:
        return "", 0.0


def claim_similarity(claim, evidence_text):
    """Return max cosine similarity between claim and evidence chunks."""
    _, best_score = _best_chunk_and_similarity(claim, evidence_text)
    return best_score


def find_best_matching_chunk(claim, evidence_text):
    """Return evidence chunk with highest similarity to the claim."""
    best_chunk, _ = _best_chunk_and_similarity(claim, evidence_text)
    return best_chunk
