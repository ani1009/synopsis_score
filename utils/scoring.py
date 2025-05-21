# utils/scoring.py

import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
import textstat

# Load transformer model once
model = SentenceTransformer("all-MiniLM-L6-v2")

def split_into_sentences(text: str) -> list[str]:
    """
    Splits text into sentences using punctuation-based regex.
    """
    text = text.strip()
    if not text:
        return []
    return re.split(r'(?<=[.!?])\s+', text)

def compute_content_coverage(article: str, synopsis: str) -> float:
    emb_art = model.encode(article, normalize_embeddings=True)
    emb_syn = model.encode(synopsis, normalize_embeddings=True)
    return util.cos_sim(emb_art, emb_syn).item()

def compute_coherence(synopsis: str) -> float:
    sentences = split_into_sentences(synopsis)
    if len(sentences) < 2:
        return 0.0
    embs = model.encode(sentences, normalize_embeddings=True)
    sims = [util.cos_sim(embs[i], embs[i + 1]).item() for i in range(len(embs) - 1)]
    return float(np.mean(sims))

def compute_clarity(synopsis: str) -> float:
    try:
        score = textstat.flesch_reading_ease(synopsis)
        score = max(0, min(score, 100))
        return score / 100.0
    except:
        return 0.5  # fallback if textstat fails

def aggregate_scores(article: str, synopsis: str) -> dict:
    cc = compute_content_coverage(article, synopsis)
    co = compute_coherence(synopsis)
    cl = compute_clarity(synopsis)
    overall = 0.5 * cc + 0.25 * co + 0.25 * cl
    return {
        "content_coverage": cc,
        "coherence": co,
        "clarity": cl,
        "overall": overall
    }

def generate_feedback(scores: dict) -> str:
    cc, co, cl = scores["content_coverage"], scores["coherence"], scores["clarity"]
    lines = []

    if cc < 0.4:
        lines.append("It looks like the synopsis misses many important points.")
    elif cc < 0.7:
        lines.append("You’ve captured most main ideas, but some key details are missing.")
    else:
        lines.append("Good job—your synopsis covers the main content well.")

    if co < 0.5:
        lines.append("Sentences feel a bit disconnected. Try smoother transitions.")
    if cl < 0.5:
        lines.append("Some sentences are a bit hard to follow—try simpler phrasing.")

    return " ".join(lines[:3])
