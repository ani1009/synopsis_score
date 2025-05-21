# utils/scoring.py
import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
import textstat

# Load the embeddings model once
model = SentenceTransformer("all-MiniLM-L6-v2")

def split_into_sentences(text: str) -> list[str]:
    """
    A simple regex-based sentence splitter: splits on '.', '?', or '!',
    as long as they are followed by whitespace. Keeps the punctuation.
    """
    text = text.strip()
    if not text:
        return []
    # Split on punctuation + space. E.g., "Hello world. How are you?" → ["Hello world.", "How are you?"]
    return re.split(r'(?<=[\.\?\!])\s+', text)

def compute_content_coverage(article: str, synopsis: str) -> float:
    """
    Cosine similarity between the embedding of the entire article
    and the embedding of the entire synopsis. Returns a float in [0, 1].
    """
    emb_art = model.encode(article, normalize_embeddings=True)
    emb_syn = model.encode(synopsis, normalize_embeddings=True)
    return util.cos_sim(emb_art, emb_syn).item()

def compute_coherence(synopsis: str) -> float:
    """
    Split the synopsis into sentences using split_into_sentences().
    Then embed each sentence and compute the average cosine similarity
    between consecutive sentences. If there is only one sentence, return 0.0.
    """
    sentences = split_into_sentences(synopsis)
    if len(sentences) < 2:
        return 0.0

    embs = model.encode(sentences, normalize_embeddings=True)
    sims = []
    for i in range(len(embs) - 1):
        sims.append(util.cos_sim(embs[i], embs[i + 1]).item())

    return float(np.mean(sims))

def compute_clarity(synopsis: str) -> float:
    """
    Use textstat's Flesch Reading Ease (range roughly 0–100).
    Clamp it to [0, 100], then divide by 100 to get a float in [0, 1].
    """
    flesch = textstat.flesch_reading_ease(synopsis)
    if flesch < 0:
        flesch = 0
    elif flesch > 100:
        flesch = 100
    return flesch / 100.0

def aggregate_scores(article: str, synopsis: str) -> dict:
    """
    Returns a dictionary of four floats (each in [0,1]):
      • content_coverage  (50% weight)
      • coherence         (25% weight)
      • clarity           (25% weight)
      • overall           (the weighted sum)
    """
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
    """
    Simple threshold-based feedback. Produces up to 3 short sentences.
    """
    cc = scores["content_coverage"]
    co = scores["coherence"]
    cl = scores["clarity"]

    lines = []

    # 1) Content coverage feedback
    if cc < 0.4:
        lines.append("It looks like the synopsis misses many important points.")
    elif cc < 0.7:
        lines.append("You’ve captured most main ideas, but some key details are missing.")
    else:
        lines.append("Good job—your synopsis covers the main content well.")

    # 2) Coherence feedback
    if co < 0.5:
        lines.append("Sentences feel a bit disconnected. Try smoother transitions.")

    # 3) Clarity feedback
    if cl < 0.5:
        lines.append("Some sentences are a bit hard to follow—try simpler phrasing.")

    return " ".join(lines[:3])
