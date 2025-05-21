# 📝 Synopsis Scoring App (Streamlit + SBERT + Heuristics)

This project is a **Streamlit-based web application** that evaluates the quality of a written synopsis based on a given article. It uses anonymization, sentence embeddings, and heuristic metrics to assign a numeric score and provide feedback.

---

## ✅ Features

- Upload an article (PDF or TXT) and your synopsis (TXT).
- Automatic **anonymization** of names, dates, and personal identifiers.
- Semantic scoring using **sentence-transformers** (MiniLM model).
- Three sub-scores:
  - **Content Coverage** (how well the synopsis matches the article)
  - **Coherence** (how smoothly the sentences connect)
  - **Clarity** (based on Flesch Reading Ease score)
- Final score: **0–100**
- 2–3 lines of feedback based on heuristics.
- Privacy-first: raw texts never leave your device.

---

## 🛠️ Tech Stack

- Python 3.8+
- [Streamlit](https://streamlit.io/) for the web UI
- [SentenceTransformers](https://www.sbert.net/) for embedding-based similarity
- [spaCy](https://spacy.io/) for anonymization (NER)
- [textstat](https://pypi.org/project/textstat/) for clarity scoring
- [PyMuPDF](https://pymupdf.readthedocs.io/) to extract text from PDFs

---

## ⚙️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/ani1009/score_synopsis.git
cd score_synopsis
##2. Create Virtual Environment:

python -m venv .venv

##3. Activate it :
.venv\Scripts\activate  (For Windows)
source .venv/bin/activate (For mac)

##4. Install Dependencies:
pip install -r requirements.txt


##5. Run the App:
streamlit run app.py



