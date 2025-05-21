# utils/anonymizer.py

import re
import spacy
import subprocess

# Try loading spaCy model; auto-download if missing (Streamlit Cloud safe)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

# Match common date formats (e.g. 12/05/2020, 2020-01-01, Jan 1, 2020)
DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},\s+\d{4})\b",
    flags=re.IGNORECASE
)

def anonymize_text(text: str) -> str:
    """
    Replaces PERSON entities and DATEs with placeholders like <NAME_1> and <DATE>.
    """
    # 1. Replace dates using regex
    text_no_dates = DATE_PATTERN.sub("<DATE>", text)

    # 2. Process with spaCy
    doc = nlp(text_no_dates)

    # 3. Replace PERSON entities with <NAME_x>
    replacements = {}
    name_counter = 1
    for ent in doc.ents:
        if ent.label_ == "PERSON" and ent.text not in replacements:
            replacements[ent.text] = f"<NAME_{name_counter}>"
            name_counter += 1

    # 4. Replace all names (whole-word only to avoid partial matches)
    anonymized = text_no_dates
    for name, tag in replacements.items():
        anonymized = re.sub(rf"\b{re.escape(name)}\b", tag, anonymized)

    return anonymized
