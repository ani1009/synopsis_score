# utils/anonymizer.py

import re
import spacy

# Load the spaCy model (will already be installed via requirements.txt)
nlp = spacy.load("en_core_web_sm")

# Pattern to catch common date formats
DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},\s+\d{4})\b",
    flags=re.IGNORECASE
)

def anonymize_text(text: str) -> str:
    text_no_dates = DATE_PATTERN.sub("<DATE>", text)
    doc = nlp(text_no_dates)

    replacements = {}
    name_counter = 1
    for ent in doc.ents:
        if ent.label_ == "PERSON" and ent.text not in replacements:
            replacements[ent.text] = f"<NAME_{name_counter}>"
            name_counter += 1

    anonymized = text_no_dates
    for name, tag in replacements.items():
        anonymized = re.sub(rf"\b{re.escape(name)}\b", tag, anonymized)

    return anonymized
