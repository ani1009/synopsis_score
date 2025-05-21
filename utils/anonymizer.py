# utils/anonymizer.py

import re
import spacy

# Load spaCy’s small model (only NER stuff is needed here)
nlp = spacy.load("en_core_web_sm")

# Basic date regex (slashes, dashes, or “Jan 1, 2020” style). Not bulletproof,
# but works for most cases. If you see more patterns, you can expand this.
DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},\s+\d{4})\b"
)

def anonymize_text(text: str) -> str:
    """
    Roughly replace dates and person names with placeholders.
    Doesn’t attempt to catch everything—just a simple, in-memory pass.
    """
    # 1) Replace dates first
    def _date_repl(match):
        return "<DATE>"

    text_no_dates = DATE_PATTERN.sub(_date_repl, text)

    # 2) Use spaCy to find PERSON entities
    doc = nlp(text_no_dates)
    # Build a map: original_person_name -> <NAME_i>
    repl_map = {}
    counter = 1

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            orig = ent.text
            if orig not in repl_map:
                placeholder = f"<NAME_{counter}>"
                repl_map[orig] = placeholder
                counter += 1

    # 3) Do a simple replace for each name found
    anonymized = text_no_dates
    for orig_name, placeholder in repl_map.items():
        # Escape punctuation in orig_name so regex doesn't break
        anonymized = re.sub(re.escape(orig_name), placeholder, anonymized)

    return anonymized
