import re
import random
from typing import List

# 1) Negation insertion
AUX_VERBS = r"\b(is|are|can|may|should)\b"
def insert_negation(text: str) -> str:
    """
    Inserts 'not' after the first auxiliary verb in the text.
    E.g. "It is safe." -> "It is not safe."
    """
    def repl(m):
        return f"{m.group(1)} not"
    return re.sub(AUX_VERBS, repl, text, count=1, flags=re.IGNORECASE)


# 2) Numeric perturbation
NUMBER_RE = re.compile(r"\b(\d+(\.\d+)?)\b")
def perturb_numbers(text: str, factor: float = 10.0) -> str:
    """
    Multiplies each numeric literal by `factor`.
    E.g. "Take 5 mg twice." -> "Take 50.0 mg twice."
    """
    def repl(m):
        val = float(m.group(1))
        return str(val * factor)
    return NUMBER_RE.sub(repl, text)


# 3) Fallback random word replacement
#    A small pool of medical-adjacent filler tokens
REPLACEMENT_POOL = [
    # Medications
    "aspirin", "ibuprofen", "acetaminophen", "penicillin", "amoxicillin", "metformin", "lisinopril", "atorvastatin",
    # Body parts
    "stomach", "lung", "heart", "kidney", "liver", "brain", "pancreas", "spleen", "intestine", "bladder",
    # Colors
    "red", "blue", "green", "yellow", "purple", "orange",
    # Symptoms
    "fever", "headache", "nausea", "dizziness", "fatigue", "cough", "sore throat", "rash",
    # Medical conditions
    "diabetes", "hypertension", "arthritis", "asthma", "anemia", "thyroid", "migraine",
    # Medical procedures
    "surgery", "biopsy", "vaccination", "transfusion", "dialysis", "chemotherapy",
    # Medical terms
    "inflammation", "infection", "allergy", "tumor", "lesion", "ulcer", "fracture"
]

def replace_random_words(text: str, replace_frac: float = 0.3) -> str:
    """
    Replaces `replace_frac` fraction of tokens in `text` with random choices
    from REPLACEMENT_POOL.
    """
    tokens = text.split()
    n = len(tokens)
    k = max(1, int(n * replace_frac))
    
    idxs = random.sample(range(n), k)
    for i in idxs:
        tokens[i] = random.choice(REPLACEMENT_POOL)
    return " ".join(tokens)


# 4) Combined augmentation pipeline
def augment_negative(
    answer: str,
    negation_prob: float = 0.5,
    perturb_prob: float = 0.8,
    number_factor: float = 10.0,
    fallback_frac: float = 0.1
) -> str:
    """
    Applies negation insertion and numeric perturbation with given probabilities.
    If neither changes the text, applies fallback random word replacement.
    """
    augmented = answer
    
    # 1) Negation
    if random.random() < negation_prob:
        augmented = insert_negation(augmented)
    
    # 2) Numeric perturbation
    if random.random() < perturb_prob:
        augmented = perturb_numbers(augmented, factor=number_factor)
    
    # 3) Fallback if still identical
    if augmented == answer:
        augmented = replace_random_words(answer, replace_frac=fallback_frac)
    
    return augmented
