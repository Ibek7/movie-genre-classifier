from spacy.lang.en.stop_words import STOP_WORDS
import re
import spacy

nlp = spacy.load("en_core_web_sm")

def tokenize(text):
    text = re.sub(r'\s+', ' ', text)
    doc = nlp(text)
    tokens = [token.lemma_.lower().strip() for token in doc if token.text.lower() not in STOP_WORDS and token.is_alpha]
    return tokens
