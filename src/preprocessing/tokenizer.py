import spacy
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load("en_core_web_sm")

def tokenize(text: str, remove_stopwords: bool = True, do_lemmatize: bool = True) -> list[str]:
    """
    Tokenize a cleaned plot string into a list of normalized tokens.
    """
    tokens = []
    for token in nlp(text):
        if not token.is_alpha:
            continue
        lemma = token.lemma_.lower() if do_lemmatize else token.text.lower()
        if remove_stopwords and lemma in STOP_WORDS:
            continue
        tokens.append(lemma)
    return tokens