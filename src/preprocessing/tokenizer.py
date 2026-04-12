"""spaCy-based tokenization utilities for movie plot summaries.

Uses the ``en_core_web_sm`` model to lemmatize and optionally filter
stop-words from cleaned plot text produced by :mod:`src.preprocessing.cleaner`.
"""

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


def tokenize_batch(
    texts: list[str],
    remove_stopwords: bool = True,
    do_lemmatize: bool = True,
    batch_size: int = 256,
) -> list[list[str]]:
    """Tokenize a list of plot strings in batches using spaCy's pipe.

    Parameters
    ----------
    texts:
        Raw or cleaned plot strings.
    remove_stopwords:
        When *True*, tokens present in spaCy's English stop-word list are
        removed.
    do_lemmatize:
        When *True*, each token is replaced by its lemma.
    batch_size:
        Number of documents processed per spaCy batch (higher → faster but
        more memory).

    Returns
    -------
    list[list[str]]
        One token list per input text, in the same order.
    """
    results: list[list[str]] = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        tokens = []
        for token in doc:
            if not token.is_alpha:
                continue
            lemma = token.lemma_.lower() if do_lemmatize else token.text.lower()
            if remove_stopwords and lemma in STOP_WORDS:
                continue
            tokens.append(lemma)
        results.append(tokens)
    return results


def detokenize(tokens: list[str]) -> str:
    """Reconstruct text from a token list using single-space joins.

    Parameters
    ----------
    tokens:
        Token list produced by :func:`tokenize` or :func:`tokenize_batch`.

    Returns
    -------
    str
        A space-joined string. Empty lists return an empty string.
    """
    if not tokens:
        return ""
    return " ".join(str(token) for token in tokens).strip()
