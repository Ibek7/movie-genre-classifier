
import pandas as pd
import re


def load_data(input_path: str) -> pd.DataFrame:
    """
    Load the raw CSV into a DataFrame.
    """
    return pd.read_csv(input_path)


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop exact duplicates based on available columns (Title and/or Plot).
    """
    # Determine which columns exist for deduplication
    subset_cols = [col for col in ("Title", "Plot") if col in df.columns]
    if not subset_cols:
        return df
    return df.drop_duplicates(subset=subset_cols)


def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with missing Plot or Genre.
    """
    return df.dropna(subset=["Plot", "Genre"])


def normalize_text(text: str) -> str:
    """
    Normalize text by lowercasing, removing HTML tags, non-alphanumeric chars, and collapsing whitespace.
    """
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)            # strip HTML tags
    text = re.sub(r"[^a-z0-9\s\|]", " ", text)    # keep letters/numbers/pipes
    text = re.sub(r"\s+", " ", text).strip()      # collapse whitespace
    return text


def clean_and_save(input_path: str, output_path: str) -> None:
    """
    Run full cleaning pipeline and save processed CSV.

    Steps:
    1. Load raw data.
    2. Drop duplicates.
    3. Drop missing values.
    4. Normalize the 'Plot' text.
    5. Save cleaned DataFrame to output_path.
    """
    df = load_data(input_path)
    df = drop_duplicates(df)
    df = drop_missing(df)
    df["Plot"] = df["Plot"].apply(normalize_text)
    df.to_csv(output_path, index=False)
