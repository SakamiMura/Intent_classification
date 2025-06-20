from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_text(train_texts, val_texts=None, test_texts=None, max_features=5000):
    """
    Converts raw text into TF-IDF vectors.

    Args:
        train_texts (list[str]): Cleaned training texts.
        val_texts (list[str], optional): Validation texts.
        test_texts (list[str], optional): Test texts.
        max_features (int): Max number of words to keep.

    Returns:
        tuple: (X_train, X_val, X_test, vectorizer)
    """

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),   # uwzględnij też dwuwyrazowe frazy
        sublinear_tf=True,    # log(tf) - lepsze zachowanie
        strip_accents='unicode'  # usuwaj akcenty
    )

    X_train = vectorizer.fit_transform(train_texts)

    X_val = vectorizer.transform(val_texts) if val_texts is not None else None
    X_test = vectorizer.transform(test_texts) if test_texts is not None else None

    return X_train, X_val, X_test, vectorizer
