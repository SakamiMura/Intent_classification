import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
    Perform basic text preprocessing:
        - Lowercasing
        - Removing punctuation
        - Removing stopwords

    Args:
        text (str): Raw input text.

    Returns:
        str: Cleaned and normalized text.
    """
    text = text.lower()
    # DO NOT remove digits (they might carry intent-relevant information)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]
    cleaned_text = " ".join(tokens)
    return cleaned_text
