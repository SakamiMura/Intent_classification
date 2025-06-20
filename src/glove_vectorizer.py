import numpy as np

def load_glove_embeddings(glove_path):
    """
    Load GloVe embeddings from a .txt file into a dictionary.

    Args:
        glove_path (str): Path to the GloVe file.

    Returns:
        dict: Mapping from word to embedding (as np.array).
    """
    embeddings = {}
    with open(glove_path, 'r', encoding='utf8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vector
    return embeddings


def vectorize_with_glove(texts, embeddings, dim=50):
    """
    Convert a list of texts to GloVe-based vectors by averaging word embeddings.

    Args:
        texts (List[str]): List of cleaned texts.
        embeddings (dict): word → np.array (GloVe vectors)
        dim (int): Dimension of embeddings (default 50)

    Returns:
        np.array: Matrix shape (len(texts), dim)
    """
    vectors = []

    for text in texts:
        tokens = text.split()
        vecs = [embeddings[word] for word in tokens if word in embeddings]

        if len(vecs) > 0:
            avg_vec = np.mean(vecs, axis=0)
        else:
            avg_vec = np.zeros(dim)

        vectors.append(avg_vec)

    return np.array(vectors)
