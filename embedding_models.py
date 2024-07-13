
from numpy import array
from numpy import asarray
from numpy import zeros
import numpy as np

def load_glove_embeddings(glove_file="embeddings\glove.6B.100d.txt"):
    """Load GloVe model from file.

    Args:
        glove_file (str): Path to GloVe model file.

    Returns:
        dict: GloVe model as a dictionary.

    """
    embeddings_dictionary = dict()
    with open(glove_file, 'r', encoding='utf-8') as f:  
        for line in f:  
            values = line.split()  
            word = values[0]  
            vector_dimensions = asarray(values[1:], dtype='float32')
            embeddings_dictionary[word] = vector_dimensions

    return embeddings_dictionary  



def apply_embedding_matrix(embeddings_index, tokenizer, embedding_dim=100):
    """Create embedding matrix for Keras model.

    Args:
        embeddings_index (dict): GloVe model as a dictionary.
        tokenizer (Tokenizer): Tokenizer object.
        embedding_dim (int): Dimension of the embedding space.

    Returns:
        np.array: Embedding matrix.

    """
    num_words = min(20000, len(tokenizer.word_index) + 1)

    embedding_matrix = zeros((num_words, 100)) 
    for word, i in tokenizer.word_index.items():  
        embedding_vector = embeddings_index.get(word)  
        if embedding_vector is not None:  
            embedding_matrix[i] = embedding_vector  
    return embedding_matrix  