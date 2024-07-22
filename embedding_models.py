
from numpy import asarray
from numpy import zeros
from gensim.models import KeyedVectors
import fasttext

### loaded from https://nlp.stanford.edu/projects/glove/ and https://www.cs.cmu.edu/~afm/projects/multilingual_embeddings.html
def load_glove_embeddings(language="english"):
    """Load GloVe model from file.

    Args:
        language (str): Language of the GloVe model. Defaults to "english".

    Returns:
        dict: GloVe model as a dictionary.

    """
    embeddings_dictionary = dict()
    if language == "english":
        glove_file="embeddings\glove_eng.txt"
        
        with open(glove_file, 'r', encoding='utf-8') as f:  
            for line in f:  
                values = line.split()  
                word = values[0]  
                vector_dimensions = asarray(values[1:], dtype='float32')
                embeddings_dictionary[word] = vector_dimensions
    else:
        glove_file="embeddings\glove_pt.txt"
        model = KeyedVectors.load_word2vec_format(glove_file)
        for word in model.key_to_index:
            embeddings_dictionary[word] = model[word]

    return embeddings_dictionary

# loaded from http://nilc.icmc.usp.br/embeddings and http://vectors.nlpl.eu/repository/
def load_word2vec_embeddings(language="english"):
    """Load Word2Vec model from file.

    Args:
        language (str): Language of the Word2Vec model. Defaults to "english".

    Returns:
        dict: Word2Vec model as a dictionary.

    """
    embeddings_dictionary = dict()
    if language == "english":
        file="embeddings\word2vec_eng.txt"
    else:
        file="embeddings\word2vec_pt.txt"
    model = KeyedVectors.load_word2vec_format(file)
    for word in model.key_to_index:
        embeddings_dictionary[word] = model[word]

    return embeddings_dictionary


### loaded from https://fasttext.cc/ and reduced to 100 dimensions
def load_fasttext_embeddings(language="english"):
    """Load fasttext model from file.

    Args:
        language (str): Language of the FastText model. Defaults to "english".

    Returns:
        dict: fasttext model as a dictionary.

    """
    embeddings_dictionary = dict()
    if language == "english":
        file="embeddings\fasttext_eng.bin"
        ft_model = fasttext.load_model(file)
        for word in ft_model.get_words():
            embeddings_dictionary[word] = ft_model.get_word_vector(word)
    else:
        file="embeddings\fasttext_eng.bin"
        ft_model = fasttext.load_model(file)
        for word in ft_model.get_words():
            embeddings_dictionary[word] = ft_model.get_word_vector(word)

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
    num_words = len(tokenizer.word_index) + 1

    embedding_matrix = zeros((num_words, embedding_dim)) 
    for word, i in tokenizer.word_index.items():  
        embedding_vector = embeddings_index.get(word)  
        if embedding_vector is not None:  
            embedding_matrix[i] = embedding_vector  
    return embedding_matrix  