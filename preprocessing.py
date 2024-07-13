import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer  
from tensorflow.keras.preprocessing.sequence import pad_sequences  

def apply_preprocessing_pipeline(english_sentences,portuguese_sentences):  
    # Lowercase the text  
    portuguese_sentences = [sentence.lower() for sentence in portuguese_sentences]  
    english_sentences = [sentence.lower() for sentence in english_sentences]
    # Remove special characters and numbers
    #portuguese_sentences = [re.sub(r'[^a-zà-ú\s]', '', sentence) for sentence in portuguese_sentences] 
    #english_sentences = [re.sub(r'[^a-z\s]', '', sentence) for sentence in english_sentences]
    # Remove sentences that start with <
    portuguese_sentences, english_sentences = zip(*[(pt, en) for pt, en in zip(portuguese_sentences, english_sentences) if not pt.startswith('<') and not en.startswith('<')]) 
    # Strip empty lines 
    portuguese_sentences, english_sentences = zip(*[(pt, en) for pt, en in zip(portuguese_sentences, english_sentences) if pt.strip() and en.strip()])  
    # Convert tuples back to lists  
    portuguese_sentences = list(portuguese_sentences)  
    english_sentences = list(english_sentences)  
    # Remove \n from the end of each sentence
    portuguese_sentences = [line.rstrip('\n') for line in portuguese_sentences]
    english_sentences = [line.rstrip('\n') for line in english_sentences]
    
    return english_sentences,portuguese_sentences

def tokenize(sentences, tokenizer=None ,decode=False):  
    if not tokenizer:  
        if decode:
            data = sentences + ['<start> <end>']
            tokenizer = Tokenizer(
                              filters = '',
                              num_words = 20000,
         
                            )  
        else:
            data = sentences
            tokenizer = Tokenizer(num_words = 20000,)  
        tokenizer.fit_on_texts(data)  
    # integer encode the sentences
    sequences = tokenizer.texts_to_sequences(sentences)
    return sequences, tokenizer

def pad(sequences, max_len=None):
    if not max_len:
        max_len = max(len(seq) for seq in sequences)
    # pad sequences to get the same length
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post') #where to truncate the sentence 
    return padded_sequences, max_len


