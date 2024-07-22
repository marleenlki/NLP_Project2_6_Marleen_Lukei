import re
import unicodedata
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer  
from tensorflow.keras.preprocessing.sequence import pad_sequences  


def remove_duplicate_translations(english_sentences, portuguese_sentences):
    """Removes duplicate translations from parallel lists of sentences."""
    unique_translations = dict()
    for en, pt in zip(english_sentences, portuguese_sentences):
        if en not in unique_translations and pt not in unique_translations.values():
            unique_translations[en] = pt
    return list(unique_translations.keys()), list(unique_translations.values())

def remove_parentheses_content(text):
  """Removes text enclosed in parentheses from a string."""
  cleaned_text = re.sub(r'\([^)]*\)', '', text)
  return cleaned_text

def normalize_string(s):
    """Normalizes a string by lowercasing, removing diacritics, removing punctuation and removing non-alphanumeric characters.
    """
    s = s.lower()
    # Split accented characters.
    s = unicodedata.normalize('NFKD', s)
    # Keep space, a to z
    s = re.sub(r"[^ a-z\s]+", r"", s) 
    s = re.sub(r'\s\s+', ' ', s).strip() # Remove multiple spaces
    return s.strip()

def apply_preprocessing_pipeline(english_sentences, portuguese_sentences):
    """Applies a preprocessing pipeline to English and Portuguese sentences."""
        # Split sentences  

    cleaned_english = []
    cleaned_portuguese = []
    

    for en_sent, pt_sent in zip(english_sentences, portuguese_sentences):
        # Remove sentences starting with '<'
        if en_sent.startswith('<') or pt_sent.startswith('<'):
            continue
        
        # Remove parentheses content
        en_sent = remove_parentheses_content(en_sent)
        pt_sent = remove_parentheses_content(pt_sent)
        
        # Normalize strings
        en_sent = normalize_string(en_sent)
        pt_sent = normalize_string(pt_sent)


        # Skip empty lines
        if en_sent.strip() and pt_sent.strip():
            cleaned_english.append(en_sent)
            cleaned_portuguese.append(pt_sent)
      
   
    # Remove duplicate translations
    cleaned_english, cleaned_portuguese = remove_duplicate_translations(cleaned_english, cleaned_portuguese)
    return cleaned_english, cleaned_portuguese

def tokenize(sentences, tokenizer=None ,decode=False):  
    if not tokenizer:  
        if decode:
            tokenizer = Tokenizer(
                              filters = '',
                              oov_token="[UNK]"
                            ) 
            tokenizer.fit_on_texts(sentences+ ['<start>', '<end>']) 
        else:
            tokenizer = Tokenizer(oov_token="[UNK]",filters = '')  
            tokenizer.fit_on_texts(sentences)  
    # integer encode the sentences
    sequences = tokenizer.texts_to_sequences(sentences)
    return sequences, tokenizer

def pad(sequences, max_len=None):

    # pad sequences to get the same length
    padded_sequences = pad_sequences(sequences, maxlen=max_len, truncating ="post",padding='post') #where to truncate the sentence 
    
    return padded_sequences


