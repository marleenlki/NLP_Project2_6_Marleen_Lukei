from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score # For METEOR
import numpy as np  
  
# Function to evaluate the model  
def calculate_bleu(model, X_test, y_test):  
    predictions = model.predict([X_test, y_test])  
    bleu_scores = []  
    for i in range(len(X_test)):  
        pred_sentence = np.argmax(predictions[i], axis=-1)  
        reference = y_test[i, 1:]  
        bleu_score = sentence_bleu([reference], pred_sentence)  
        bleu_scores.append(bleu_score)  
    return np.mean(bleu_scores)


def calculate_meteor(model, X_test, y_test, pt_tokenizer):
    """Evaluates the model using the METEOR score."""
    predictions = model.predict([X_test, y_test[:, :-1]])
    meteor_scores = []
    for i in range(len(X_test)):
        pred_sentence = np.argmax(predictions[i], axis=-1)
        reference = y_test[i, 1:]

        # Convert predictions and reference to words
        predicted_words = [pt_tokenizer.index_word.get(idx) for idx in pred_sentence if idx != 0]
        reference_words = [pt_tokenizer.index_word.get(idx) for idx in reference if idx != 0]

        meteor_score = single_meteor_score(reference_words, predicted_words)
        meteor_scores.append(meteor_score)
    return np.mean(meteor_scores)



def translate_sentence(input_seq, encoder_model, decoder_model, idx2word_target, pt_max_len,word2idx_outputs):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2idx_outputs['<start>']
    eos = word2idx_outputs['<end>']
    output_sentence = []

    for _ in range(pt_max_len):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        idx = np.argmax(output_tokens[0, 0, :])

        if eos == idx:
            break

        word = ''

        if idx > 0:
            word = idx2word_target[idx]
            output_sentence.append(word)

        target_seq[0, 0] = idx
        states_value = [h, c]

    return ' '.join(output_sentence)
