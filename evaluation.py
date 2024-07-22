from nltk.translate.bleu_score import sentence_bleu
import numpy as np  
import pandas as pd
import nltk
import pandas as pd
from tqdm import tqdm
import evaluate
meteor = evaluate.load('meteor')
  

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


def translate_dataset(test_eng_seq, test_eng, test_pt, encoder_model, decoder_model, idx2word_target, pt_max_len, word2idx_outputs, save_interval=100, save_path='translations/translation_evaluation.csv'):
    """
    Evaluates the translation for the entire test dataset and prepares a dataset with the source sentence, reference translation and generated translation.
    
    Parameters:
    - test_eng_seq: The test dataset sentences as sequenced inputs.
    - test_eng: List of source sentences in English.
    - test_pt: List of reference translations in Portuguese.
    - encoder_model: The trained Encoder model.
    - decoder_model: The trained Decoder model.
    - idx2word_target: Index-to-word mapping for the target language (Portuguese).
    - pt_max_len: Maximum length of the target sequence (Portuguese).
    - word2idx_outputs: Word-to-index mapping for the target language (Portuguese).
    - save_interval: Number of sentences after which to save the progress.
    - save_path: Path to the file where results will be saved.

    Returns:
    - evaluation_df: A DataFrame containing source sentences, reference translations, and generated translations.
    """
    translations = []

    # Translate each sentence in the test dataset
    for i in tqdm(range(len(test_eng_seq)), desc="Translating sentences"):
        input_seq = test_eng_seq[i:i+1]
        # Translate the current sentence
        translation = translate_sentence(input_seq, encoder_model, decoder_model, idx2word_target, pt_max_len, word2idx_outputs)

        # Append the results
        translations.append({
            'Source Sentence': test_eng[i],
            'Reference Translation': test_pt[i],
            'Model Translation': translation
        })

        # Save progress periodically
        if (i + 1) % save_interval == 0:
            temp_df = pd.DataFrame(translations)
            temp_df.to_csv(save_path, index=False)
    
    # Convert the results to a DataFrame
    evaluation_df = pd.DataFrame(translations)
    # Save the final DataFrame
    evaluation_df.to_csv(save_path, index=False)

    return evaluation_df


def calculate_bleu_score(candidate_translation, reference_translations):
    """
    Calculate BLEU score for a single translation.

    Parameters:
    - candidate_translation: The generated translation.
    - reference_translations: A list of reference translations.

    Returns:
    - bleu_score_value: The calculated BLEU score.
    """
    # Tokenize the candidate and reference translations
    candidate_tokens = candidate_translation.split()
    reference_tokens = [ref.split() for ref in reference_translations]  # Tokenize each reference
    smoothing_function = nltk.translate.bleu_score.SmoothingFunction().method1

    # Calculate the Blue Score
    bleu_score_value = nltk.translate.bleu_score.sentence_bleu(
        reference_tokens, candidate_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing_function
    )

    return bleu_score_value



def calculate_meteor_score(candidate_translation, reference_translation):
    """
    Calculate METEOR score for a single translation.

    Parameters:
    - candidate_translation: The generated translation.
    - reference_translation: The reference translation.

    Returns:
    - meteor_score_value: The calculated METEOR score.
    """
    # Calculate METEOR score
    meteor_score_value = meteor.compute(predictions=[candidate_translation], references=[reference_translation])
    return meteor_score_value["meteor"]

def calculate_scores(evaluation_df):
    """
    Calculate BLEU and METEOR scores for all translations in the evaluation dataset.

    Parameters:
    - evaluation_df: A DataFrame containing source sentences, reference translations,
                      and generated translations.

    Returns:
    - evaluation_df: The DataFrame with additional columns for BLEU, BERT, and METEOR scores.
    """
    bleu_scores = []
    meteor_scores = []

    for i in range(len(evaluation_df)):
        candidate_translation = evaluation_df.loc[i, 'Model Translation']
        reference_translation = evaluation_df.loc[i, 'Reference Translation']

        # Calculate BLEU score (assuming a single reference translation per candidate)
        bleu_score_value = calculate_bleu_score(candidate_translation, [reference_translation])
        bleu_scores.append(bleu_score_value)

        # Calculate METEOR score
        meteor_score_value = calculate_meteor_score(candidate_translation, reference_translation)
        meteor_scores.append(meteor_score_value)


    evaluation_df['BLEU Score'] = bleu_scores
    evaluation_df['METEOR Score'] = meteor_scores

    return evaluation_df

