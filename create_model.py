from tensorflow.keras.layers import Input, LSTM, Embedding, Dense  
from tensorflow.keras.models import Model  

def create_seq2seq_wordlevel_model(name,embedding_matrix_encoder,source_vocab_size, target_vocab_size, max_len_encoder, max_len_decoder, embedding_dim, latent_dim=256):  
    # Encoder  
    encoder_inputs = Input(shape=(max_len_encoder,),name="encoder_input")  #length of padded sequences for source
    encoder_embedding = Embedding(input_dim=source_vocab_size,   
                                  output_dim=embedding_dim,  #dimension of the embeddings (e.g. 100 for GloVe.6B.100d.txt)
                                  weights=[embedding_matrix_encoder],   
                                  name="encoder_embedding")(encoder_inputs)  
    encoder_lstm = LSTM(latent_dim, return_state=True, name="encoder_lstm")  
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)  #the encoder LSTM returns the final hidden state and cell state of the last time step
    encoder_states = [state_h, state_c] #the encoder states saves the hidden state and cell state of the last time step 
  
    # Decoder  
    decoder_inputs = Input(shape=(max_len_decoder,), name="decoder_input")  #length of padded sequences for target
    decoder_embedding = Embedding(input_dim=target_vocab_size,   
                                  output_dim=256,     
                                  name="decoder_embedding")(decoder_inputs)  
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")  #obtain decoder outputs for each time step
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)  #the decoder LSTM returns the full output sequence 
    decoder_dense = Dense(target_vocab_size, activation='softmax', name="decoder_dense")  #probability distribution over the vocabulary
    decoder_outputs = decoder_dense(decoder_outputs)  #produces the output sequence
  
    # Define the model  
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name=name)  
  

    #since the output of the decoder has to predict the next character from all the available target characters, it becomes a multiclass classification.
    return model  
