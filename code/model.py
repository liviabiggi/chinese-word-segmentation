from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, Input, Embedding, Bidirectional, LSTM, TimeDistributed, Dense


def create_keras_model(vocab_size_unigrams, vocab_size_bigrams, embedding_unigrams, embedding_bigrams, hidden_size, input_dropout, rec_dropout):
    """ Creates a Bidirectional LSTM model for word segmentation using unigram and bigram features.

    :param vocab_size_unigrams: unigrams vocabulary size
    :param vocab_size_bigrams: bigrams vocabulary size
    :param embedding_unigrams: length of the unigrams embedding vector
    :param embedding_bigrams: length of the bigrams embedding vector
    :param hidden_size: hidden size of the model
    :return: tf.keras.Model
    """

    input_unigram = Input(shape=(None,), name='input_unigrams')
    input_bigram = Input(shape=(None,), name='input_bigrams')

    unigram_embedding = Embedding(vocab_size_unigrams, embedding_unigrams, mask_zero=True, name='unigram_embeddings')(input_unigram)
    bigram_embedding = Embedding(vocab_size_bigrams, embedding_bigrams, mask_zero=True, name='bigram_embeddings')(input_bigram)
    embeddings = concatenate([unigram_embedding, bigram_embedding], name='embedding_matrix')

    biLSTM = Bidirectional(LSTM(hidden_size, dropout=input_dropout, recurrent_dropout=rec_dropout, return_sequences=True), name='biLSTM', merge_mode='concat')(embeddings)
    biLSTM = TimeDistributed(Dense(4, activation='softmax'))(biLSTM)

    model = Model(inputs=[input_unigram, input_bigram], outputs=[biLSTM])

    return model
