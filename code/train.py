import tensorflow.keras as K
from tensorflow.keras.optimizers import Adagrad
import numpy as np
import model


batch_size = 256
epochs = 10
steps = 1000
embedding_length = 256
hidden_size = 256
optimizer = Adagrad()


def batch_generator(unigrams, bigrams, labels, batch_size):
    """ Generates batches of data to use when training the model. These include unigrams and bigrams, as well as
    one-hot encoded labels.

    :param unigrams: train set split into unigrams (list of lists of int)
    :param bigrams: train set split into bigram (list of lists of int)
    :param labels: one-hot encoded labels (list of lists of one-hot encodings)
    :param batch_size: number of training samples in forward/backward pass (int)
    :return:
    """
    while True:
        perm = np.random.permutation(len(labels))

        for start in range(0, len(unigrams), batch_size):
            end = start + batch_size
            yield [unigrams[perm[start:end]], bigrams[perm[start:end]]], labels[perm[start:end]]


def save_model_info(k_model, path_model=''):
    """ Saves the model (json format), its summary (.txt format) and its weights (.h5 format) for later use

    :param k_model: trained model (tf.keras.Model type)
    :param path_model: where to save the model (str)
    :return: None
    """

    with open(path_model+'model_summary.txt', 'w') as summary:
        k_model.summary(print_fn=lambda x: summary.write(x + '\n'))

    #  convert the model into json format
    model_json = k_model.to_json()
    with open(path_model+'model.json', 'w') as json_file:
        json_file.write(model_json)

    # save model weights
    k_model.save_weights(path_model+'model_weights.h5')


def train_model(vb_size_unigram, vb_size_bigram, train_x_unigram, train_x_bigram, train_y, val_x_unigram, val_x_bigram, val_y):
    """ Trains the Bidirectional LSTM using the specified parameters and saves the results

    :param vb_size_unigram: size of the vocabulary of unigrams (int)
    :param vb_size_bigram: size of the vocabulary of bigrams (int)
    :param train_x_unigram: train set of unigrams (list of list of int)
    :param train_x_bigram: train set of bigrams (list of list of int)
    :param train_y: one-hot encoded labels of the train set (list of list of one-hot encodings)
    :param val_x_unigram: unigram validation set (list of list of int)
    :param val_x_bigram: bigram validation set (list of list of int)
    :param val_y: one-hot encoded labels of the validation set (list of list of one-hot encodings)
    :return: trained model (tf.keras.Model type)
    """

    keras_model = model.create_keras_model(vocab_size_unigrams=vb_size_unigram, vocab_size_bigrams=vb_size_bigram,
                                        embedding_unigrams=embedding_length, embedding_bigrams=embedding_length, hidden_size=hidden_size)

    keras_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # generate batches of data for the training and validation sets
    data_gen = batch_generator(unigrams=train_x_unigram, bigrams=train_x_bigram, labels=train_y, batch_size=batch_size)
    val_gen = batch_generator(unigrams=val_x_unigram, bigrams=val_x_bigram, labels=val_y, batch_size=batch_size)

    # callbacks (tensorboard)
    cbk = K.callbacks.TensorBoard('logs/')

    # train the model
    history = keras_model.fit_generator(data_gen, steps_per_epoch=steps, epochs=epochs, validation_data=([val_x_unigram, val_x_bigram], val_y), callbacks=[cbk])

    # save a summary, model and weights
    save_model_info(keras_model)

    return history
