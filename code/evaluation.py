from tensorflow.keras.models import model_from_json
import numpy as np
import math
from itertools import chain


def load_model(model_path):
    """ Loads the model and its weights.

    :param model_path: model path (str)
    :return: model (tf.keras.Model)
    """

    with open(model_path+'model.json', 'r') as file_json:
        model_json = file_json.read()

    model = model_from_json(model_json)

    model.load_weights(model_path+'model_weights.h5')

    return model


def sentences_info(dataset_sentence_length, vocab_maximum_length):
    """ Retrieves information regarding the length of the sentences of the input file.
    It checks each sentence length against the maximum length of the train set to find
    those whose length exceeds the maximum allowed. It then saves the sentences' indices
    and the amount by which they exceed the maximum length.

    :param dataset_sentence_length: list of each input dataset's sentence length (list)
    :param vocab_maximum_length: maximum length of the vocabulary used in the model (int)
    :return: indices of the longest sentences in the input dataset (list), number of
            sub-sentences the original sentence will be split into
    """

    indices_longest_sentences = []
    number_sub_sentences = []

    for idx, length in enumerate(dataset_sentence_length):

        if length > vocab_maximum_length:
            num_sub_sentences = math.ceil(length/vocab_maximum_length)

            indices_longest_sentences.append(idx)
            number_sub_sentences.append(num_sub_sentences)

    return indices_longest_sentences, number_sub_sentences


def split_list(sentence, vocab_maximum_length):
    """ Splits a sentence into as many lists as needed for it to be shorter than or equal to
    the maximum length of the model's training input sentences.

    :param sentence: sentence exceeding the maximum length (list)
    :param vocab_maximum_length: maximum length of the model's input sentences (int)
    :return: generator
    """
    for idx in range(0, len(sentence), vocab_maximum_length):
        yield sentence[idx:idx+vocab_maximum_length]


def sentences_to_concat(indices_longest_sentences, sentence_extra_length):
    """ Obtains the indices of the lines in the prediction that need to be concatenated
    because they are part of the same (input) sentence.

    :param indices_longest_sentences: list of indices of the longest sentences (list)
    :param sentence_extra_length: number of sub-lists each sentence is split into (list)
    :return: list of indices of all sub-sentences (list of tuples)
    """
    sentences = []
    count = 0

    for idx, element in enumerate(indices_longest_sentences):

        if element == 0:
            tup = np.arange(sentence_extra_length[element])
            sentences.append(tup)
            count = tup[-1]

        elif idx == 0 and element != 0:
            tup = np.arange(element, element + sentence_extra_length[idx])
            sentences.append(tup)
            count = tup[-1]

        else:
            updated_idx = count + element - indices_longest_sentences[idx - 1]
            tup = np.arange(updated_idx, updated_idx + sentence_extra_length[idx])
            sentences.append(tup)
            count = tup[-1]

    return sentences


def prediction_to_bies(sentence):
    """ Converts the predictions into the BIES format.

    :param sentence: sentence to be converted (list)
    :return: sentence in the BIES format (list)
    """
    labels = {0: 'B', 1: 'I', 2: 'E', 3: 'S'}

    prediction_to_int = list(map(lambda pred: np.argmax(pred), sentence))
    bies_format = [labels[category] for category in prediction_to_int]

    return bies_format


def concatenate_sentences(predictions, sentences_to_concat):
    """ Concatenate all sub-sentences in order to recover the predictions of the full sentence.

    :param predictions: model's predictions (array)
    :param sentences_to_concat: list of indices of all sub-sentences (list of tuples)
    :return: predictions (array)
    """

    for sentence in sentences_to_concat[::-1]:
        merged_sentences = list(chain.from_iterable(predictions[sentence[0]:(sentence[1] + 1)]))
        predictions[sentence[0]:(sentence[1] + 1)] = [merged_sentences]

    return predictions


def delete_padding(length_of_sentence, predictions):
    """ Deletes the padding according to the (true) length of each sentence in the set.

    :param length_of_sentence: length of each sentence in the test set (list)
    :param predictions: model's predictions (array)
    :return: None
    """
    for idx, sentence in zip(length_of_sentence, predictions):
        del sentence[idx:]
