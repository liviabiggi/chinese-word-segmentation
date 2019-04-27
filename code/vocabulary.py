from itertools import chain
from collections import OrderedDict, deque, Counter
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


def generate_vocabulary(data, unigram=True, num_bigrams=None):
    """ Generates the vocabulary for a given set of data (unigrams or bigrams).
    After defining the unique characters in the dataset, the function creates a vocabulary
    with a correspondence character-to-integer (index), and adds indices to the keys
    representing padding and out-of-vocabulary words.

    :param data: input dataset (list of lists of strings)
    :param unigram: decide whether the vocabulary should be built for unigrams or bigrams (default=True)
    :param num_bigrams: number of bigrams to add to the vocabulary (default=None)
    :return: vocabulary and vocabulary size (i.e. number of unique characters/keys in the vocabulary)
    """
    if unigram:
        # find the unique (single) characters in the dataset
        unique_char = list(OrderedDict.fromkeys(chain.from_iterable(data)))

    else:
        # find the most common bigrams in the dataset
        bigram_count = Counter(chain.from_iterable(data))
        most_common_bigrams = bigram_count.most_common(num_bigrams)

        unique_char = list(chain.from_iterable(most_common_bigrams))[::2]

    # create vocabulary of characters and indices
    vocabulary = {}
    for idx, char in enumerate(unique_char, 2):
        vocabulary[char] = idx

    vocabulary['<PAD>'] = 0
    vocabulary['<UNK>'] = 1

    vocabulary_size = len(unique_char)+2

    return vocabulary, vocabulary_size


def pad_dataset(data, vocabulary, compute_length=True, sentence_length=None):
    """ Adds padding to the sentences in the dataset in order to work with same-size sequences.
    The padding is added at the end of each sentence extending it such that its length will be
    equal to the longest sentence in the dataset. When the input dataset is the unigram dataset,
    the length of the longest sentence is used as a measure to pad the remaining sentences;
    when the input dataset is the bigram dataset, the length of the longest sentence in the
    unigram dataset is used as the max_sentence_length parameter

    :param data: input dataset (list of lists of strings)
    :param vocabulary: vocabulary containing word-to-index correspondence (dict)
    :param compute_length: computes the length of the longest sentence in the dataset (default=True)
    :param sentence_length: whether to use a pre-computed length as the sentence length (default=None)
    :return: length of the longest sentence, padded dataset
    """
    if compute_length:
        max_sentence_length = len(max(data, key=len))

    else:
        max_sentence_length = sentence_length

    # map characters to their respective index
    dataset_id = []
    for sentence in data:
        dataset_id.append([vocabulary[char] if char in vocabulary.keys() else 1 for char in sentence])

    # pad sentences
    padded_data = pad_sequences(dataset_id, truncating='pre', padding='post', maxlen=max_sentence_length)

    return max_sentence_length, padded_data


def set_one_hot_labels(gold_file, max_sentence_length):
    """ Obtains one-hot-encoded labels of the gold file. The latter is first padded according to
    the longest sentence in the dataset, and then transformed in its one-hot encodings.

    :param gold_file: path and name of the gold file (str)
    :param max_sentence_length: length of longest sentence in the dataset (int)
    :return: one-hot-encoded gold file (list of lists of lists)
    """
    with open(gold_file, 'r') as file:
        gold = file.read().splitlines()
        gold = list(map(list, gold))

    # substitute category labels with their respective numerical value
    labelled_set = list(map(lambda line: categorical_encodings(line), gold))

    # pad (label) sentences
    padded_labelled_set = pad_sequences(labelled_set, truncating='pre', padding='post', maxlen=max_sentence_length)

    one_hot_labels = to_categorical(padded_labelled_set, num_classes=4)

    return one_hot_labels


def categorical_encodings(sentence):
    """ Converts the input sentence in the BIES format into a list of (integer) classes according
    to the specified labels.

    :param sentence: input sentence (lst)
    :return: list of integers/classes (lst)
    """

    labels = {'B': 0, 'I': 1, 'E': 2, 'S': 3}

    label_set = [labels[category] for category in sentence]

    return label_set


def add_end_info(data):
    """ Adds end-of-sentence symbols (</s>) to each sentence
    in the dataset.

    :param data: input dataset (list of lists of strings)
    :return: modified dataset with start-of-sentence and end-of-sentence symbols (list of list of strings)
    """

    data = list(map(deque, data))

    # add end-of-sentence symbol
    for sentence in data:
        sentence.append('</s>')
    #   sentence.appendleft('<s>')

    data = list(map(list, data))

    return data


def generate_bigrams(data):
    """ Generate bigrams from the dataset containing information on beginning and ending of the
    individual sentences (i.e. with the addition of </s> symbols).

    :param data: input dataset containing </s> symbols (list of lists of strings)
    :return: dataset of bigrams
    """

    # add </s> symbols to the dataset
    data = add_end_info(data)

    # generate sets of bigrams for each sentence
    bigram_set = [zip(*[sentence[i:] for i in range(2)]) for sentence in data]
    bigram_set = list(map(list, bigram_set))

    return bigram_set
