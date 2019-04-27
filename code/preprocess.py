import re
import unicodedata
from hanziconv import HanziConv
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from collections import Counter


def format_input_file(file_path, simplified):
    """ Reads and configures the dataset into the correct input format:
    splits words and single characters and separates potential
    punctuation symbols from them

    :param file_path: input file path (str)
    :param simplified: specifies whether the input dataset is already in simplified Chinese (bool)
    :return: sentences of words and punctuation/symbols (list of lists of strings)
    """

    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        original_dataset = file.readlines()

        if simplified:
            # separate the dataset into sentences
            for line in original_dataset:
                line = line.split()

                #  separate symbols from words and single characters
                sentences = []
                for char in line:
                    char = unicodedata.normalize('NFKC', char)
                sentences += re.split('([\-~\'`《》【】?;:,。、!“”『』()「」‘’/])+', char, flags=re.UNICODE)

                # filter out empty strings
                sentences = list(filter(None,sentences))
                data.append(sentences)

        else:
            # separate the dataset into sentences
            for line in original_dataset:
                line = line.split()

                # convert the dataset into simplified Chinese
                line = list(map(HanziConv.toSimplified, line))

                #  separate symbols from words and single characters
                sentences = []
                for char in line:
                    char = unicodedata.normalize('NFKC', char)
                    sentences += re.split('([\-~\'`《》【】?;:,。、!“”『』()「」‘’/])+', char, flags=re.UNICODE)

                # filter out empty strings
                sentences = list(filter(None, sentences))
                data.append(sentences)

    return data
        

def create_train_set(input_file, train_path, simplified=True):
    """ Removes whitespaces from the formatted input data and generates train set file

    :param input_file: input file path (str)
    :param train_path: path where the train set will be saved (str)
    :param simplified: specifies whether the input dataset is already in simplified Chinese (default=True)
    :return: None
    """
    # convert the file into the right format
    data = format_input_file(input_file, simplified)
    save_as_file(data, train_path)


def create_gold_file(input_file, gold_path, simplified=True):
    """ Converts the formatted input file into the BIES format
    (Beginning - Inside - End - Single) and saves it in the chosen directory

    :param input_file: input file path (str)
    :param gold_path: path where the gold file will be saved (str)
    :param simplified: specifies whether the input dataset is already in simplified Chinese (bool, default=True)
    :return: gold file (.txt format)
    """

    data = format_input_file(input_file, simplified)

    with open(gold_path, 'w') as file:
        for line in data:
            for char in line:
                if len(char) == 1:
                    file.write('S')
                else:
                    file.write('B'+'I'*int(len(char)-2)+'E')
            file.write('\n')


def concatenate_datasets(output_file_path, *input_file_path):
    """ Concatenates datasets and saves them as a unique file called 'final_dataset.uft8'

    :param output_file_path: path where the output file should be saved (str)
    :param input_file_path: input dataset paths (as individual strings)
    :return: final dataset (.utf8 format)
    """

    filenames = [*input_file_path]
    with open(output_file_path+'concat_dataset.utf8', 'w', encoding='utf-8') as concat_dataset:
        for file in filenames:
            with open(file, 'r', encoding='utf-8') as small_dataset:
                concat_dataset.write(small_dataset.read())


def to_simplified(input_path, output_path, save_file=True):
    """ Converts the input file (still to be pre-processed) from traditional to simplified Chinese and saves it.

    :param input_path: path of the input file in traditional Chinese
    :param output_path: path where the output file (in simplified Chinese) will be saved
    :param save_file: whether to save the output file or simply use it as a local variable (bool, default=True)
    :return:
    """
    with open(input_path, 'r', encoding='utf-8') as file:
        data = file.readlines()
        data = list(map(lambda sentence: ' '.join(sentence.split()), data))
        data = list(map(HanziConv.toSimplified, data))
        data = list(map(lambda l: l.split(), data))

    if save_file:
        with open(output_path, 'w', encoding='utf-8') as file:
            data = map(lambda l: ' '.join(l) + '\n', data)
            file.writelines(data)

    return data


def remove_duplicate_sentences(input_file, output_file):
    """ Removes duplicate sentences in the input dataset by performing a set over all lines in the file and saves it.

    :param input_file: input file (potentially) containing duplicate sentences
    :param output_file: path where the output file without duplicates will be saved
    :return: None
    """
    with open(input_file, 'r', encoding='utf-8') as file:
        duplicate_data = file.readlines()
        duplicate_data = list(map(lambda sentence: sentence.split(), duplicate_data))

        # remove duplicate sentences from the dataset
        data = set(map(tuple, duplicate_data))

        # keep only the sentences that have at most 100 words
        data = list(filter(lambda sentence: len(sentence) <= 100, data))

    with open(output_file, 'w', encoding='utf-8') as file:
        data = map(lambda l: ' '.join(l) + '\n', data)
        file.writelines(data)


def subset_dataset(input_file, output_file, perc):
    """ Creates and saves a subset of the input dataset, given a percentage (0-1).

    :param input_file: (large) input dataset
    :param output_file: subset of the input dataset
    :param perc: number in (0,1]
    :return: None
    """
    assert perc > 0 and perc <= 1, 'Percentage must be between 0 and 1: (0,1]'

    with open(input_file, 'r', encoding='utf-8') as file:
        data = file.readlines()
        data = list(map(lambda l: l.split(), data))

    percentage = int(len(data)*perc)
    data = list(np.random.permutation(data))[:percentage]

    with open(output_file, 'w', encoding='utf-8') as file:
        subset = map(lambda l: ' '.join(l) + '\n', data)
        file.writelines(subset)


def save_as_file(dataset, file_path):
    """ Saves a dataset as a .utf8 file (without whitespace)

    :param dataset: data to be saved as a file (.utf8 format)
    :param file_path: path where the file will be saved (str)
    :return: None
    """

    with open(file_path, 'w', encoding='utf-8') as file:
        # remove whitespaces between words
        line = map(lambda l: ''.join(l)+'\n', dataset)
        file.writelines(line)


def read_file(file_path):
    """ Reads the input file and transforms it into a list of lists

    :param file_path: path of the input dataset/file
    :return: dataset as a list of lists
    """

    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read().splitlines()
        data = list(map(list, data))

    return data


def train_test_val_split(data, data_labels, save_datasets=False, save_labels=False, path_labels=None):
    """ Split the input dataset into train, test and validation sets

    :param data: input dataset
    :param data_labels: input labels
    :param save_datasets: whether to save the datasets as files (bool, default=False)
    :param save_labels: whether to save the labels as files (bool, default=False)
    :param path_labels: where to save the label files (str, default=None)
    :return: train, test and validation sets and their respective labels
    """

    train_x, test_x, train_y, test_y = train_test_split(data, data_labels, test_size=0.3, random_state=13)
    val_x, test_x, val_y, test_y, = train_test_split(test_x, test_y, test_size=0.33, random_state=13)

    if save_datasets:
        save_as_file(train_x, path_labels+'train_x.utf8')
        save_as_file(test_x, path_labels+'test_x.utf8')
        save_as_file(val_x, path_labels+'val_x.utf8')

    if save_labels:
        save_as_file(train_y, path_labels+'train_y.txt')
        save_as_file(test_y, path_labels+'test_y.txt')
        save_as_file(val_y, path_labels+'val_y.txt')

    return train_x, test_x, val_x, train_y, test_y, val_y


def plot_sentence_length(input_file, output_file, save_img=True):
    """ Plot the 100 most frequent sentence lengths of the input dataset. The words in the sentences may
    comprise single or multiple characters.

    :param input_file: where the input dataset is stored (str)
    :param output_file: destination path - where to save the image (str)
    :param save_img: whether to save the image (bool, default=True)
    :return: plot
    """
    with open(input_file, 'r', encoding='utf-8') as file:
        data = file.readlines()
        data = list(map(lambda sentence: sentence.split(), data))
        data = set(map(tuple, data))

    length_frequency = Counter(map(len, data))

    labels, values = zip(*sorted(length_frequency.most_common(100), key=itemgetter(0)))
    indexes = np.array(labels)
    width = 1
    plt.bar(indexes, values, width, color='#2EA1BC', alpha=0.8)
    plt.xlabel('Sentence length')
    plt.ylabel('Frequency')

    if save_img:
        plt.savefig(output_file+'sentence_length.png')

    plt.show()
