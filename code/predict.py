from argparse import ArgumentParser
import preprocess
import vocabulary as vb
import evaluation as eval
import json
import unicodedata


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_path", help="The path of the input file")
    parser.add_argument("output_path", help="The path of the output file")
    parser.add_argument("resources_path", help="The path of the resources needed to load your model")

    return parser.parse_args()


def predict(input_path, output_path, resources_path):
    """ The predict function builds the model, loads the weights from the checkpoint and writes a new file (output_path)
    with the predictions in the BIES format.

    :param input_path: the path of the input file to predict.
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """

    with open(resources_path+'vocabulary_info.txt', 'r', encoding='utf-8') as json_file:
        file = json.load(json_file)

    max_length = file['maximum_length']
    vocab_unigram = file['vocabulary_unigram']
    vocab_bigram = {tuple(value): int(key) for key, value in file['vocabulary_bigram'].items()}

    # Build the model
    model = eval.load_model(resources_path)

    # Read the test set and convert potential full-width punctuation marks into half-width characters
    with open(input_path, 'r', encoding='utf-8') as file:
        test = file.readlines()
        test = list(map(lambda sentence: ' '.join(sentence.split()), test))
        test = list(map(lambda l: unicodedata.normalize('NFKC', l), test))
        test = list(map(list, test))

    # Save the length of each sentence in the test set
    sentence_length = list(map(len, test))

    # If there are sentences longer than the maximum sentence length allowed by the model
    # Split them into sub-sentences, obtain predictions for them and then concatenate them
    if list(filter(lambda x: x > max_length, sentence_length)):

        # Save index and length of test sentences that are longer than the maximum length allowed
        # And find the indices of all sub-sentences
        idx_longest_sentences, num_sub_sentences = eval.sentences_info(sentence_length, max_length)
        tups_to_concat = eval.sentences_to_concat(idx_longest_sentences, num_sub_sentences)

        # Create a dataset whose sentences are split according to their length
        sub_sentences = []
        for sentence in test:
            sentence = list(eval.split_list(sentence, max_length))
            sub_sentences += sentence

        # Define inputs to be predicted by the model (i.e. test set's unigrams & bigrams)
        _, test_unigram = vb.pad_dataset(sub_sentences, vocab_unigram, compute_length=False, sentence_length=max_length)
        test_bigram_set = vb.generate_bigrams(sub_sentences)
        _, test_bigram = vb.pad_dataset(test_bigram_set, vocab_bigram, compute_length=False, sentence_length=max_length)

        # Obtain model predictions, transform them into BIES format and concatenate them
        prediction = model.predict([test_unigram, test_bigram])
        bies_pred = list(map(lambda line: eval.prediction_to_bies(line), prediction))

        bies_pred = eval.concatenate_sentences(bies_pred, tups_to_concat)

    else:
        _, test_unigram = vb.pad_dataset(test, vocab_unigram, compute_length=False, sentence_length=max_length)
        test_bigram_set = vb.generate_bigrams(test)
        _, test_bigram = vb.pad_dataset(test_bigram_set, vocab_bigram, compute_length=False, sentence_length=max_length)

        # predict test set
        prediction = model.predict([test_unigram, test_bigram])
        bies_pred = list(map(lambda line: eval.prediction_to_bies(line), prediction))


    eval.delete_padding(sentence_length, bies_pred)
    preprocess.save_as_file(bies_pred, output_path)

    pass


if __name__ == '__main__':
    args = parse_args()
    predict(args.input_path, args.output_path, args.resources_path)
