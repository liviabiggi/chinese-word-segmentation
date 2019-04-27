import preprocess
import vocabulary as vb
import train
import json


main_folder= '../data/'


# Convert AS and CITYU to simplified Chinese
preprocess.to_simplified(main_folder+'as_training.utf8', output_path=main_folder+'as_simplified.utf8')
preprocess.to_simplified(main_folder+'cityu_training.utf8', output_path=main_folder+'cityu_simplified.utf8')


# Concatenate all four datasets
preprocess.concatenate_datasets(main_folder, main_folder+'as_simplified.utf8', main_folder+'cityu_simplified.utf8',
                                main_folder+'msr_training.utf8', main_folder+'pku_training.utf8')


# Check sentence length distribution
preprocess.plot_sentence_length(main_folder+'concat_dataset.utf8', main_folder, save_img=False)

# Remove duplicates in the dataset
preprocess.remove_duplicate_sentences(input_file=main_folder+'concat_dataset.utf8', output_file=main_folder+'final_dataset.utf8')

# Take a subset of the whole dataset (60%)
preprocess.subset_dataset(main_folder+'final_dataset.utf8', main_folder+'subset_dataset.utf8', perc=0.6)


# Create input file and labels
preprocess.create_train_set(main_folder+'subset_dataset.utf8', main_folder+'train_set.utf8', simplified=True)
preprocess.create_gold_file(main_folder+'subset_dataset.utf8', main_folder+'labels.txt', simplified=True)

# Split the dataset into train, validation and test sets
dataset = preprocess.read_file(main_folder+'train_set.utf8')
labels = preprocess.read_file(main_folder+'labels.txt')
train_x, test_x, val_x, *_ = preprocess.train_test_val_split(dataset, labels, save_datasets=True, save_labels=True, path_labels=main_folder)


# Generate padded dataset for unigrams
vocab_unigram, vocab_size_unigram = vb.generate_vocabulary(train_x)
max_length, padded_dataset_unigram = vb.pad_dataset(train_x, vocab_unigram)


# Generate padded dataset for bigrams
bigrams = vb.generate_bigrams(train_x)
vocab_bigram, vocab_size_bigram = vb.generate_vocabulary(bigrams, unigram=False, num_bigrams=10000)
_, padded_dataset_bigram = vb.pad_dataset(bigrams, vocab_bigram, compute_length=False, sentence_length=max_length)


# Obtain Inputs (unigram + bigram datasets) and Labels to be fed to the model for training
train_labels = vb.set_one_hot_labels(main_folder+'train_y.txt', max_length)
train_unigram = padded_dataset_unigram
train_bigram = padded_dataset_bigram


_, val_unigram = vb.pad_dataset(val_x, vocab_unigram, compute_length=False, sentence_length=max_length)
validation_bigram_set = vb.generate_bigrams(val_x)
_, val_bigram = vb.pad_dataset(validation_bigram_set, vocab_bigram, compute_length=False, sentence_length=max_length)
val_labels = vb.set_one_hot_labels(main_folder+'val_y.txt', max_length)


# Save the (train set) sentences' maximum length, the vocabulary of unigrams and that of the bigrams
reverse_vocab_bigram = {value : key for key, value in vocab_bigram.items()}

vocab_info = {}
vocab_info['maximum_length'] = max_length
vocab_info['vocabulary_unigram'] = vocab_unigram
vocab_info['vocabulary_bigram'] = reverse_vocab_bigram

with open(main_folder+'vocabulary_info.txt', 'w', encoding='utf-8') as json_file:
    json.dump(vocab_info, json_file, ensure_ascii=False, indent=2)


# Train the model using the specified parameters
history = train.train_model(vocab_size_unigram, vocab_size_bigram, train_unigram, train_bigram, train_labels, val_unigram, val_bigram, val_labels)
