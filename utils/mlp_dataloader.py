import itertools
import numpy as np
import random

def get_mlp_train_x_y(cfg, train_label_to_sentences, train_sentence_to_encoding):

    num_samples = len(list(itertools.chain.from_iterable(train_label_to_sentences.values())))

    train_x = np.zeros((num_samples, cfg.encoding_size))
    train_y = np.zeros((num_samples, ))

    i = 0
    for train_label, sentences in train_label_to_sentences.items():
        for sentence in sentences:
            train_x[i, :] = train_sentence_to_encoding[sentence]
            train_y[i] = train_label
            i += 1

    return train_x, train_y

def get_mlp_test_x_y(cfg, test_sentence_to_label, test_sentence_to_encoding):

    test_x = np.zeros((len(test_sentence_to_label), cfg.encoding_size))
    test_y = np.zeros((len(test_sentence_to_label), ))
    
    for i, (test_sentence, label) in enumerate(test_sentence_to_label.items()):
        test_x[i, :] = test_sentence_to_encoding[test_sentence]
        test_y[i] = label
    
    return test_x, test_y