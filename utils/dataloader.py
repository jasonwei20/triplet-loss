import itertools
import numpy as np
import random
import torch
from utils import augmentation, bert_avgpool

#######################################
############ triplet stuff ############
#######################################

def get_sentence_to_label(csv_file):

    lines = open(csv_file, 'r').readlines()
    
    sentence_to_label = {}
    label_to_sentences = {}
    
    for line in lines:
        parts = line[:-1].split('\t')
        label = int(parts[0])
        sentence = parts[1]
        sentence_to_label[sentence] = label

        if label in label_to_sentences:
            label_to_sentences[label].append(sentence)
        else:
            label_to_sentences[label] = [sentence]

    return sentence_to_label, label_to_sentences

def get_train_subset(train_label_to_sentences, nc):
    return {train_label: random.sample(sentences, nc) for train_label, sentences in train_label_to_sentences.items()}

def generate_triplet(label_to_sentences):

    labels = label_to_sentences.keys()
    label_p, label_n = random.sample(labels, 2)
    anchor, pos = random.sample(label_to_sentences[label_p], 2)
    neg = random.sample(label_to_sentences[label_n], 1)[0]
    return anchor, pos, neg

def generate_triplet_batch(label_to_sentences, train_sentence_to_embedding, device, mb_size=64):

    anchor_list = []; pos_list = []; neg_list = []
    for _ in range(mb_size):
        anchor, pos, neg = generate_triplet(label_to_sentences)
        anchor_list.append(train_sentence_to_embedding[anchor])
        pos_list.append(train_sentence_to_embedding[pos])
        neg_list.append(train_sentence_to_embedding[neg])

    anchor_embeddings = torch.tensor(anchor_list)
    pos_embeddings = torch.tensor(pos_list)
    neg_embeddings = torch.tensor(neg_list)

    return anchor_embeddings.to(device), pos_embeddings.to(device), neg_embeddings.to(device)

def load_ap_data(cfg):

    if cfg.aug_type in {'sr', 'swap', 'bt'}:
        return load_ap_data_aug(cfg)
    else:
        return load_ap_data_no_aug(cfg)

def load_ap_data_no_aug(cfg):

    train_sentence_to_label, train_label_to_sentences = get_sentence_to_label(cfg.train_path)
    test_sentence_to_label, _ = get_sentence_to_label(cfg.test_path)

    train_sentence_to_encoding = bert_avgpool.get_encoding_dict(train_sentence_to_label, cfg.train_path, cfg.aug_type)
    test_sentence_to_encoding = bert_avgpool.get_encoding_dict(test_sentence_to_label, cfg.test_path, None)

    if cfg.train_nc:
        train_label_to_sentences = get_train_subset(train_label_to_sentences, cfg.train_nc)

    return train_sentence_to_label, train_label_to_sentences, test_sentence_to_label, train_sentence_to_encoding, test_sentence_to_encoding

def generate_aug_train_sentences(train_sentence_to_label, train_label_to_sentences, cfg):

    sentence_to_aug_sentences = augmentation.get_augmented_sentences(cfg)

    train_sentence_aug_to_label = {}; train_label_to_sentences_aug = {label: [] for label in train_label_to_sentences}

    for label, train_sentences in train_label_to_sentences.items():

        for train_sentence in train_sentences:

            train_sentence_aug_to_label[train_sentence] = label
            train_label_to_sentences_aug[label].append(train_sentence)

            aug_sentences = sentence_to_aug_sentences[train_sentence]

            for aug_sentence in aug_sentences:
                train_sentence_aug_to_label[aug_sentence] = label
                train_label_to_sentences_aug[label].append(aug_sentence)
        
    return train_sentence_aug_to_label, train_label_to_sentences_aug

def load_ap_data_aug(cfg):

    train_sentence_to_label, train_label_to_sentences = get_sentence_to_label(cfg.train_path)
    test_sentence_to_label, _ = get_sentence_to_label(cfg.test_path)

    train_sentence_aug_to_label, train_label_to_sentences_aug = generate_aug_train_sentences(train_sentence_to_label, train_label_to_sentences, cfg)

    train_sentence_to_encoding = bert_avgpool.get_encoding_dict(train_sentence_aug_to_label, cfg.train_path, cfg.aug_type)
    test_sentence_to_encoding = bert_avgpool.get_encoding_dict(test_sentence_to_label, cfg.test_path, None)

    if cfg.train_nc:
        train_label_to_sentences = get_train_subset(train_label_to_sentences, cfg.train_nc)
        _, train_label_to_sentences_aug = generate_aug_train_sentences(train_sentence_to_label, train_label_to_sentences, cfg)

    return train_sentence_aug_to_label, train_label_to_sentences_aug, test_sentence_to_label, train_sentence_to_encoding, test_sentence_to_encoding

#######################################
############## mlp stuff ##############
#######################################

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