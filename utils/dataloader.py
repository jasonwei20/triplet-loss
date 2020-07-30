import random
import torch

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

def generate_triplet_batch(label_to_sentences, train_sentence_to_embedding, mb_size=64):

    anchor_list = []; pos_list = []; neg_list = []
    for _ in range(mb_size):
        anchor, pos, neg = generate_triplet(label_to_sentences)
        anchor_list.append(train_sentence_to_embedding[anchor])
        pos_list.append(train_sentence_to_embedding[pos])
        neg_list.append(train_sentence_to_embedding[neg])

    anchor_embeddings = torch.tensor(anchor_list)
    pos_embeddings = torch.tensor(pos_list)
    neg_embeddings = torch.tensor(neg_list)

    return anchor_embeddings, pos_embeddings, neg_embeddings