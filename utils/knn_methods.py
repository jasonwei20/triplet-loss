from utils import dataloader, bert_avgpool, triplet_models, visualization
import torch
import torch.optim as optim
from tqdm import tqdm
from scipy.spatial import distance
from pathlib import Path

def eval_model(
    train_sentence_to_label,
    train_label_to_sentences,
    train_sentence_to_encoding,
    test_sentence_to_label,
    test_sentence_to_encoding,
):
        
    def get_closest_train_sentence(test_sentence_encoding, train_sentence_to_encoding):
        train_sentence_to_dist_list = [ (train_sentence, distance.cosine(test_sentence_encoding, train_sentence_encoding)) for train_sentence, train_sentence_encoding in train_sentence_to_encoding.items()]
        sorted_train_sentence_dist_list = list(sorted(train_sentence_to_dist_list, key=lambda tup: tup[1]))
        return sorted_train_sentence_dist_list[0][0]

    num_correct = 0 #probably should be refactored
    for test_sentence, label in tqdm(test_sentence_to_label.items()):
        test_sentence_encoding = test_sentence_to_encoding[test_sentence]
        closest_train_sentence = get_closest_train_sentence(test_sentence_encoding, train_sentence_to_encoding)
        predicted_label = train_sentence_to_label[closest_train_sentence]
        if predicted_label == label:
            num_correct += 1
    
    acc = num_correct / len(test_sentence_to_label)
    return acc

def train_eval_model(
    cfg
):

    #load data
    train_sentence_to_label, train_label_to_sentences, test_sentence_to_label, train_sentence_to_encoding, test_sentence_to_encoding = dataloader.load_ap_data(cfg)

    val_acc = eval_model(
        train_sentence_to_label, 
        train_label_to_sentences, 
        train_sentence_to_encoding, 
        test_sentence_to_label, 
        test_sentence_to_encoding,
    )

    print(f"val_acc={val_acc:.3f}")

