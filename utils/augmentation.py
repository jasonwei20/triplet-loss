import random
from random import shuffle
random.seed(1)
from pathlib import Path
from utils import common
from tqdm import tqdm

from pybacktrans import BackTranslator
translator = BackTranslator()

#stop words list
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 
            'ours', 'ourselves', 'you', 'your', 'yours', 
            'yourself', 'yourselves', 'he', 'him', 'his', 
            'himself', 'she', 'her', 'hers', 'herself', 
            'it', 'its', 'itself', 'they', 'them', 'their', 
            'theirs', 'themselves', 'what', 'which', 'who', 
            'whom', 'this', 'that', 'these', 'those', 'am', 
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 
            'have', 'has', 'had', 'having', 'do', 'does', 'did',
            'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
            'because', 'as', 'until', 'while', 'of', 'at', 
            'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after', 
            'above', 'below', 'to', 'from', 'up', 'down', 'in',
            'out', 'on', 'off', 'over', 'under', 'again', 
            'further', 'then', 'once', 'here', 'there', 'when', 
            'where', 'why', 'how', 'all', 'any', 'both', 'each', 
            'few', 'more', 'most', 'other', 'some', 'such', 'no', 
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
            'very', 's', 't', 'can', 'will', 'just', 'don', 
            'should', 'now', '']

#cleaning up text
import re
def get_only_chars(line):

    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

# def random_swap(words, n):
#     new_words = words.copy()
#     for _ in range(n):
#         new_words = swap_word(new_words)
#     return new_words

# def swap_word(new_words):
#     random_idx_1 = random.randint(0, len(new_words)-1)
#     random_idx_2 = random_idx_1
#     counter = 0
#     while random_idx_2 == random_idx_1:
#         random_idx_2 = random.randint(0, len(new_words)-1)
#         counter += 1
#         if counter > 3:
#             return new_words
#     new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
#     return new_words

# def get_swap_sentence(sentence, alpha):
    
#     sentence = get_only_chars(sentence)
#     words = sentence.split(' ')
#     words = [word for word in words if word is not '']
#     num_words = len(words)

#     n_rs = max(1, int(alpha*num_words))
#     a_words = random_swap(words, n_rs)
#     augmented_sentence = ' '.join(a_words)

#     return augmented_sentence

# def get_swap_sentences(sentences, alpha=0.1):

#     return [get_swap_sentence(sentence, alpha) for sentence in sentences]

########################################################################
# synonym replacement
########################################################################

from nltk.corpus import wordnet 

def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            #print("replaced", random_word, "with", synonym)
            num_replaced += 1
        if num_replaced >= n: #only replace up to n words
            break

    #this is stupid but we need it, trust me
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word): 
        for l in syn.lemmas(): 
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym) 
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def get_sr_sentence(sentence, alpha=0.1):
    
    sentence = get_only_chars(sentence)
    words = sentence.split(' ')
    words = [word for word in words if word is not '']
    num_words = len(words)

    n_sr = max(1, int(alpha*num_words))
    a_words = synonym_replacement(words, n_sr)
    augmented_sentence = ' '.join(a_words)

    return augmented_sentence

def get_sr_data_dict(pkl_path, train_path, n_aug, alpha):
    
    if not pkl_path.exists():
        
        print(f"creating {pkl_path}")

        sentences, _ = common.get_sentences_and_labels_from_txt(train_path)

        sentence_to_augmented_sentences = {}
        for sentence in tqdm(sentences):
            sr_sentences = [get_sr_sentence(sentence, alpha) for _ in range(n_aug)]
            sentence_to_augmented_sentences[sentence] = sr_sentences

        common.save_pickle(pkl_path, sentence_to_augmented_sentences)
    
    return common.load_pickle(pkl_path)

def get_synonym_replacement_sentences(train_path, n_aug, alpha):

    pkl_path = Path(train_path).parent.joinpath(f"train_aug_sr_alpha{alpha:.2f}_data.pkl")
    sentence_to_aug_sentences = get_sr_data_dict(pkl_path, train_path, n_aug, alpha)
    return sentence_to_aug_sentences

########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words, p):

    #obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words

    #randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    #if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    return new_words

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words

def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
    return new_words

########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################

def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words

def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)

########################################################################
# main data augmentation function
########################################################################

def eda(sentence, alpha=0.1, num_aug=4):
    
    alpha_sr = alpha
    alpha_ri = alpha 
    alpha_rs = alpha 
    p_rd = alpha

    sentence = get_only_chars(sentence)
    words = sentence.split(' ')
    words = [word for word in words if word is not '']
    num_words = len(words)
    
    augmented_sentences = []
    num_new_per_technique = int(num_aug/4)+1
    n_sr = max(1, int(alpha_sr*num_words))
    n_ri = max(1, int(alpha_ri*num_words))
    n_rs = max(1, int(alpha_rs*num_words))

    #sr
    for _ in range(num_new_per_technique):
        a_words = synonym_replacement(words, n_sr)
        augmented_sentences.append(' '.join(a_words))

    #ri
    for _ in range(num_new_per_technique):
        a_words = random_insertion(words, n_ri)
        augmented_sentences.append(' '.join(a_words))

    #rs
    for _ in range(num_new_per_technique):
        a_words = random_swap(words, n_rs)
        augmented_sentences.append(' '.join(a_words))

    #rd
    for _ in range(num_new_per_technique):
        a_words = random_deletion(words, p_rd)
        augmented_sentences.append(' '.join(a_words))

    augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
    shuffle(augmented_sentences)

    #trim so that we have the desired number of augmented sentences
    augmented_sentences = augmented_sentences[:num_aug]

    return augmented_sentences

def get_eda_data_dict(pkl_path, train_path, n_aug, alpha):
    
    if not pkl_path.exists():
        
        print(f"creating {pkl_path}")

        sentences, _ = common.get_sentences_and_labels_from_txt(train_path)

        sentence_to_augmented_sentences = {}
        for sentence in tqdm(sentences):
            eda_sentences = eda(sentence, alpha=alpha, num_aug=n_aug)
            sentence_to_augmented_sentences[sentence] = eda_sentences

        common.save_pickle(pkl_path, sentence_to_augmented_sentences)
    
    return common.load_pickle(pkl_path)

def get_eda_sentences(train_path, n_aug, alpha):

    pkl_path = Path(train_path).parent.joinpath(f"train_aug_eda_alpha{alpha:.2f}_data.pkl")
    sentence_to_aug_sentences = get_eda_data_dict(pkl_path, train_path, n_aug, alpha)
    return sentence_to_aug_sentences

########################################################################
# backtranslation
########################################################################

# def backtrans_string(s):
#     result = translator.backtranslate(s, src='en', mid='fr')
#     return get_only_chars(result.text)

# def get_backtrans_data_dict(pkl_path, train_path, n_aug):

#     assert n_aug == 1
    
#     if not pkl_path.exists():
        
#         print(f"creating {pkl_path}")

#         sentences, _ = common.get_sentences_and_labels_from_txt(train_path)

#         sentence_to_augmented_sentences = {}
#         for sentence in tqdm(sentences):
#             sentence_to_augmented_sentences[sentence] = [backtrans_string(sentence)]

#         common.save_pickle(pkl_path, sentence_to_augmented_sentences)
    
#     return common.load_pickle(pkl_path)

# def get_backtrans_sentences(sentences, cfg):

#     pkl_path = Path(cfg.train_path).parent.joinpath(f"train_aug_backtrans_data.pkl")
#     sentence_to_aug_sentence = get_backtrans_data_dict(pkl_path, cfg.train_path)
#     return [sentence_to_aug_sentence[sentence] for sentence in sentences]

########################################################################
# master augment method that takes in cfg
########################################################################

def get_augmented_sentences(aug_type, train_path, n_aug, alpha):

    if aug_type == "sr":
        return get_synonym_replacement_sentences(train_path, n_aug, alpha)
    elif aug_type == "eda":
        return get_eda_sentences(train_path, n_aug, alpha)