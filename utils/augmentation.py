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

def get_rd_sentence(sentence, alpha):
    
    sentence = get_only_chars(sentence)
    words = sentence.split(' ')
    words = [word for word in words if word is not '']
    num_words = len(words)

    a_words = random_deletion(words, alpha)
    augmented_sentence = ' '.join(a_words)

    return augmented_sentence

def get_rd_data_dict(pkl_path, train_path, n_aug, alpha):
    
    if not pkl_path.exists():
        
        print(f"creating {pkl_path}")

        sentences, _ = common.get_sentences_and_labels_from_txt(train_path)

        sentence_to_augmented_sentences = {}
        for sentence in tqdm(sentences):
            rd_sentences = [get_rd_sentence(sentence, alpha) for _ in range(n_aug)]
            sentence_to_augmented_sentences[sentence] = rd_sentences

        common.save_pickle(pkl_path, sentence_to_augmented_sentences)
    
    return common.load_pickle(pkl_path)

def get_rd_sentences(train_path, n_aug, alpha):

    pkl_path = Path(train_path).parent.joinpath(f"train_aug_rd_alpha{alpha:.2f}_data.pkl")
    sentence_to_aug_sentences = get_rd_data_dict(pkl_path, train_path, n_aug, alpha)
    return sentence_to_aug_sentences

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

def backtrans_string(s):
    result_fr = get_only_chars(translator.backtranslate(s, src='en', mid='fr').text)
    result_cn = get_only_chars(translator.backtranslate(s, src='en', mid='zh-cn').text)
    result_ar = get_only_chars(translator.backtranslate(s, src='en', mid='ar').text)
    result_es = get_only_chars(translator.backtranslate(s, src='en', mid='es').text)
    return [result_fr, result_cn, result_ar, result_es]

def get_backtrans_data_dict(pkl_path, train_path):
    
    if not pkl_path.exists():
        
        print(f"creating {pkl_path}")

        sentences, _ = common.get_sentences_and_labels_from_txt(train_path)

        sentence_to_augmented_sentences = {}
        for sentence in tqdm(sentences):
            sentence_to_augmented_sentences[sentence] = backtrans_string(sentence)

        common.save_pickle(pkl_path, sentence_to_augmented_sentences)
    
    return common.load_pickle(pkl_path)

def get_bt_sentences(train_path):

    pkl_path = Path(train_path).parent.joinpath(f"train_aug_bt_data.pkl")
    sentence_to_aug_sentence = get_backtrans_data_dict(pkl_path, train_path)
    return sentence_to_aug_sentence

########################################################################
# noisemix
########################################################################

from noisemix import noise

def noisemix_string(s):
    return [noise.randnoise(s) for _ in range(4)]

def get_noisemix_data_dict(pkl_path, train_path):

    if not pkl_path.exists():
        
        print(f"creating {pkl_path}")

        sentences, _ = common.get_sentences_and_labels_from_txt(train_path)

        sentence_to_augmented_sentences = {}
        for sentence in tqdm(sentences):
            augmented_sentences = noisemix_string(sentence)
            sentence_to_augmented_sentences[sentence] = augmented_sentences

        common.save_pickle(pkl_path, sentence_to_augmented_sentences)
    
    return common.load_pickle(pkl_path)

def get_nm_sentences(train_path):

    pkl_path = Path(train_path).parent.joinpath(f"train_aug_nm_data.pkl")
    sentence_to_aug_sentence = get_noisemix_data_dict(pkl_path, train_path)
    return sentence_to_aug_sentence

########################################################################
# misspellings
########################################################################

def load_mispellings_dict():
    file_path = 'data/common-misspellings.txt'
    mispellings_dict = {}
    lines = open(file_path, 'r').readlines()
    for line in lines:
        parts = line[:-1].split('->')
        correct_word = parts[1]
        misspelled_word = parts[0]
        mispellings_dict[correct_word] = misspelled_word
    return mispellings_dict

# mispellings_dict = load_mispellings_dict()

def get_mispelled_sentence(s):
    words = s.split(' ')
    candidates = set([word for word in words if word in mispellings_dict])
    if len(candidates) == 0:
        return s
    else:
        if random.uniform(0, 1) < len(candidates) / 4:
            chosen_candidate = random.sample(candidates, 1)[0]
            aug_words = []
            for word in words:
                aug_words.append(mispellings_dict[word] if word == chosen_candidate else word)
            return ' '.join(aug_words)
        else:
            return s

def get_misspelled_sentences(s):
    return [get_mispelled_sentence(s) for _ in range(4)]

def get_misspelled_data_dict(pkl_path, train_path):

    if not pkl_path.exists():
        
        print(f"creating {pkl_path}")

        sentences, _ = common.get_sentences_and_labels_from_txt(train_path)

        sentence_to_augmented_sentences = {}
        for sentence in tqdm(sentences):
            augmented_sentences = get_misspelled_sentences(sentence)
            sentence_to_augmented_sentences[sentence] = augmented_sentences

        common.save_pickle(pkl_path, sentence_to_augmented_sentences)
    
    return common.load_pickle(pkl_path)

def get_ms_sentences(train_path):

    pkl_path = Path(train_path).parent.joinpath(f"train_aug_ms_data.pkl")
    sentence_to_aug_sentence = get_misspelled_data_dict(pkl_path, train_path)
    return sentence_to_aug_sentence

########################################################################
# switchout
########################################################################

def load_all_words(sentences):
    all_words = set()
    for sentence in sentences:
        words = sentence.split(' ')
        for word in words:
            all_words.add(word)
    return all_words

def get_switchout_sentence(s, alpha, all_words):
    words = s.split(' ')
    aug_words = []
    for word in words:
        if random.uniform(0, 1) < alpha:
            aug_words.append(random.sample(all_words, 1)[0])
        else:
            aug_words.append(word)
    return ' '.join(aug_words)

def get_switchout_sentences(s, n_aug, alpha, all_words):
    return [get_switchout_sentence(s, alpha, all_words) for _ in range(n_aug)]

def get_switchout_data_dict(pkl_path, train_path, n_aug, alpha):

    if not pkl_path.exists():
        
        print(f"creating {pkl_path}")

        sentences, _ = common.get_sentences_and_labels_from_txt(train_path)
        all_words = load_all_words(sentences)

        sentence_to_augmented_sentences = {}
        for sentence in tqdm(sentences):
            augmented_sentences = get_switchout_sentences(sentence, n_aug, alpha, all_words)
            sentence_to_augmented_sentences[sentence] = augmented_sentences

        common.save_pickle(pkl_path, sentence_to_augmented_sentences)
    
    return common.load_pickle(pkl_path)

def get_so_sentences(train_path, n_aug, alpha):

    pkl_path = Path(train_path).parent.joinpath(f"train_aug_so_data.pkl")
    sentence_to_aug_sentence = get_switchout_data_dict(pkl_path, train_path, n_aug, alpha)
    return sentence_to_aug_sentence

########################################################################
# master augment method that takes in cfg
########################################################################

def get_augmented_sentences(aug_type, train_path, n_aug, alpha):

    if aug_type == "sr":
        return get_synonym_replacement_sentences(train_path, n_aug, alpha)
    elif aug_type == "eda":
        return get_eda_sentences(train_path, n_aug, alpha)
    elif aug_type == "rd":
        return get_rd_sentences(train_path, n_aug, alpha)
    elif aug_type == "bt":
        return get_bt_sentences(train_path)
    elif aug_type == "nm":
        return get_nm_sentences(train_path)
    elif aug_type == "ms":
        return get_ms_sentences(train_path)
    elif aug_type == "so":
        return get_so_sentences(train_path, n_aug, alpha)