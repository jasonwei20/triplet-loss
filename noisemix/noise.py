import string

import random
from datetime import datetime

random.seed(datetime.now())
import json
from noisemix import utils
from noisemix.perturbation import Perturbation

# add_letter,repeat_letter,remove_letter,lowercase,remove_punct,word_swap,char_swap
NOISE_TYPES =[]
SENTENCE_NOISE_TYPES=[]

def default_types(perturbation_list):
    result = []
    param = perturbation_list["ADD_LETTER"]
    if param["ENABLED"]:
        result.append(Perturbation(name="add_letter", function=add_letter,frequency=param["FREQUENCY"],repeat_num=param["REPEAT_NUM"] ))
    param = perturbation_list["REPEAT_LETTER"]
    if param["ENABLED"]:
        result.append(Perturbation(name="repeat_letter", function=repeat_letter,frequency=param["FREQUENCY"],repeat_num=param["REPEAT_NUM"]))
    param = perturbation_list["REMOVE_LETTER"]
    if param["ENABLED"]:
        result.append(Perturbation(name="remove_letter", function=remove_letter,frequency=param["FREQUENCY"],repeat_num=param["REPEAT_NUM"]))
    param = perturbation_list["LOWERCASE"]
    if param["ENABLED"]:
        result.append(Perturbation(name="lowercase", function=lowercase,frequency=param["FREQUENCY"],repeat_num=param["REPEAT_NUM"]))
    param = perturbation_list["REMOVE_PUNCT"]
    if param["ENABLED"]:
        result.append(Perturbation(name="remove_punct", function=remove_punct,frequency=param["FREQUENCY"],repeat_num=param["REPEAT_NUM"]))
    param = perturbation_list["WORD_SWAP"]
    if param["ENABLED"]:
        result.append(Perturbation(name="word_swap", function=word_swap,frequency=param["FREQUENCY"],repeat_num=param["REPEAT_NUM"]))
    param = perturbation_list["CHAR_SWAP"]
    if param["ENABLED"]:
        result.append(Perturbation(name="char_swap", function=char_swap,frequency=param["FREQUENCY"],repeat_num=param["REPEAT_NUM"]))
    param = perturbation_list["FLIP_LETTERS"]
    if param["ENABLED"]:
        result.append(Perturbation(name="flip_letters", function=flip_letters,frequency=param["FREQUENCY"],repeat_num=param["REPEAT_NUM"]))
    param = perturbation_list["TYPO_QWERTY"]
    if param["ENABLED"]:
        result.append(Perturbation(name="typo_qwerty", function=typo_qwerty,frequency=param["FREQUENCY"],repeat_num=param["REPEAT_NUM"]))
    return result


def sentence_types(perturbation_list):
    result = []
    param = perturbation_list["REMOVE_SPACE"]
    if param["ENABLED"]:
        result.append(Perturbation(name="remove_space", function=remove_space,frequency=param["FREQUENCY"],repeat_num=param["REPEAT_NUM"]))
    param = perturbation_list["FLIP_WORDS"]
    if param["ENABLED"]:
        result.append(Perturbation(name="flip_words", function=flip_words,frequency=param["FREQUENCY"],repeat_num=param["REPEAT_NUM"]))
    return result

with open('noisemix/config.json', 'r') as f:
    config = json.load(f)
LANGUAGE = config["LANGUAGE"]
KEYBOARD = config["KEYBOARD_LAYOUT"]
MAX_SENTENCE = int(config["SENTENCE_PERTURBATION"]["MAX_PERTURBATION"])
MAX_WORD =  int(config["WORD_PERTURBATION"]["MAX_PERTURBATION"])
MIN_LEN = int(config["WORD_PERTURBATION"]["MIN_LEN"])
SENTENCE_MIN_LEN = int(config["SENTENCE_PERTURBATION"]["MIN_LEN"])

def sentence_noise(sentence, p_max=MAX_SENTENCE):
    words = sentence.split(' ')
    for p in random.sample(SENTENCE_NOISE_TYPES, SENTENCE_MIN_LEN):
        repeat_num = p.repeat_num
        while True:
            if repeat_num == 0:
                break
            if len(words) >= SENTENCE_MIN_LEN:
                if random.random() < p.frequency:
                    words = p.function(words)
            repeat_num = repeat_num - 1
    sentence = (' ').join(words)
    return sentence

def remove_space(words):
    pos = random.randint(0, len(words) - 1)
    words[pos:pos + 2] = [''.join(words[pos:pos + 2])]
    return words

def flip_words(words):
    result = random.sample(words, 2)
    a, b = words.index(result[0]), words.index(result[1])
    words[b], words[a] = words[a], words[b]
    return words


def randnoise(s, p_max=MAX_WORD):
    """ Apply random noisification and return the new string """

    # The distribution is designed so that there will be fewer sentences with
    # more noisemix types.

    for p in random.sample(NOISE_TYPES, p_max):
        repeat_num = p.repeat_num
        while True:
            if repeat_num == 0:
                break
            # if random.random() < p.frequency:
            if len(s) >= MIN_LEN:
                s = p.function(s)
            repeat_num = repeat_num - 1
    return s


def _rand_pos(s):
    return utils.rand_index(s)


def flip_letters(s):
    i, j = _rand_pos(s), _rand_pos(s)
    # s[b], s[a] = s[a], s[b]
    if i > j:
        i, j = j, i  # swaps i and j
    s = s[:i] + s[j] + s[i + 1:j] + s[i] + s[j + 1:]
    return s


def repeat_letter(s):
    # print("addition change---before:", s)
    pos = _rand_pos(s)
    s = s[:pos] + s[pos] + s[pos:]
    # print("addition change---after:", s)
    return s


def remove_letter(s):
    # print("subtraction change---before:", word)
    pos = _rand_pos(s)
    s = s[:pos] + s[(pos + 1):]
    # print("subtraction change---after:", word)
    return s


def lowercase(s):
    # print("case change---before:", s)
    s = s.lower()
    # print("case change---after:", s)
    return s


punct_trans = str.maketrans('', '', string.punctuation)


def remove_punct(s):
    return s.translate(punct_trans)


from noisemix.data import word_transforms, char_transforms, keyboard_transforms, letter_transforms


def add_letter(s):
    # print("addition change---before:", s)
    pos = _rand_pos(s)
    addition = random.choice(letter_transforms[LANGUAGE])
    s = s[:pos] + addition + s[pos:]
    # print("addition change---after:", s)
    return s


def word_swap(s):
    for t in word_transforms[LANGUAGE]:
        if s in t:
            return random.sample(t, 1)[0]
    return s


def char_swap(s):
    for t in char_transforms[LANGUAGE]:
        for c in s:
            if c in t:
                return s.replace(c, random.sample(t, 1)[0])
    return s


def typo_qwerty(s):
    # https://stackoverflow.com/a/7621105/4486860
    pos = _rand_pos(s)
    if (s[pos] in keyboard_transforms[LANGUAGE][KEYBOARD]):
        neighborList = keyboard_transforms[LANGUAGE][KEYBOARD][s[pos]]
    else:
        return s
    tempList = []
    for key, value in neighborList:
        if value <= 2:
            tempList.append(key)
        else:
            break
    value = random.sample(tempList, 1)[0]
    s = s[:pos] + value + s[pos:]
    return s

# print("Noise types:", *map(lambda p: p.__name__, PERTURBATIONS))
NOISE_TYPES = default_types(config["WORD_PERTURBATION"])
SENTENCE_NOISE_TYPES = sentence_types(config["SENTENCE_PERTURBATION"])
