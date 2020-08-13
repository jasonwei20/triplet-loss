import string
from noisemix import utils

word_transforms = {}

word_transforms['en'] = [
    ['their', 'there'], ['its', "it's"],
    ['fourty', 'forty'], ['1', 'one'], # 2...
    ['and', '&', '+', 'n'],
    ['+', 'plus'], ['gray','grey'],
    ['%', 'percent'],
]

char_transforms = {}
char_transforms['en'] = [
    ['‘', '’', '“', '”',  "'", "'"],
    ['‐', '-', '‒', '–', '—', '―'],
    ['…', '...']
]

letter_transforms = {}
letter_transforms['en'] = list(string.ascii_letters)

keyboard_transforms={}

keyboard_transforms['en']={}
keyboard_transforms['en']['qwerty']={
    "uppercase":[["~", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "_", "+", ""],
                 ["", "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "{", "}", "|"],
                 ["", "A", "S", "D", "F", "G", "H", "J", "K", "L", ":", '"', "", ""],
                 ["", "Z", "X", "C", "V", "B", "N", "M", "<", ">", "?", "", "", ""]],
    "lowercase":[["`", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "=", ""],
                 ["", "q", "w", "e", "r", "t", "y", "u", "i", "o", "p", "[", "]", ""],
                 ["", "a", "s", "d", "f", "g", "h", "j", "k", "l", ";", "'", "", ""],
                 ["", "z", "x", "c", "v", "b", "n", "m", ",", ".", "/", "", "", ""]]
}
keyboard_transforms['en']['qwerty']=utils.transform_keys(keyboard_transforms['en']['qwerty'])
