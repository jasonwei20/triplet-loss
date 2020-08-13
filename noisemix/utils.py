# After randomly initing those lists with 100 random numbers in the correct range
from random import choices, randrange

RAND_INDICES={}
INDEX_LENGTH=100
LONGEST_WORD_LENGTH=27
POS = {};
def rand_init(val=0):
    if(val==0):
        for i in range(2, LONGEST_WORD_LENGTH):
            RAND_INDICES[i]=choices(range(0,i), k=INDEX_LENGTH)
            POS[i]=0
    else:
        RAND_INDICES[val] = choices(range(0, val), k=INDEX_LENGTH)
        POS[val] = 0
    return

def rand_index(word):
    length=len(word)
    if length<=1:
        return 0
    # Get the current position in the list of random indices for words of length n
    if length in POS:
        pos = POS[length]
    else:
        rand_init(length)
        pos = POS[length]
    # Get the index from that position in the list
    index = RAND_INDICES[length][pos]

    # Update the position in the list of random indices
    if pos + 1 == INDEX_LENGTH:
        POS[length] = randrange(INDEX_LENGTH) # Another option would be to init this with a random number between 0 and 99.
    else:
        POS[length] = pos + 1
    return index


import operator
from math import sqrt, ceil

def transform_keys (keyboard):
    temp={}
    for row,rowS in zip(keyboard["uppercase"], keyboard["lowercase"]):
        y1 =keyboard["uppercase"].index(row)
        for letter, letterS in zip(row, rowS):
            if letter == '':
                continue
            x1=row.index(letter)
            temp[letter]={}
            temp[letterS]={}
            for row2, row2S in zip(keyboard["uppercase"],keyboard["lowercase"]):
                y2= keyboard["uppercase"].index(row2)
                for letter2, letter2S in zip(row2, row2S):
                    if letter2 == '' or letter==letter2:
                        continue
                    x2 = row2.index(letter2)
                    dist=ceil(sqrt(pow((y1 - y2), 2) + pow((x1 - x2), 2)))
                    temp[letterS][letter2S]=dist
                    temp[letter][letter2] =dist
            temp[letterS] = sorted(temp[letterS].items(), key=operator.itemgetter(1))
            temp[letter]=sorted(temp[letter].items(), key=operator.itemgetter(1))
    return temp