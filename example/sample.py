# -*- coding: UTF-8 -*-
"""
===============================================================
author：xieqiqi
email：xieqiqi@jd.com
date：2018
introduction:
            Language Model for Chinese poetry based on Lstm!
===============================================================
"""
import pickle
from example.utils import *

model = torch.load('poetry-gen.pt')
max_length = 100

with open('wordDic', "rb") as fr:
    word_to_ix = pickle.load(fr)

def invert_dict(word_to_ix):
    return dict((word_to_ix[k], k) for k in word_to_ix)

ix_to_word = invert_dict(word_to_ix)


# Sample from a category and starting letter
def sample(startWord='<START>'):
    input = make_one_hot_vec_target(startWord, word_to_ix)
    hidden = None
    output_name = ""
    if (startWord != "<START>"):
        output_name = startWord
    for i in range(max_length):
        #
        if torch.cuda.is_available():
            input=input.cuda()
        #
        output, hidden = model(input, hidden)
        topv, topi = output.data.topk(1)
        topi = topi[0][0]
        w = ix_to_word[topi]
        if w == "<EOP>":
            break
        else:
            output_name += w
        input = make_one_hot_vec_target(w, word_to_ix)
    return output_name

#
print (sample("春"))
print (sample("花"))
print (sample("秋"))
print (sample("月"))
print (sample("夜"))
print (sample("山"))
print (sample("水"))
print (sample("葉"))