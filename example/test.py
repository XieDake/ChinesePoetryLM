# -*- coding: UTF-8 -*-
"""
===============================================================
author：xieqiqi
email：xieqiqi@jd.com
date：2018
introduction:
===============================================================
"""
import pickle

with open(source_W2Id_saveName, 'wb') as fw:
    pickle.dump(source_word2index, fw)

