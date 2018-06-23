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
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Poetry_LM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        '''
        初始化！
        '''
        super(Poetry_LM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim)
        #
        self.linear1 = nn.Linear(self.hidden_dim, vocab_size)
        # self.dropout = nn.Dropout(0.2)
        self.softmax = nn.LogSoftmax()

    def forward(self, input_x, hidden=None):
        '''
        非batch版本，每一次input是一个sequence的一个word！！
        '''
        length = input_x.size(0)
        embeds = self.embeddings(input_x).view((length, 1, -1))
        output, hidden = self.lstm(embeds, hidden)
        output = F.relu(self.linear1(output.view(length, -1)))
        # output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden