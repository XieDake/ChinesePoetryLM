# -*- coding: UTF-8 -*-
"""
===============================================================
author：XieDake
email：DakeXqq@126.com
date：2018
introduction:
            Chinese poetry LM based on Lstm!
            No miniBatch！
===============================================================
"""
import torch
import torch.nn.functional as F

class Poetry_LM(torch.nn.Module):
    def __init__(self,vocab_size, embedding_dim, hidden_dim):
        '''
        初始化！
        '''
        super(Poetry_LM,self).__init__()
        # Base parameters and structures!
        self.vocab_size=vocab_size
        self.embedding_dim=embedding_dim
        self.hidden_dim=hidden_dim
        # embedding——>lstm——>outPutLayer(relu?貌似没必要啊！)——>softmax!
        self.embeddings=torch.nn.Embedding(self.vocab_size,self.embedding_dim)
        self.lstm=torch.nn.LSTM(self.embedding_dim,self.hidden_dim)
        self.out=torch.nn.Linear(self.hidden_dim,self.vocab_size)
        self.softmax=torch.nn.LogSoftmax()

    def forward(self,seq_input,hidden=None):
        '''
        非batch版本，每一次input是一个sequence！输出是一个sequence的预测！
        注意：非batch的情况下,不需要统一sequence长度。所以，时间step=1或者step=任意长度是一样的！
        seq_inpput:[4]
        '''
        H=seq_input.size(0)
        #
        input_embed=self.embeddings(seq_input)# H*embe_dim
        input_embed=input_embed.unsqueeze(1)# H*1*embe_dim
        # outPut:[H,B=1,hidden_dim]
        # hidden:[1,B=1,hidden_dim]
        outPut,hidden=self.lstm(input_embed,hidden)
        #outPut:[H,B=1,hidden_dim]-->[H,B=1,vocab_size]
        outPut=outPut.view(H, -1)
        outPut = self.out(outPut)
        # 加激活函数？sigmoid或者relu！
        outPut=F.relu(outPut)
        # softmax
        outPut=self.softmax(outPut)
        # outPut:[H,vocab_size]
        return outPut,hidden
