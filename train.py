# -*- coding: UTF-8 -*-
"""
===============================================================
author：xieqiqi
email：xieqiqi@jd.com
date：2018
introduction:
===============================================================
"""
import torch
from data_helper import generate_one_sample,sent2Id

def train_one_epoch(epoch_num,model,optimizer,data_filter_train,batch_size,w2id):
    '''
    完成一个epoch数据训练！
    每一个batch进行一次BP！每一个Batch，print训练信息！
    注意：该函数执行之前data_filter需要乱序！
    '''
    data_size=len(data_filter_train)
    criterion = torch.nn.NLLLoss()
    for batchIndex in range(int(data_size / batch_size)):
        optimizer.zero_grad()
        batch_loss= 0.0
        counts = 0
        for step in range(batchIndex * batch_size, min((batchIndex + 1) * batch_size, data_size)):
            sample_id = sent2Id(sent=data_filter_train[step],w2id=w2id)
            input, real_out = generate_one_sample(sample_id)
            if torch.cuda.is_available():
                input=input.cuda()
                real_out=real_out.cuda()
            output, hidden = model(input)
            #
            batch_loss += criterion(output,real_out)
            #
            counts += 1
        #
        print("At epoch:{},batch:{}——>loss_avg:{}".format({epoch_num}, {batchIndex},{batch_loss.data[0] / counts}))
        #
        batch_loss.backward()
        #
        optimizer.step()

def eval_after_one_epoch(model,data_filter_val):
    '''
    每一个epoch结束，进行一次eval！
    '''
    loss=0.0
    data_size=len(data_filter_val)
    for step in range(len(data_filter_val)):
        sample = data_filter_val[step]
        input, real_out = generate_one_sample(sample)
        if torch.cuda.is_available():
            input = input.cuda()
            real_out = real_out.cuda()
        output, hidden = model(input)
        #
        loss += torch.nn.NLLLoss(output, real_out)
    #
    return loss/data_size