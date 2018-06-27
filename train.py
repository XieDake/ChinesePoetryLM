# -*- coding: UTF-8 -*-
"""
===============================================================
author：XieDake
email：DakeXqq@126.com
date：2018
introduction:
===============================================================
"""
import torch
from torch.autograd import Variable
from data_helper import generate_one_sample

def train_one_epoch(epoch_num,model,optimizer,data_filter_train,batch_size,w2id):
    '''
    完成一个epoch数据训练！
    每一个batch进行一次BP！每一个Batch，print训练信息！
    注意：该函数执行之前data_filter需要乱序！
    '''
    data_size=len(data_filter_train)
    criterion = torch.nn.NLLLoss()
    for batchIndex in range(int(data_size / batch_size)):
        model.zero_grad()
        batch_loss= 0
        counts = 0
        for step in range(batchIndex * batch_size, min((batchIndex + 1) * batch_size, data_size)):
            input, real_out = generate_one_sample(sent=data_filter_train[step],w2id=w2id)
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

def eval_after_one_epoch(model,data_filter_val,w2id):
    '''
    每一个epoch结束，进行一次eval！
    '''
    loss=0.0
    data_size=len(data_filter_val)
    criterion=torch.nn.NLLLoss()
    for step in range(len(data_filter_val)):
        input, real_out = generate_one_sample(sent=data_filter_val[step],w2id=w2id)
        if torch.cuda.is_available():
            input = input.cuda()
            real_out = real_out.cuda()
        output, hidden = model(input)
        #
        loss += criterion(output, real_out)
    #
    return loss / data_size

def inference(input,model,max_length,i2Wd):
    '''

    '''
    hidden=None
    input = Variable(torch.LongTensor([input]))
    predict = ""
    for i in range(max_length):
        if (torch.cuda.is_available()):
            input = input.cuda()
        output, hidden = model(input,hidden)
        topv, topi = output.data.topk(1)
        topi = topi[0][0]
        w = i2Wd[topi]
        if w == "<EOS>":
            break
        else:
            predict += w
        #
        input = Variable(torch.LongTensor([topi]))
    #
    return predict
