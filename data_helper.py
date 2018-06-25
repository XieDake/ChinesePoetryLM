# -*- coding: UTF-8 -*-
"""
===============================================================
author：xieqiqi
email：xieqiqi@jd.com
date：2018
introduction:
===============================================================
"""
import os,re,json
import pickle,random
import torch
import numpy as np
from torch.autograd import Variable

def parseRawData(source_fileName,author = None, constrain = None):
    '''
    解析Json数据文件！
    '''
    def sentenceProcess(sents):
        '''
        数据预处理！
        '''
        #过滤掉特殊字符！
        result, number = re.subn("（.*）", "", sents)
        result, number = re.subn("{.*}", "", result)
        result, number = re.subn("《.*》", "", result)
        result, number = re.subn("《.*》", "", result)
        result, number = re.subn("[\]\[]", "", result)
        r = ""
        #过滤掉数字！
        for s in result:
            if s not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-']:
                r += s
        r, number = re.subn("。。", "。", r)
        return r

    def handleJson(file):
        '''
        解析Json!
        '''
        poetries = []
        data = json.loads(open(file).read())
        for poetry in data:
            sents = ""
            if (author!=None and poetry.get("author")!=author):
                continue
            p = poetry.get("paragraphs")
            flag = False
            for s in p:
                sp = re.split("[，！。]", s)
                for tr in sp:
                    if constrain != None and len(tr) != constrain and len(tr)!=0:
                        flag = True
                        break
                    if flag:
                        break
            if flag:
                continue
            for sentence in poetry.get("paragraphs"):
                sents += sentence
            pdata = sentenceProcess(sents)
            if pdata!="":
                poetries.append(pdata)
        #
        return poetries
    #
    data = []
    for filename in os.listdir(source_fileName):
        if filename.startswith("poet.tang"):
            data.extend(handleJson(source_fileName+filename))
    #
    data_filter=filter(data)
    #
    print("Num of poetries all:{},Num of after filtering poetries:{}".format({len(data)},{len(data_filter)}))
    #
    return data_filter

def filter(data):
    '''
    过滤掉：只有一句的以及不足一句几个字的！
    data:[poetry1,poetry2,...]
    poetries_filter:[[p11,p12,p13..],[p21,p22,p23...],[...]...]
    '''
    poetries_filter=[]
    for p in data:
        if(p.count('。')<=1):
            #只有一句或者
            continue
        else:
            tmp=list(p)
            tmp.append('<EOS>')
            poetries_filter.append(p)
    #
    return poetries_filter

def word2Id_id2Word(data_filter,w2Id_save_fileName,i2Wd_save_fileName):
    '''
    word2ID dict 保存！
    index2Wd dict 保存！
    SOS目测不需要！
    '''
    #
    w2Id={}
    i2Wd={}
    for pety in data_filter:
        for wd in pety:
            if(wd in w2Id):
                continue
            else:
                w2Id[wd]=len(w2Id)
                i2Wd[w2Id[wd]]=wd
    #
    w2Id['<EOS>'] = len(w2Id)
    i2Wd[w2Id['<EOS>']] = '<EOS>'
    # w2Id['<SOS>'] = len(w2Id)
    # i2Wd[w2Id['<SOS>']] = '<SOS>'
    #save
    print("Saving word2ID dict...!")
    with open(w2Id_save_fileName, 'wb') as fw:
        pickle.dump(w2Id, fw)
    print("Saving index2Wd dict...!")
    with open(i2Wd_save_fileName, 'wb') as fw:
        pickle.dump(w2Id, fw)
    #
    vocab_size=len(w2Id)
    #
    print("vocab size:{}".format(vocab_size))
    #
    return vocab_size

def train_val_split(data_filter):
    '''
    train and val data set split!
    T:V=8:2
    '''
    #
    random.shuffle(data_filter)
    #
    data_size=len(data_filter)
    split_point=round(data_size*0.8)
    #
    train_data_filter=data_filter[:split_point]
    val_data_filter = data_filter[split_point:]
    #
    print("All data size:{},split val:train->{}".format(len(data_filter),(len(val_data_filter))/(len(train_data_filter))))
    #
    return train_data_filter,val_data_filter

def sent2Id(sent,w2id):
    '''
    One sentence to id!
    注意：sent已经添加<EOS>,没必要添加<SOS>吧!
    '''
    sent2id=[]
    for char in sent:
        sent2id.append(w2id[char])
    #
    return sent2id

def generate_one_sample(sentId):
    '''
    Input sequence!
    OutPut sequence!
    '''
    #
    seq_input = []
    seq_output = []
    #
    for i in range(1, len(sentId)):
        seq_input.append(sentId[i-1])
        seq_output.append(sentId[i])
    #
    seq_input=Variable(torch.from_numpy(np.array(seq_input)))
    seq_output=Variable(torch.from_numpy(np.array(seq_output)))
    #
    if(torch.cuda.is_available()):
        seq_input=seq_input.cuda()
        seq_output=seq_output.cuda()
    #
    return seq_input, seq_output