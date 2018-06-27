# -*- coding: UTF-8 -*-
"""
===============================================================
author：xieqiqi
email：xieqiqi@jd.com
date：2018
introduction:
            Model load and test!
===============================================================
"""
import torch
import argparse,os,pickle
from model import Poetry_LM
from data_helper import parseRawData,word2Id_id2Word,sent2Id
from train import inference

def parse_arguments():
    parse = argparse.ArgumentParser(description='Hyperparams of this project!')
    #
    parse.add_argument('--hidden_dim', type=int, default=256,help='Hidden dim of encoder!')
    parse.add_argument('--embed_dim', type=int, default=256,help='Embed dim of encoder!')
    #
    parse.add_argument('--epochs', type=int, default=10,help='number of epochs for train')
    parse.add_argument('--batch_size', type=int, default=100,help='number of epochs for train')
    parse.add_argument('--lr', type=float, default=0.011,help='initial learning rate')
    parse.add_argument('--max_length', type=int, default=100, help='max_length')
    #
    parse.add_argument('--Base_path', type=str, default='data/', help='Base path!')
    parse.add_argument('--Save_path', type=str, default='data/save_test_01/',help='Save path!')
    #
    parse.add_argument('--mode',type=str,default='inference',help='Type of mode!')
    #
    return parse.parse_args()
#
args=parse_arguments()
print("===============================================================")
source_file_name=os.path.join(args.Base_path,"chinese-poetry/json/")
w2Id_save_fileName=os.path.join(args.Save_path,'w2Id')
i2Wd_save_fileName=os.path.join(args.Save_path,'id2Wd')

model_save_fileName=os.path.join(args.Save_path,'Poetry_LM.pt')

with open(w2Id_save_fileName, "rb") as fr:
    w2Id = pickle.load(fr)
with open(i2Wd_save_fileName, "rb") as fr:
    i2Wd = pickle.load(fr)
print("===============================================================")
data_filter=parseRawData(source_fileName=source_file_name)
vocab_size=word2Id_id2Word(data_filter=data_filter,
                            w2Id_save_fileName=w2Id_save_fileName,
                            i2Wd_save_fileName=i2Wd_save_fileName)
# print("=============================定义Model网络===============================")
# print("Models initializing....")
# model=Poetry_LM(vocab_size=vocab_size,
#                 embedding_dim=args.embed_dim,
#                 hidden_dim=args.hidden_dim)
# if(torch.cuda.is_available()):
#     model=model.cuda()
print("==========================加载Model网络参数==============================")
# 加载网络参数
model_load = torch.load(model_save_fileName)
if(torch.cuda.is_available()):
    model_load=model_load.cuda()
print("==========================Start evaluation!==============================")
if(args.mode=="inference"):
    while(True):
        start_wd = input()
        start_wd_lst=[]
        if start_wd == '' or start_wd.isspace():
            print('See you next time!')
            break
        else:
            print("start_wd——>", start_wd)
            start_wd_lst.append(start_wd)
            start_wd_id = sent2Id(sent=start_wd_lst,w2id=w2Id)
            # predit!
            outPut=inference(input=start_wd_id,model=model_load,max_length=args.max_length,i2Wd=i2Wd)
            #
            print(str(start_wd)+outPut)
