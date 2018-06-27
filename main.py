# -*- coding: UTF-8 -*-
"""
===============================================================
author：xieqiqi
email：xieqiqi@jd.com
date：2018
introduction:
            Main Program for Chinese poetry LM based on Lstm!
===============================================================
"""
import torch
import argparse,os,pickle
from model import Poetry_LM
from data_helper import parseRawData,word2Id_id2Word,train_val_split
from train import train_one_epoch,eval_after_one_epoch

def parse_arguments():
    parse = argparse.ArgumentParser(description='Hyperparams of this project!')
    #
    parse.add_argument('--hidden_dim', type=int, default=256,help='Hidden dim of encoder!')
    parse.add_argument('--embed_dim', type=int, default=256,help='Embed dim of encoder!')
    parse.add_argument('--max_length', type=float, default=100, help='max_length')
    #
    parse.add_argument('--epochs', type=int, default=10,help='number of epochs for train')
    parse.add_argument('--batch_size', type=int, default=100,help='number of epochs for train')
    parse.add_argument('--lr', type=float, default=0.011,help='initial learning rate')
    #
    parse.add_argument('--Base_path', type=str, default='data/', help='Base path!')
    parse.add_argument('--Save_path', type=str, default='data/save_test_00/',help='Save path!')
    #
    parse.add_argument('--mode',type=str,default='train',help='Type of mode!')
    #
    return parse.parse_args()
#
args=parse_arguments()
print("===============================================================")
print("Path setting...")
source_file_name=os.path.join(args.Base_path,"chinese-poetry/json/")
w2Id_save_fileName=os.path.join(args.Save_path,'w2Id')
i2Wd_save_fileName=os.path.join(args.Save_path,'id2Wd')

model_save_fileName=os.path.join(args.Save_path,'Poetry_LM.pt')

save_path=args.Save_path
if(not os.path.exists(save_path)):
    os.mkdir(save_path)

print("===============================================================")
print("Loading data and Data processing!")
data_filter=parseRawData(source_fileName=source_file_name)
# #
# for p in data_filter:
#     print(p)
# #
print(data_filter[0])
vocab_size=word2Id_id2Word(data_filter=data_filter,
                w2Id_save_fileName=w2Id_save_fileName,
                i2Wd_save_fileName=i2Wd_save_fileName)

with open(w2Id_save_fileName, "rb") as fr:
    w2Id = pickle.load(fr)
print("=============================定义Model网络===============================")
print("Models initializing....")
model=Poetry_LM(vocab_size=vocab_size,
                embedding_dim=args.embed_dim,
                hidden_dim=args.hidden_dim)
if(torch.cuda.is_available()):
    model=model.cuda()
print("Structure of Model....")
print(model)
#
optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=0.0001)
#
print("=============================Model_Training!===============================")
if(args.mode=="train"):
    train_data_filter, val_data_filter = train_val_split(data_filter=data_filter, ratio=0.98)
    for epoch in range(args.epochs):
        #train one epoch!
        train_one_epoch(epoch_num=epoch,model=model,
                        optimizer=optimizer,
                        data_filter_train=train_data_filter,
                        batch_size=args.batch_size,w2id=w2Id)
        #
        avg_val_loss=eval_after_one_epoch(model=model,data_filter_val=val_data_filter,w2id=w2Id)
        print("After training on epoch:{},Model performance on val_set——>avg_val_loss:{}".format({epoch},{avg_val_loss.data[0]}))
    #save model!
    print("Saving model...!")
    torch.save(model,model_save_fileName)
    print("Training stop!")