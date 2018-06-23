# coding:utf-8
import torch.nn as nn
import torch.optim as optim
from example import dataHandler
from example.model import Poetry_LM
from example.utils import *

import pickle

data = dataHandler.parseRawData()  # All if author=None
# data = dataHandler.parseRawData(author="李白".decode('utf-8'),constrain=5)  # All if author=None
# random.shuffle(data)
for s in data:
    print(s)
#
word_to_ix = {}
#
for sent in data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
#
word_to_ix['<EOP>'] = len(word_to_ix)
word_to_ix['<START>'] = len(word_to_ix)

VOCAB_SIZE = len(word_to_ix)

print("VOCAB_SIZE:", VOCAB_SIZE)
print("data_size", len(data))

for i in range(len(data)):
    data[i] = toList(data[i])
    data[i].append("<EOP>")

# save the word dic for sample method
with open('wordDic', 'wb') as fw:
    pickle.dump(word_to_ix, fw)

# save all avaible word
# wordList = open('wordList','w')
# for w in word_to_ix:
#     wordList.write(w.encode('utf-8'))
# wordList.close()

model = Poetry_LM(len(word_to_ix), 256, 256)
if torch.cuda.is_available():
    model.cuda()  # running on GPU,if you want to run it on CPU,delete all .cuda() usage.

optimizer = optim.RMSprop(model.parameters(), lr=0.01, weight_decay=0.0001)
criterion = nn.NLLLoss()

one_hot_var_target = {}
for w in word_to_ix:
    one_hot_var_target.setdefault(w, make_one_hot_vec_target(w, word_to_ix))

epochNum = 10
TRAINSIZE = len(data)
batch = 100
def test():
    v = int(TRAINSIZE / batch)
    loss = 0
    counts = 0
    for case in range(v * batch, min((v + 1) * batch, TRAINSIZE)):
        s = data[case]
        t, o = makeForOneCase(s, one_hot_var_target)
        if(torch.cuda.is_available()):
            t=t.cuda()
            o=o.cuda()
        output, hidden = model(t)
        loss += criterion(output, o)
        counts += 1
    loss = loss / counts
    print("=====",loss.data[0])

print("start training")
for epoch in range(epochNum):
    for batchIndex in range(int(TRAINSIZE / batch)):
        model.zero_grad()
        loss = 0
        counts = 0
        for case in range(batchIndex * batch, min((batchIndex + 1) * batch, TRAINSIZE)):
            s = data[case]
            t, o = makeForOneCase(s, one_hot_var_target)
            if torch.cuda.is_available():
                t=t.cuda()
                o=o.cuda()
            output, hidden = model(t)
            loss += criterion(output,o)
            counts += 1
        loss = loss / counts
        loss.backward()
        print(epoch, loss.data[0])
        optimizer.step()
    test()
torch.save(model, 'poetry-gen.pt')
