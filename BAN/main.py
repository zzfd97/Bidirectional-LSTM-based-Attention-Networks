import re
import logging
import jieba
import torch
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
from BAN import BAN
import sys
import math
from sklearn import metrics
from torch.utils.data import DataLoader
import multiprocessing

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def padding(seq, maxlen):
    x = (np.ones(maxlen) * 0).astype("int64")
    trunc = seq[-maxlen:]
    trunc = np.asarray(trunc, dtype="int64")
    x[:len(trunc)] = trunc
    return x

data = []
file = open("weibo_data.txt", "r", encoding="utf-8")
data += eval(file.read())
file.close()
data = [item for item in data if "sentiment" in item.keys() and "aspects" in item.keys() and type(item['sentiment']) == float
        and item['full'] == 3]


sentences = []
aspects = []
labels = []
for i in range(len(data)):
    labels.append(data[i]["sentiment"] + 1)
    string = data[i]["text"]
    data[i]["text"] = [word for word in jieba.cut(string) if word != " "]
    sentences.append(data[i]["text"])
    aspect = []
    for j in range(len(data[i]["aspects"])):
        words = str(data[i]["aspects"][j])
        for word in jieba.cut(words):
            aspect.append(word)
    aspect_words = [word for word in list(set(aspect)) if word in data[i]["text"]]
    if len(aspect_words) == 0:
        aspect_words = [word for word in jieba.cut(data[i]["keywords"])]
    data[i]["aspects"] = aspect_words
    aspects.append(data[i]["aspects"])

vocab = []
for sentence in sentences:
    for word in sentence:
        vocab.append(word)
vocab = list(set(vocab))
word2idx = {word: i for i, word in enumerate(vocab)}
idx2word = {i: word for i, word in enumerate(vocab)}

# wvmodel = KeyedVectors.load_word2vec_format("word.vector")
# wvmodel = KeyedVectors.load_word2vec_format("C:\\Users\\73974\\Downloads\\sgns.weibo.word")

vocab_size = len(vocab)
embed_size = 300
max_len = 80
batch_size = 1
embedding_matrix = torch.zeros(vocab_size, embed_size)
# for i in range(len(wvmodel.index2word)):
#     try:
#         index = word2idx[wvmodel.index2word[i]]
#     except:
#         continue
#     embedding_matrix[index, :] = torch.from_numpy(wvmodel.get_vector(idx2word[word2idx[wvmodel.index2word[i]]]))

all_data = []
for i in range(len(sentences)):
    context_indices = [word2idx[word] for word in sentences[i]]
    context_indices = padding(context_indices, max_len)
    aspect_indices = [word2idx[word] for word in aspects[i]]
    aspect_indices = padding(aspect_indices, max_len)
    dataset = {
        "sentence": sentences[i],
        "context": context_indices,
        "aspect": aspect_indices,
        "label": int(labels[i])
    }
    all_data.append(dataset)

trainset = all_data[:2000]
valset = all_data[2000:]

train_data_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)


model = BAN(embedding_matrix, embed_size, 300, 3)
params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(params, lr=0.0001, weight_decay=0.01)
criterion = torch.nn.CrossEntropyLoss()

for p in model.parameters():
    if p.requires_grad:
        if len(p.shape) > 1:
            torch.nn.init.xavier_uniform_(p)
        else:
            stdv = 1. / math.sqrt(p.shape[0])
            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

global_step = 0
max_val_acc = 0.76
for epoch in range(10):
    logger.info('epoch: {}'.format(epoch))
    n_correct, n_total, loss_total = 0, 0, 0
    model.train()
    for i, sample in enumerate(train_data_loader):
        global_step += 1
        optimizer.zero_grad()
        inputs = [sample["context"], sample["aspect"]]
        outputs = model(inputs)
        targets = sample["label"]
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
        n_total += len(outputs)
        loss_total += loss.item() * len(outputs)
        if global_step % 5 == 0:
            train_acc = n_correct / n_total
            train_loss = loss_total / n_total
            logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for batch, sample_bathed in enumerate(val_data_loader):
            v_inputs = [sample_bathed["context"], sample_bathed["aspect"]]
            v_targets = sample_bathed["label"]
            v_outputs = model(v_inputs)

            correct += (torch.argmax(v_outputs, -1) == v_targets).sum().item()
            total += len(v_outputs)

    val_acc = correct / total
    logger.info("正确数：{:f}".format(correct))
    logger.info("总数：{:f}".format(total))
    logger.info('> val_acc: {:.4f}'.format(val_acc))
    if val_acc > max_val_acc:
        max_val_acc = val_acc
        torch.save(model.state_dict(), "model_val_acc{}.pkl".format(round(val_acc, 4)))
