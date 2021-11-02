from random import shuffle, choices, choice
from collections import deque, defaultdict, Counter
from itertools import islice
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

'''
Сделать вложение алфавита в вектор
'''

CHUNK_SIZE = 128
BATCH_SIZE = 256
BATCHES_IN_TRAIN = 8
BATCHES_IN_TEST = 2
TRAIN_SIZE = BATCH_SIZE * BATCHES_IN_TRAIN * 25
TEST_SIZE = BATCH_SIZE * BATCHES_IN_TEST
MIN_OCCURENCES = 10
MEMORY = 100
LAYERS = 2
ALPHABET_STRING = '37 47 32 33 76 65 78 71 85 69 58 43 101 119 73 110 102 114 99 80 111 104 105 98 116 118 115 108 97 77 100 83 117 112 10 68 79 84 67 45 95 86 82 66 88 75 70 74 49 46 107 103 40 41 123 60 62 125 61 34 109 42 50 106 59 87 72 51 53 122 120 36 121 44 64 63 48 52 54 55 89 56 57 96 124 113 38 91 93 90 39 81 92'

def fmt(number):
    return '{:.5f}'.format(number)

rawTexts = []
alphabet = Counter()
import string
printable = set(string.printable)

for filename in open('filenames_msh2481.txt'):
    if len(rawTexts) > TRAIN_SIZE + TEST_SIZE:
        break
    text = open(filename.strip(), encoding='utf-8').read()
    text = ''.join([x for x in text if  x in printable])
    # if 'debug' in text or 'DEBUG' in text or '000' in text:
    #     continue
    for c in text:
        assert c in printable
    print(text)
    for pos in range(0, len(text) - CHUNK_SIZE + 1):
        rawTexts.append(text[pos : pos + CHUNK_SIZE])

alphabet = [chr(int(x)) for x in ALPHABET_STRING.split()]
ALPHABET_SIZE = len(alphabet)
# print(f'alphabet of length {len(alphabet)}: {alphabetCount}')

print(*[ord(c) for c in alphabet])

shuffle(rawTexts)
print(f'{len(rawTexts)} texts in total')

charToIndexMap = { c : i for i, c in enumerate(alphabet) }
def charToIndex(c):
    return torch.as_tensor(charToIndexMap.get(c, ALPHABET_SIZE - 1), dtype=torch.long)

def stringToTensor(cur):
    x = torch.zeros(size=(len(cur), ALPHABET_SIZE))
    for j in range(len(cur)):
        x[j][charToIndex(cur[j])] = 1
    return x

class StringDataset(Dataset):
    def __init__(self, strings):
        super(StringDataset, self).__init__()
        self.strings = strings
    def __len__(self):
        return len(self.strings)
    def __getitem__(self, i):
        return stringToTensor(self.strings[i])

assert TRAIN_SIZE + TEST_SIZE <= len(rawTexts)
trainSet = DataLoader(StringDataset(rawTexts[: TRAIN_SIZE]), batch_size=BATCH_SIZE, shuffle=True)
testSet = DataLoader(StringDataset(rawTexts[TRAIN_SIZE : TRAIN_SIZE + TEST_SIZE]), batch_size=BATCH_SIZE, shuffle=False)


print(len(trainSet), len(testSet))
print('---')
# print(next(iter(trainSet)))
print('---')



lossFunction = nn.CrossEntropyLoss()

def parametersTensor(predictor):
    return torch.cat(tuple(elem.view(-1) for elem in predictor.parameters()))

def gradientsTensor(predictor):
    return torch.cat(tuple(elem.grad.view(-1) for elem in predictor.parameters()))

X_ORT = None
Y_ORT = None

def tensorTo2D(v):
    global X_ORT, Y_ORT
    if X_ORT is None:
        assert Y_ORT is None
        X_ORT = torch.rand(v.shape, dtype=torch.double)
        Y_ORT = torch.rand(v.shape, dtype=torch.double)
        X_ORT = F.normalize(X_ORT, dim=0)
        Y_ORT = F.normalize(Y_ORT, dim=0)
    print(X_ORT.sum(), X_ORT.mean(), X_ORT.std())
    vx = torch.mul(v, X_ORT)
    print(vx.mean(), vx.std())
    vy = torch.mul(v, Y_ORT)
    return vx.sum(), vy.sum()

def evaluateOnBatch(predictor, batch):
    N = batch.shape[0]
    assert batch.shape == (N, CHUNK_SIZE, ALPHABET_SIZE)
    h0 = torch.randn((LAYERS, N, ALPHABET_SIZE))    
    c0 = torch.randn((LAYERS, N, ALPHABET_SIZE + MEMORY))
    data = batch[:, :-1, :]
    answer = batch[:, -1, :].argmax(dim=-1)
    assert answer.shape == (N, )
    output = predictor(data, (h0, c0))[0][:, -1, :]
    output = output[:, :ALPHABET_SIZE]
    assert output.shape == (N, ALPHABET_SIZE)
    loss = lossFunction(output, answer)
    accuracy = (output.argmax(dim=-1) == answer).float().mean()
    return accuracy, loss
        
def train(predictor, optimizer, startEpoch):
    predictor.train()
    trainAccuracy = 0
    trainLogLoss = 0
    trainSize = 0
    for batch in islice(trainSet, BATCHES_IN_TRAIN):
        optimizer.zero_grad()
        accuracy, loss = evaluateOnBatch(predictor, batch)
        loss.backward()
        optimizer.step()
        trainAccuracy += accuracy
        trainLogLoss += loss.item()
    trainAccuracy /= BATCHES_IN_TRAIN
    trainLogLoss /= BATCHES_IN_TRAIN

    with torch.no_grad():
        predictor.eval()
        testAccuracy = 0
        testLogLoss = 0
        testSize = 0
        for batch in islice(testSet, BATCHES_IN_TEST):
            accuracy, logLoss = evaluateOnBatch(predictor, batch)
            testAccuracy += accuracy
            testLogLoss += loss.item()
        testAccuracy /= BATCHES_IN_TEST
        testLogLoss /= BATCHES_IN_TEST

        p = parametersTensor(predictor)
        g = gradientsTensor(predictor)
        print(f'State: parameters = ({fmt(p.mean())}, {fmt(p.std())}) gradients = ({fmt(g.mean())},  {fmt(g.std())})')
        print(f'#{startEpoch}: {fmt(trainAccuracy)} {fmt(trainLogLoss)} {fmt(testAccuracy)} {fmt(testLogLoss)}')
        print(flush=True)

def guessNext(predictor, text):
    data = stringToTensor(text).view(1, -1, ALPHABET_SIZE)
    h0 = torch.randn((LAYERS, 1, ALPHABET_SIZE))    
    c0 = torch.randn((LAYERS, 1, ALPHABET_SIZE + MEMORY))    
    output = predictor(data, (h0, c0))[0][:, -1, :]
    output = output[0, :ALPHABET_SIZE]
    return output

def guessNextK(predictor, prefix, k):
    for it in range(k):
        p = guessNext(predictor, prefix)
        i = p.argmax(dim=0).item()
        c = alphabet[i]
        prefix += c
    return prefix

predictor = nn.LSTM(input_size=ALPHABET_SIZE,
                   hidden_size=ALPHABET_SIZE+MEMORY,
                   num_layers=LAYERS,
                   bias=True,
                   batch_first=True,
                   dropout=0,
                   bidirectional=False,
                   proj_size=ALPHABET_SIZE)
predictor.load_state_dict(torch.load('models/lstm/3000.p'))
optimizer = torch.optim.Adam(predictor.parameters())

for i in range(10 ** 9):
    train(predictor, optimizer, i)
    if all(c == '0' for c in str(i)[1:]):
        torch.save(predictor.state_dict(), f'models/{i}.p')
    if i % 10 == 9:
        print(guessNextK(predictor, choice(rawTexts), 300))
    if i % 100 == 99:
        print(guessNextK(predictor, choice(rawTexts), 1000))