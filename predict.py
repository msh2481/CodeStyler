from random import shuffle, choices, choice
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from collections import deque, defaultdict, Counter

CHUNK_SIZE = 128
BATCH_SIZE = 256
BATCHES_IN_TRAIN = 8
BATCHES_IN_TEST = 2
TRAIN_SIZE = BATCH_SIZE * BATCHES_IN_TRAIN * 100
TEST_SIZE = BATCH_SIZE * BATCHES_IN_TEST
MIN_OCCURENCES = 10
MEMORY = 100
LAYERS = 2

def fmt(number):
    return '{:.5f}'.format(number)

rawTexts = []
alphabet = Counter()
import string
printable = set(string.printable)

for filename in open('filenames.txt'):
    if len(rawTexts) > TRAIN_SIZE + TEST_SIZE:
        break
    text = open(filename.strip(), encoding='utf-8').read()
    text = ''.join([x for x in text if  x in printable])
    if 'debug' in text or 'DEBUG' in text or '000' in text:
        continue
    for c in text:
        assert c in printable
    alphabet.update(text)
    for pos in range(0, len(text) - CHUNK_SIZE + 1):
        rawTexts.append(text[pos : pos + CHUNK_SIZE])
alphabetCount = Counter()
alphabetCount['%'] = 0
for x, y in alphabet.items():
    if y >= MIN_OCCURENCES:
        alphabetCount[x] += y
    else:
        alphabetCount['%'] += y
alphabet = [x for x, y in alphabetCount.items()]
ALPHABET_SIZE = len(alphabet)


shuffle(rawTexts)

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

trainSet = DataLoader(StringDataset(rawTexts[: TRAIN_SIZE]), batch_size=BATCH_SIZE, shuffle=True)
testSet = DataLoader(StringDataset(rawTexts[TRAIN_SIZE : TRAIN_SIZE + TEST_SIZE]), batch_size=BATCH_SIZE, shuffle=False)



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

def guessNextK(predictor, prefix, k, boringness):
    print(prefix, end='', flush=True)
    for it in range(k):
        p = guessNext(predictor, prefix)
        if len(prefix) >= 10 and all(c in ' \t\n' for c in prefix[len(prefix)-10:]):
            p[charToIndex(' ')] = 0
            p[charToIndex('\t')] = 0
            p[charToIndex('\n')] = 0
        p *= boringness
        p = p.softmax(dim=-1)
        p = list(p.detach().numpy())
        c = choices(alphabet, weights=p, k=1)[0]
        prefix += c
        print(c, end='', flush=True)
    print(flush=True)

predictor = nn.LSTM(input_size=ALPHABET_SIZE,
                   hidden_size=ALPHABET_SIZE+MEMORY,
                   num_layers=LAYERS,
                   bias=True,
                   batch_first=True,
                   dropout=0,
                   bidirectional=False,
                   proj_size=ALPHABET_SIZE)

predictor.load_state_dict(torch.load('models/2000.p'))

while True:
    boringness = float(input('Boringness: '))
    length = int(input('Length: '))
    s = input()
    guessNextK(predictor, s, length, boringness)