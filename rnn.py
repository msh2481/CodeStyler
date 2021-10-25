from random import shuffle, choices, choice
from collections import deque, defaultdict, Counter
from itertools import islice
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys
import codecs

def uprint(*objects, sep=' ', end='\n', file=sys.stdout):
    enc = file.encoding
    if enc == 'UTF-8':
        print(*objects, sep=sep, end=end, file=file)
    else:
        f = lambda obj: str(obj).encode(enc, errors='backslashreplace').decode(enc)
        print(*map(f, objects), sep=sep, end=end, file=file)

CHUNK_SIZE = 6
BATCH_SIZE = 256
BATCHES_IN_TRAIN = 10
BATCHES_IN_TEST = 2
TRAIN_SIZE = BATCH_SIZE * BATCHES_IN_TRAIN
TEST_SIZE = BATCH_SIZE * BATCHES_IN_TEST
MIN_OCCURENCES = 10
MEMORY = 100
DEPTH = 2
L1_WEIGHT = 1e-4

def fmt(number):
    return '{:.5f}'.format(number)

rawTexts = []
alphabet = Counter()
for filename in open('filenames.txt'):
    if len(rawTexts) > TRAIN_SIZE + TEST_SIZE:
        break
    text = open(filename.strip()).read()
    if 'debug' in text or 'DEBUG' in text:
        continue
    alphabet.update(text)
    for pos in range(0, len(text) - CHUNK_SIZE + 1):
        rawTexts.append(text[pos : pos + CHUNK_SIZE])
alphabetCount = Counter()
alphabetCount['█'] = 0
for x, y in alphabet.items():
    if y >= MIN_OCCURENCES:
        alphabetCount[x] += y
    else:
        alphabetCount['█'] += y
alphabet = [x for x, y in alphabetCount.items()]
ALPHABET_SIZE = len(alphabet)
uprint(f'alphabet of length {len(alphabet)}: {alphabetCount}')

shuffle(rawTexts)
uprint(f'{len(rawTexts)} texts in total')
# uprint(rawTexts[:3])

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

uprint(len(trainSet), len(testSet))
# uprint('---')
# uprint(next(iter(trainSet)))
# uprint('---')

class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.linear1 = nn.Linear(ALPHABET_SIZE + MEMORY, ALPHABET_SIZE + MEMORY, dtype=torch.double, bias=False)
        self.linear2 = nn.Linear(ALPHABET_SIZE + MEMORY, ALPHABET_SIZE + MEMORY, dtype=torch.double)
        self.linear3 = nn.Linear(ALPHABET_SIZE + MEMORY, MEMORY, dtype=torch.double)
        
        self.batchNorm = nn.BatchNorm1d(ALPHABET_SIZE + MEMORY, dtype=torch.double)
    
    def forward(self, state, answer):
        assert state.shape[1:] == (MEMORY, )
        if answer.shape[1:] != (ALPHABET_SIZE, ):
            uprint(state.shape, answer.shape)
        assert answer.shape[1:] == (ALPHABET_SIZE, )
        inputTensor = torch.cat((state, answer), dim=1)

        relevanceTensor = torch.sigmoid(self.linear3(inputTensor))

        inputTensor = torch.cat((torch.mul(state, relevanceTensor), answer), dim=1)
        updateTensor = torch.sigmoid(self.linear2(inputTensor))
        deltaTensor = torch.relu(self.batchNorm(self.linear1(inputTensor))) - inputTensor

        resultTensor = inputTensor + torch.mul(updateTensor, deltaTensor)
        state, answer = resultTensor[:, : MEMORY], resultTensor[:, MEMORY :]
        return state, F.softmax(answer, dim=-1)

lossFunction = nn.NLLLoss()

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
    vx = torch.mul(v, X_ORT).sum()
    vy = torch.mul(v, Y_ORT).sum()
    return vx, vy

def evaluateOnBatch(predictor, batch):
    N = batch.shape[0]
    batch = batch.permute(1, 0, 2)
    assert batch.shape == (CHUNK_SIZE, N, ALPHABET_SIZE)
    state = torch.rand((N, MEMORY), dtype=torch.double, requires_grad=True)
    answer = torch.rand((N, ALPHABET_SIZE), dtype=torch.double, requires_grad=True)
    loss = torch.tensor(0, dtype=torch.double)
    accuracy = torch.tensor(0, dtype=torch.double)
    for i in range(CHUNK_SIZE):
        expected = batch[i].argmax(dim=-1)
        assert expected.shape == (N, )
        for it in range(DEPTH):
            state, answer = predictor(state, answer)
            assert state.shape == (N, MEMORY)
            assert answer.shape == (N, ALPHABET_SIZE)
            if i > CHUNK_SIZE // 2:
                loss += lossFunction(answer.log(), expected)
        if i > CHUNK_SIZE // 2:
            accuracy += (answer.argmax(dim=-1) == expected).double().mean()
        answer = batch[i]
    return accuracy / (CHUNK_SIZE - CHUNK_SIZE // 2), loss / ((CHUNK_SIZE - CHUNK_SIZE // 2) * DEPTH)
        
def train(predictor, optimizer, startEpoch):
    predictor.train()
    trainAccuracy = 0
    trainLogLoss = 0
    trainSize = 0
    for batch in islice(trainSet, BATCHES_IN_TRAIN):
        optimizer.zero_grad()
        accuracy, loss = evaluateOnBatch(predictor, batch)
        loss = loss + parametersTensor(predictor).abs().sum() * L1_WEIGHT
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

        px, py = tensorTo2D(parametersTensor(predictor))
        gx, gy = tensorTo2D(gradientsTensor(predictor))
        uprint(f'State in 2D: parameters = ({px}, {py}) gradients = ({gx},  {gy})')
        uprint(f'#{startEpoch}: {fmt(trainAccuracy)} {fmt(trainLogLoss)} {fmt(testAccuracy)} {fmt(testLogLoss)}')
        print(flush=True)

def samplePrediction(predictor, length):
    s = ''
    sFull = ''
    prefix = choice(rawTexts)
    state = torch.rand((1, MEMORY), dtype=torch.double)
    answer = torch.rand((1, ALPHABET_SIZE), dtype=torch.double)
    for i in range(length):
        for it in range(DEPTH):
            state, answer = predictor(state, answer)
            guess = answer[0].argmax(dim=-1).item()
            sFull += alphabet[guess]
        w = list(*answer.detach())
        guess = choices(alphabet, w)[0]
        if i < len(prefix):
            guess = prefix[i]
        answer = torch.zeros((1, ALPHABET_SIZE))
        answer[0][charToIndex(guess)] = 1
        sFull += '>' + guess
        s += guess
    uprint(f'=== {len(s)}, {len(sFull)} ===')
    uprint(f's:{s}')
    uprint(f'sFull:{sFull}')

predictor = Predictor()
optimizer = torch.optim.Adam(predictor.parameters(), lr=0.001, weight_decay=1e-2)
print(flush=True)

for i in range(10 ** 9):
    train(predictor, optimizer, i)
    # for e in predictor.parameters():
    #     print(e.grad)
    uprint(i)
    samplePrediction(predictor, 64)