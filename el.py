#! -*- coding: utf-8 -*-
# 2019年百度的实体链指比赛（ ccks2019，https://biendata.com/competition/ccks_2019_el/ ），一个baseline

import json
from tqdm import tqdm
import os
import numpy as np
from random import choice
from itertools import groupby


mode = 0
min_count = 2
char_size = 128


id2kb = {}
with open('../ccks2019_el/kb_data') as f:
    for l in tqdm(f):
        _ = json.loads(l)
        subject_id = _['subject_id']
        subject_alias = list(set([_['subject']] + _.get('alias', [])))
        subject_alias = [alias.lower() for alias in subject_alias]
        subject_desc = '\n'.join(u'%s：%s' % (i['predicate'], i['object']) for i in _['data'])
        subject_desc = subject_desc.lower()
        if subject_desc:
            id2kb[subject_id] = {'subject_alias': subject_alias, 'subject_desc': subject_desc}


kb2id = {}
for i,j in id2kb.items():
    for k in j['subject_alias']:
        if k not in kb2id:
            kb2id[k] = []
        kb2id[k].append(i)


train_data = []
with open('../ccks2019_el/train.json') as f:
    for l in tqdm(f):
        _ = json.loads(l)
        train_data.append({
            'text': _['text'].lower(),
            'mention_data': [(x['mention'].lower(), int(x['offset']), x['kb_id'])
                for x in _['mention_data'] if x['kb_id'] != 'NIL'
            ]
        })


if not os.path.exists('../all_chars_me.json'):
    chars = {}
    for d in tqdm(iter(id2kb.values())):
        for c in d['subject_desc']:
            chars[c] = chars.get(c, 0) + 1
    for d in tqdm(iter(train_data)):
        for c in d['text']:
            chars[c] = chars.get(c, 0) + 1
    chars = {i:j for i,j in chars.items() if j >= min_count}
    id2char = {i+2:j for i,j in enumerate(chars)} # 0: mask, 1: padding
    char2id = {j:i for i,j in id2char.items()}
    json.dump([id2char, char2id], open('../all_chars_me.json', 'w'))
else:
    id2char, char2id = json.load(open('../all_chars_me.json'))


if not os.path.exists('../random_order_train.json'):
    random_order = range(len(train_data))
    np.random.shuffle(random_order)
    json.dump(
        random_order,
        open('../random_order_train.json', 'w'),
        indent=4
    )
else:
    random_order = json.load(open('../random_order_train.json'))


dev_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 == mode]
train_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 != mode]


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=64):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = range(len(self.data))
            np.random.shuffle(idxs)
            X1, X2, S1, S2, Y, T = [], [], [], [], [], []
            for i in idxs:
                d = self.data[i]
                text = d['text']
                x1 = [char2id.get(c, 1) for c in text]
                s1, s2 = np.zeros(len(text)), np.zeros(len(text))
                mds = {}
                for md in d['mention_data']:
                    if md[0] in kb2id:
                        j1 = md[1]
                        j2 = j1 + len(md[0])
                        s1[j1] = 1
                        s2[j2 - 1] = 1
                        mds[(j1, j2)] = (md[0], md[2])
                if mds:
                    j1, j2 = choice(mds.keys())
                    y = np.zeros(len(text))
                    y[j1: j2] = 1
                    x2 = choice(kb2id[mds[(j1, j2)][0]])
                    if x2 == mds[(j1, j2)][1]:
                        t = [1]
                    else:
                        t = [0]
                    x2 = id2kb[x2]['subject_desc']
                    x2 = [char2id.get(c, 1) for c in x2]
                    X1.append(x1)
                    X2.append(x2)
                    S1.append(s1)
                    S2.append(s2)
                    Y.append(y)
                    T.append(t)
                    if len(X1) == self.batch_size or i == idxs[-1]:
                        X1 = seq_padding(X1)
                        X2 = seq_padding(X2)
                        S1 = seq_padding(S1)
                        S2 = seq_padding(S2)
                        Y = seq_padding(Y)
                        T = seq_padding(T)
                        yield [X1, X2, S1, S2, Y, T], None
                        X1, X2, S1, S2, Y, T = [], [], [], [], [], []


from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam


def seq_maxpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask = x
    seq -= (1 - mask) * 1e10
    return K.max(seq, 1)


x1_in = Input(shape=(None,)) # 待识别句子输入
x2_in = Input(shape=(None,)) # 实体语义表达输入
s1_in = Input(shape=(None,)) # 实体左边界（标签）
s2_in = Input(shape=(None,)) # 实体右边界（标签）
y_in = Input(shape=(None,)) # 实体标记
t_in = Input(shape=(1,)) # 是否有关联（标签）


x1, x2, s1, s2, y, t = x1_in, x2_in, s1_in, s2_in, y_in, t_in
x1_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x1)
x2_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x2)

embedding = Embedding(len(id2char)+2, char_size)


x1 = embedding(x1)
x1 = Dropout(0.2)(x1)
x1 = Lambda(lambda x: x[0] * x[1])([x1, x1_mask])
x1 = Bidirectional(CuDNNLSTM(char_size//2, return_sequences=True))(x1)
x1 = Lambda(lambda x: x[0] * x[1])([x1, x1_mask])
x1 = Bidirectional(CuDNNLSTM(char_size//2, return_sequences=True))(x1)
x1 = Lambda(lambda x: x[0] * x[1])([x1, x1_mask])

h = Conv1D(char_size, 3, activation='relu', padding='same')(x1)
ps1 = Dense(1, activation='sigmoid')(h)
ps2 = Dense(1, activation='sigmoid')(h)

s_model = Model(x1_in, [ps1, ps2])


y = Lambda(lambda x: K.expand_dims(x, 2))(y)
x1 = Concatenate()([x1, y])
x1 = Conv1D(char_size, 3, padding='same')(x1)

x2 = embedding(x2)
x2 = Dropout(0.2)(x2)
x2 = Lambda(lambda x: x[0] * x[1])([x2, x2_mask])
x2 = Bidirectional(CuDNNLSTM(char_size//2, return_sequences=True))(x2)
x2 = Lambda(lambda x: x[0] * x[1])([x2, x2_mask])
x2 = Bidirectional(CuDNNLSTM(char_size//2, return_sequences=True))(x2)
x2 = Lambda(lambda x: x[0] * x[1])([x2, x2_mask])

x1 = Lambda(seq_maxpool)([x1, x1_mask])
x2 = Lambda(seq_maxpool)([x2, x2_mask])
x12 = Multiply()([x1, x2])
x = Concatenate()([x1, x2, x12])
x = Dense(char_size, activation='relu')(x)
pt = Dense(1, activation='sigmoid')(x)

t_model = Model([x1_in, x2_in, y_in], pt)


train_model = Model([x1_in, x2_in, s1_in, s2_in, y_in, t_in],
                    [ps1, ps2, pt])

s1 = K.expand_dims(s1, 2)
s2 = K.expand_dims(s2, 2)

s1_loss = K.binary_crossentropy(s1, ps1)
s1_loss = K.sum(s1_loss * x1_mask) / K.sum(x1_mask)
s2_loss = K.binary_crossentropy(s2, ps2)
s2_loss = K.sum(s2_loss * x1_mask) / K.sum(x1_mask)
pt_loss = K.mean(K.binary_crossentropy(t, pt))

loss = s1_loss + s2_loss + pt_loss

train_model.add_loss(loss)
train_model.compile(optimizer=Adam(1e-3))
train_model.summary()


def extract_items(text_in):
    _x1 = [char2id.get(c, 1) for c in text_in]
    _x1 = np.array([_x1])
    _k1, _k2 = s_model.predict(_x1)
    _k1, _k2 = _k1[0, :, 0], _k2[0, :, 0]
    _k1, _k2 = np.where(_k1 > 0.5)[0], np.where(_k2 > 0.5)[0]
    _subjects = []
    for i in _k1:
        j = _k2[_k2 >= i]
        if len(j) > 0:
            j = j[0]
            _subject = text_in[i: j+1]
            _subjects.append((_subject, i, j))
    if _subjects:
        R = []
        _X2, _Y = [], []
        _S, _IDXS = [], {}
        for _s in _subjects:
            _y = np.zeros(len(text_in))
            _y[_s[1]: _s[2]] = 1
            _IDXS[_s] = kb2id.get(_s[0], [])
            for i in _IDXS[_s]:
                _x2 = id2kb[i]['subject_desc']
                _x2 = [char2id.get(c, 1) for c in _x2]
                _X2.append(_x2)
                _Y.append(_y)
                _S.append(_s)
        if _X2:
            _X2 = seq_padding(_X2)
            _Y = seq_padding(_Y)
            _X1 = np.repeat(_x1, len(_X2), 0)
            scores = t_model.predict([_X1, _X2, _Y])[:, 0]
            for k, v in groupby(zip(_S, scores), key=lambda s: s[0]):
                v = np.array([j[1] for j in v])
                kbid = _IDXS[k][np.argmax(v)]
                R.append((k[0], k[1], kbid))
        return R
    else:
        return []


class Evaluate(Callback):
    def __init__(self):
        self.F1 = []
        self.best = 0.
    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = self.evaluate()
        self.F1.append(f1)
        if f1 > self.best:
            self.best = f1
            train_model.save_weights('best_model.weights')
        print 'f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' % (f1, precision, recall, self.best)
    def evaluate(self):
        A, B, C = 1e-10, 1e-10, 1e-10
        for d in tqdm(iter(dev_data)):
            R = set(extract_items(d['text']))
            T = set(d['mention_data'])
            A += len(R & T)
            B += len(R)
            C += len(T)
        return 2 * A / (B + C), A / B, A / C


evaluator = Evaluate()
train_D = data_generator(train_data)

train_model.fit_generator(train_D.__iter__(),
                          steps_per_epoch=len(train_D),
                          epochs=40,
                          callbacks=[evaluator]
                         )
