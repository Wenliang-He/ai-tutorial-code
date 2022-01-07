### Learning Resources
# 1. Andrew Ng's Deep Learning Specialization
# https://www.coursera.org/specializations/deep-learning
# 2. Tensorflow or Keras Tutorial
# https://www.tensorflow.org/tutorials
# https://keras.io/guides/
# 3. Assorted Resources
# Brandon Rohrer Neural Networks Playlist
# https://www.youtube.com/watch?v=ILsA4nyG7I0&list=PLLFi8Yg0Qyea3iA0kByAThOiFWRIaP7-4
# 3Blue1Brown Neural Networks Playlist
# https://www.bilibili.com/video/BV1Lt411M7q4
# DNN Visualization
# https://playground.tensorflow.org/

### References
# History of Deep Learning Frameworks
# https://towardsdatascience.com/a-brief-history-of-deep-learning-frameworks-8debf3ba6607



########################################
### Part 1. Intro to Neural Networks ###
########################################

### 1.1 Import Modules
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.preprocessing import sequence, text
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import imdb

from tensorflow.keras.layers import Embedding, Dropout, Dense, Activation
from tensorflow.keras.layers import Conv1D, GlobalAvgPool1D, GlobalMaxPool1D
from tensorflow.keras.layers import LSTM, Bidirectional, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, Callback, LambdaCallback

from tensorflow.python.client import device_lib

devices = device_lib.list_local_devices()
for i, device in enumerate(devices):
    if device.device_type == 'GPU':
        desc = device.physical_device_desc
        print("GPU device: '{}' ({:.2f} GB)".format(device.name, device.memory_limit / 1e9))
        print(desc[desc.find('name:'):], '\n')


### 1.2 Load & Inspect Data
vocab_size = 5000       # load most-freq 'vocab_size-1' words: Index in [1, vocab_size-1]; 0 is reversed for padding
max_len = 200           # max document length to pad or truncate into
padding = 'post'        # 'pre' < 'post'
truncating = 'post'     # 'pre' > 'post'

index_from = 2
(x_tran, y_tran), (x_test, y_test) = imdb.load_data(
    num_words=vocab_size,       # default=None; most frequent words are shown and use 'oov' to represent others
    maxlen=None,                # default; if n is integer, keep only the first n words for each document
    skip_top=0,                 # default; if n>=1, skip the top n words
    start_char=1,               # default; 1 is reserved for '<start>'
    oov_char=2,                 # default; 2 is reserved for '<oov>'
    index_from=index_from,      # default==3; highly recommend using 2 instead of 3
    seed=113,                   # default; seed for shuffling the data
)
print('training X shape:', x_tran.shape, '\t', 'training Y shape:', y_tran.shape)
print('test     X shape:', x_test.shape, '\t', 'test     Y shape:', y_test.shape)
x_tran[:10]
y_tran[:10]
set(y_tran)

print("Documents of varying length:", [len(doc) for doc in x_tran[:10]])
vocab = sorted(set([w for doc in x_tran for w in doc]))
print("Size of vocabulary:", len(vocab))
print("First vocab index:", vocab[0])
print("Last  vocab index:", vocab[-1])

word2idx = imdb.get_word_index()
word2idx = {w: (i + index_from) for w, i in word2idx.items()}
idx2word = {i: w for w, i in word2idx.items()}
idx2word.update({0: '<pad>', 1: '<start>', 2: '<oov>'})
for i in range(10):
    print(' '.join([idx2word[x] for x in x_tran[i]]))
    print()

# traditional bag-of-words approach can handle texts of variable lengths
samples_text = [' '.join([idx2word[i] for i in doc]).replace("'", '') for doc in x_tran[:10]]
samples_text
len(samples_text)

# neural networks need fixed-length texts
samples_padding = sequence.pad_sequences(x_tran[:10], maxlen=max_len, padding=padding, truncating=truncating)
samples_padding
samples_padding.shape


def load_data_integers(vocab_size=5000, max_len=200, padding='post', truncating='post'):
    (x_tran, y_tran), (x_test, y_test) = imdb.load_data(num_words=vocab_size, skip_top=0, index_from=2)
    x_tran = sequence.pad_sequences(x_tran, maxlen=max_len, padding=padding, truncating=truncating)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len, padding=padding, truncating=truncating)
    x_trn, x_val, y_trn, y_val = train_test_split(x_tran, y_tran, train_size=0.9, random_state=2021)
    return x_trn, y_trn, x_val, y_val, x_test, y_test

def load_data_words(vocab_size=5000, max_len=200, padding='post', truncating='post'):
    (x_tran, y_tran), (x_test, y_test) = imdb.load_data(num_words=vocab_size, skip_top=0, index_from=2)
    word2idx = imdb.get_word_index()
    word2idx = {w: (i + 2) for w, i in word2idx.items()}
    idx2word = {i: w for w, i in word2idx.items()}
    idx2word.update({0: '<pad>', 1: '<start>', 2: '<oov>'})

    # perform truncating to ensure identical texts for classification
    x_tran = sequence.pad_sequences(x_tran, maxlen=max_len, padding=padding, truncating=truncating)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len, padding=padding, truncating=truncating)

    # convert encoded integers back to words
    x_tran = [' '.join([idx2word[i] for i in doc if i not in [0, 1]]).replace("'", '') for doc in x_tran]
    x_test = [' '.join([idx2word[i] for i in doc if i not in [0, 1]]).replace("'", '') for doc in x_test]

    x_trn, x_val, y_trn, y_val = train_test_split(x_tran, y_tran, train_size=0.9, random_state=2021)
    return x_trn, y_trn, x_val, y_val, x_test, y_test


#####################################
### Part 2. Bag-of-words Approach ###
#####################################

### 2.1 Random Forest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

vocab_size = 5000
max_len = 200
x_trn, y_trn, x_val, y_val, x_test, y_test = load_data_words(vocab_size=vocab_size, max_len=max_len)

vec = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.95,
    max_features=vocab_size,
    sublinear_tf=True
)
x_trn = vec.fit_transform(x_trn).todense().astype(np.float16)  # use np.float16 to alleviate memory consumption
x_val = vec.transform(x_val).todense().astype(np.float16)
x_test = vec.transform(x_test).todense().astype(np.float16)
print('train X shape:', x_trn.shape, '\t', 'train Y shape:', y_trn.shape)
print('valid X shape:', x_val.shape, '\t', 'valid Y shape:', y_val.shape)
print('testa X shape:', x_test.shape, '\t', 'testa Y shape:', y_test.shape)

rf = RandomForestClassifier(n_jobs=-1, random_state=2021)
rf.fit(x_trn, y_trn)

y_pred = rf.predict(x_test)
print(classification_report(y_test, y_pred, digits=4))  # 0.8166


### 2.2 Dense Neural Network - DNN
# Fully Connected Neural Network (FCNN) or Multi-layer Perceptrons (MLP)
# https://www.youtube.com/watch?v=ILsA4nyG7I0
# https://www.bilibili.com/video/BV1Lt411M7q4

hidden_dim = 150    # number of nodes for dense hidden layer(s)
drop_rate = 0.4     # dropout ratio
batch_size = 64     # batch size for stochastic gradient descent

model = Sequential()  # model = []
# add 1st dense layer
model.add(Dense(hidden_dim, activation='relu', input_shape=(vocab_size,)))
model.add(Dropout(drop_rate))
# add 2nd dense layer
model.add(Dense(hidden_dim, activation='relu'))
model.add(Dropout(drop_rate))
# add 3rd dense layer
# model.add(Dense(hidden_dim, activation='relu'))
# model.add(Dropout(drop_rate))
# add the output layer
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# show the model
print(model.summary())

early_stopping = EarlyStopping(
    monitor='val_loss',             # default
    min_delta=1e-5,                 # default=0; minimum change counted as an improvement
    patience=5,                     # default=0; number of epochs with no improvement until stopping
    restore_best_weights=True,      # default=False
    verbose=1,                      # verbosity level
    mode='auto',                    # auto-inferred from 'monitor'
    baseline=None,                  # stop training given no improvement over baseline with 'patience'
)
callbacks = [early_stopping]

history = model.fit(
    x_trn, y_trn,
    validation_data=(x_val, y_val),
    batch_size=batch_size,
    epochs=100,
    callbacks=callbacks,
    verbose=1
)

model.evaluate(x_test, y_test)  # 0.8606

y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5).astype(int).reshape(-1)
print(classification_report(y_test, y_pred, digits=4))

# 1. compare n_dense_layers=2 vs. n_dense_layers=3
# result: alternative does not affect result much (if not making it worse a bit)
# 2. compare hidden_dim=150 vs. hidden_dim=300
# result: alternative does not affect result much (if not making it worse a bit)
# 3. compare drop_rate=0.4 vs. drop_rate=0.0
# result: alternative does not affect result much (if not making it worse a bit)
# 4. compare max_len=200 vs. max_len=400
# result: alternative improves result markedly
# 5. compare vocab_size=5000 vs. vocab_size=20000
# result: alternative improves result markedly (note: memory overflow hazard)
# conclusions:
# (a) DNN is not quite responsive to increased model complexity
# (b) but benefits from more training data
# (c) and exacts a heavy toll on memory


## 2.3 DNN Visualization Demo
# https://playground.tensorflow.org/
# Simple Case - Four Squares - 1 hidden layer with 4 nodes - lr=0.03
# linear vs. relu vs. tanh vs. sigmoid
# (1) linear: why linear not working - n layers = 1 layer
# (2) relu: good initialization (>=2 patterns) vs. bad initialization (1 pattern)
# (3) tanh: works but a bit slower
# (4) sigmoid: works but with lr=0.1 which is much slower
# (5) LEGO principle





#############################################
### Part 3. Quasi Sequence-based Approach ###
#############################################

### 3.1 Embedding + Global Average Pooling
# https://www.kaggle.com/allank/simple-keras-fasttext-with-increased-training-data

vocab_size = 5000
max_len = 200
x_trn, y_trn, x_val, y_val, x_test, y_test = load_data_integers(vocab_size=vocab_size, max_len=max_len)

embedding_dim = 50  # vector dimension for embedding
hidden_dim = 150    # number of nodes for dense hidden layer(s)
drop_rate = 0.4     # dropout ratio
batch_size = 64     # batch size for stochastic gradient descent

model = Sequential()
# add an embedding layer
model.add(Embedding(input_dim=vocab_size,
                    output_dim=embedding_dim,
                    input_length=max_len,
                    mask_zero=False))
model.add(Dropout(drop_rate))
# add pooling layer
model.add(GlobalAvgPool1D())
# add first dense layer
model.add(Dense(hidden_dim, activation='relu'))
model.add(Dropout(drop_rate))
# add the output layer
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# show the model
print(model.summary())

early_stopping = EarlyStopping(
    monitor='val_loss',             # default
    min_delta=1e-5,                 # default=0; minimum change counted as an improvement
    patience=5,                     # default=0; number of epochs with no improvement until stopping
    restore_best_weights=True,      # default=False
    verbose=1,                      # verbosity level
    mode='auto',                    # auto-inferred from 'monitor'
    baseline=None,                  # stop training given no improvement over baseline with 'patience'
)
callbacks = [early_stopping]

history = model.fit(
    x_trn, y_trn,
    validation_data=(x_val, y_val),
    batch_size=batch_size,
    epochs=100,
    callbacks=callbacks,
    verbose=1
)

model.evaluate(x_test, y_test)  # 0.8612

y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5).astype(int).reshape(-1)
print(classification_report(y_test, y_pred, digits=4))

# 1. compare embedding_dim=50 vs. embedding_dim=150
# result: alternative does not affect result much (if not making it worse a bit)
# 2. compare drop_rate=0.4 vs. drop_rate=0.0
# result: alternative does not affect result much (if not making it worse a bit)
# 3. compare hidden_dim=150 vs. hidden_dim=250
# result: alternative does not affect result much (if not making it worse a bit)
# 4. compare max_len=200 vs. max_len=400
# result: alternative improves result
# 5. compare vocab_size=5000 vs. vocab_size=20000
# result: alternative does not affect result much (if not making it worse a bit)
# conclusions:
# (a) GlobalAvgPool1D is not quite responsive to increased model complexity
# (b) but benefits from more training data


### 3.2 FastText
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models.utils_any2vec import ft_ngram_hashes

vocab_size = 5000
max_len = 200
x_trn, y_trn, x_val, y_val, x_test, y_test = load_data_words(vocab_size=vocab_size, max_len=max_len)

tokenizer = Tokenizer(char_level=False, num_words=vocab_size)
tokenizer.fit_on_texts(x_trn)

min_ngram = 1
max_ngram = 2
bucket_num = 200000
maxlen = 2500
def transform_texts(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    for idx, seq in enumerate(sequences):
        ngram_list = ft_ngram_hashes(texts[idx], min_ngram, max_ngram, bucket_num)
        seq.extend(ngram_list)
    sequences = pad_sequences(sequences, maxlen=maxlen)
    return sequences

x_trn = transform_texts(x_trn)
x_val = transform_texts(x_val)
x_test = transform_texts(x_test)


embedding_dim = 50  # vector dimension for embedding
hidden_dim = 150    # number of nodes for dense hidden layer(s)
drop_rate = 0.4     # dropout ratio
batch_size = 64     # batch size for stochastic gradient descent

model = Sequential()
# add an embedding layer
model.add(Embedding(input_dim=len(tokenizer.word_index) + bucket_num,
                    output_dim=embedding_dim,
                    input_length=maxlen,
                    mask_zero=False))
model.add(Dropout(drop_rate))
# add pooling layer
model.add(GlobalAvgPool1D())
# add first dense layer
model.add(Dense(hidden_dim, activation='relu'))
model.add(Dropout(drop_rate))
# add the output layer
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# show the model
print(model.summary())

early_stopping = EarlyStopping(
    monitor='val_loss',             # default
    min_delta=1e-5,                 # default=0; minimum change counted as an improvement
    patience=5,                     # default=0; number of epochs with no improvement until stopping
    restore_best_weights=True,      # default=False
    verbose=1,                      # verbosity level
    mode='auto',                    # auto-inferred from 'monitor'
    baseline=None,                  # stop training given no improvement over baseline with 'patience'
)
callbacks = [early_stopping]

history = model.fit(
    x_trn, y_trn,
    validation_data=(x_val, y_val),
    batch_size=batch_size,
    epochs=100,
    callbacks=callbacks,
    verbose=1
)

model.evaluate(x_test, y_test)  # 0.8602

y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5).astype(int).reshape(-1)
print(classification_report(y_test, y_pred, digits=4))


### 3.3 Convolutional Neural Network - CNN
vocab_size = 5000
max_len = 200
x_trn, y_trn, x_val, y_val, x_test, y_test = load_data_integers(vocab_size=vocab_size, max_len=max_len)

embedding_dim = 50  # vector dimension for embedding
filters = 250       # number of filters for Convolution1D
window_size = 3     # window size of the filters
strides = 1         # the stride with which window moves
hidden_dim = 150    # number of nodes for dense hidden layer(s)
drop_rate = 0.4     # dropout ratio
batch_size = 64     # batch size for stochastic gradient descent

model = Sequential()
# add an embedding layer
model.add(Embedding(input_dim=vocab_size,
                    output_dim=embedding_dim,
                    input_length=max_len,
                    mask_zero=False))
model.add(Dropout(drop_rate))
# add an convolution layer
model.add(Conv1D(filters=filters,
                 kernel_size=window_size,
                 strides=strides,
                 padding='valid',
                 activation='relu'))
# add pooling layer
model.add(GlobalMaxPool1D())
# add first dense layer
model.add(Dense(hidden_dim, activation='relu'))
model.add(Dropout(drop_rate))
# add the output layer
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# show the model
print(model.summary())


early_stopping = EarlyStopping(
    monitor='val_loss',             # default
    min_delta=1e-5,                 # default=0; minimum change counted as an improvement
    patience=5,                     # default=0; number of epochs with no improvement until stopping
    restore_best_weights=True,      # default=False
    verbose=1,                      # verbosity level
    mode='auto',                    # auto-inferred from 'monitor'
    baseline=None,                  # stop training given no improvement over baseline with 'patience'
)
callbacks = [early_stopping]

history = model.fit(
    x_trn, y_trn,
    validation_data=(x_val, y_val),
    batch_size=batch_size,
    epochs=100,
    callbacks=callbacks,
    verbose=1
)

model.evaluate(x_test, y_test)  # 0.8656

y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5).astype(int).reshape(-1)
print(classification_report(y_test, y_pred, digits=4))

# 1. compare embedding_dim=50 vs. embedding_dim=150
# result: alternative improves result a bit
# 2. compare drop_rate=0.4 vs. drop_rate=0.0
# result: alternative improves result a tiny little bit
# 3. compare hidden_dim=150 vs. hidden_dim=250
# result: alternative does not affect result much (if not making it worse a bit)
# 4. compare max_len=200 vs. max_len=400
# result: alternative improves result markedly
# 5. compare vocab_size=5000 vs. vocab_size=20000
# result: alternative does not affect result much (if not making it worse a bit)
# conclusions:
# (a) Conv1D is more responsive to increased model complexity
# (b) and benefits from more training data
# try max_len=400; embedding_dim=150; drop_rate=0.0;


### 3.4 Multi-channel CNN
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate

input = Input(shape=(max_len,))
embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len, mask_zero=False)(input)
embedding = Dropout(drop_rate)(embedding)

conv1 = Conv1D(filters=filters, kernel_size=2, strides=strides, padding='valid', activation='relu')(embedding)
conv1 = GlobalMaxPool1D()(conv1)
conv2 = Conv1D(filters=filters, kernel_size=3, strides=strides, padding='valid', activation='relu')(embedding)
conv2 = GlobalMaxPool1D()(conv2)
conv3 = Conv1D(filters=filters, kernel_size=5, strides=strides, padding='valid', activation='relu')(embedding)
conv3 = GlobalMaxPool1D()(conv3)

hidden = Concatenate()([conv1, conv2, conv3])
hidden = Dense(hidden_dim, activation='relu')(hidden)
hidden = Dropout(drop_rate)(hidden)
output = Dense(1, activation='sigmoid')(hidden)

model = Model(inputs=input, outputs=output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

early_stopping = EarlyStopping(
    monitor='val_loss',             # default
    min_delta=1e-5,                 # default=0; minimum change counted as an improvement
    patience=5,                     # default=0; number of epochs with no improvement until stopping
    restore_best_weights=True,      # default=False
    verbose=1,                      # verbosity level
    mode='auto',                    # auto-inferred from 'monitor'
    baseline=None,                  # stop training given no improvement over baseline with 'patience'
)

callbacks = [early_stopping]

history = model.fit(
    x_trn, y_trn,
    validation_data=(x_val, y_val),
    batch_size=batch_size,
    epochs=100,
    callbacks=callbacks,
    verbose=1
)

model.evaluate(x_test, y_test)  # 0.8683

y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5).astype(int).reshape(-1)
print(classification_report(y_test, y_pred, digits=4))



#######################################
### Part 4. Sequence-based Approach ###
#######################################

### 4.1 Long-short Term Memory - LSTM
# http://dprogrammer.org/rnn-lstm-gru
vocab_size = 5000
max_len = 200
x_trn, y_trn, x_val, y_val, x_test, y_test = load_data_integers(vocab_size=vocab_size, max_len=max_len)

embedding_dim = 50
lstm_dim = 128      # vector dimension for hidden state
hidden_dim = 150
batch_size = 64     # batch size for stochastic gradient descent
drop_rate = 0.4

model = Sequential()
# add an embedding layer
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
model.add(Dropout(drop_rate))
# add an LSTM layer
model.add(LSTM(lstm_dim))     # parameters: 4 * ((lstm_dim + embedding_dim + 1) * lstm_dim)
# add a dense layer
model.add(Dense(hidden_dim, activation='relu'))
model.add(Dropout(drop_rate))
# add the output layer
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

early_stopping = EarlyStopping(
    monitor='val_loss',             # default
    min_delta=1e-5,                 # default=0; minimum change counted as an improvement
    patience=5,                     # default=0; number of epochs with no improvement until stopping
    restore_best_weights=True,      # default=False
    verbose=1,                      # verbosity level
    mode='auto',                    # auto-inferred from 'monitor'
    baseline=None,                  # stop training given no improvement over baseline with 'patience'
)
callbacks = [early_stopping]

model.fit(x_trn, y_trn,
          validation_data=(x_val, y_val),
          batch_size=batch_size,
          epochs=100,
          callbacks=callbacks,
          verbose=5)

model.evaluate(x_test, y_test)  # 0.7411


### 4.2 Bidirectional LSTM - BiLSTM
vocab_size = 5000
max_len = 200
x_trn, y_trn, x_val, y_val, x_test, y_test = load_data_integers(vocab_size=vocab_size, max_len=max_len)

embedding_dim = 50
lstm_dim = 128      # vector dimension for hidden state
hidden_dim = 150
batch_size = 64     # batch size for stochastic gradient descent
drop_rate = 0.4

model = Sequential()
# add an embedding layer
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
model.add(Dropout(drop_rate))
# add an LSTM layer
model.add(Bidirectional(LSTM(lstm_dim)))
# add a dense layer
model.add(Dense(hidden_dim, activation='relu'))
model.add(Dropout(drop_rate))
# add the output layer
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

early_stopping = EarlyStopping(
    monitor='val_loss',             # default
    min_delta=1e-5,                 # default=0; minimum change counted as an improvement
    patience=5,                     # default=0; number of epochs with no improvement until stopping
    restore_best_weights=True,      # default=False
    verbose=1,                      # verbosity level
    mode='auto',                    # auto-inferred from 'monitor'
    baseline=None,                  # stop training given no improvement over baseline with 'patience'
)

callbacks = [early_stopping]

model.fit(x_trn, y_trn,
          validation_data=(x_val, y_val),
          batch_size=batch_size,
          epochs=100,
          callbacks=callbacks,
          verbose=5)

model.evaluate(x_test, y_test)  # 0.8442


### 4.3 Stacked LSTM
vocab_size = 5000
max_len = 200
x_trn, y_trn, x_val, y_val, x_test, y_test = load_data_integers(vocab_size=vocab_size, max_len=max_len)

embedding_dim = 50
lstm_dim = 128      # vector dimension for hidden state
hidden_dim = 150
batch_size = 64     # batch size for stochastic gradient descent
drop_rate = 0.4

model = Sequential()

model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
model.add(Dropout(drop_rate))

model.add(LSTM(lstm_dim, return_sequences=True))
model.add(LSTM(lstm_dim))

model.add(Dense(hidden_dim, activation='relu'))
model.add(Dropout(drop_rate))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


early_stopping = EarlyStopping(
    monitor='val_loss',             # default
    min_delta=1e-5,                 # default=0; minimum change counted as an improvement
    patience=5,                     # default=0; number of epochs with no improvement until stopping
    restore_best_weights=True,      # default=False
    verbose=1,                      # verbosity level
    mode='auto',                    # auto-inferred from 'monitor'
    baseline=None,                  # stop training given no improvement over baseline with 'patience'
)
callbacks = [early_stopping]

model.fit(x_trn, y_trn,
          validation_data=(x_val, y_val),
          batch_size=batch_size,
          epochs=100,
          callbacks=callbacks,
          verbose=5)

model.evaluate(x_test, y_test)  # 0.6929


### 4.4 CNN + LSTM
vocab_size = 5000
max_len = 200
x_trn, y_trn, x_val, y_val, x_test, y_test = load_data_integers(vocab_size=vocab_size, max_len=max_len)

embedding_dim = 50  # vector dimension for embedding
filters = 250       # number of filters for Convolution1D
window_size = 3     # window size of the filters
strides = 1         # the stride with which window moves
pool_size = 4
lstm_dim = 128      # vector dimension for embedding
hidden_dim = 150    # number of nodes for dense hidden layer(s)
drop_rate = 0.4     # dropout ratio
batch_size = 64     # batch size for stochastic gradient descent

model = Sequential()
# add an embedding layer
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
model.add(Dropout(drop_rate))
# add a CNN layer
model.add(Conv1D(filters=filters,
                 kernel_size=window_size,
                 strides=strides,
                 padding='valid',
                 activation='relu'))
model.add(MaxPooling1D(pool_size=pool_size))
# add a LSTM layer
model.add(LSTM(lstm_dim))
# add a dense layer
model.add(Dense(hidden_dim, activation='relu'))
model.add(Dropout(drop_rate))
# add the output layer
model.add(Dense(1, activation='sigmoid'))
# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

early_stopping = EarlyStopping(
    monitor='val_loss',             # default
    min_delta=1e-5,                 # default=0; minimum change counted as an improvement
    patience=5,                     # default=0; number of epochs with no improvement until stopping
    restore_best_weights=True,      # default=False
    verbose=1,                      # verbosity level
    mode='auto',                    # auto-inferred from 'monitor'
    baseline=None,                  # stop training given no improvement over baseline with 'patience'
)
callbacks = [early_stopping]

model.fit(x_trn, y_trn,
          validation_data=(x_val, y_val),
          batch_size=batch_size,
          epochs=100,
          callbacks=callbacks,
          verbose=5)
model.evaluate(x_test, y_test)  # 0.8748



############################
### Part 5. Transformers ###
############################

### 5.1 Bidirectional Encoder Representations from Transformers - BERT
# https://tensorflow.google.cn/text/tutorials/nmt_with_attention
# https://tensorflow.google.cn/text/tutorials/transformer
# https://tensorflow.google.cn/text/tutorials/classify_text_with_bert

from transformers import BertTokenizer, BertConfig, TFBertModel
from transformers import AutoTokenizer, AutoConfig, TFAutoModel
from tensorflow.keras.layers import Input, Lambda, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

model_fullname = "bert-base-uncased"
model_fullname = "distilbert-base-uncased"
batch_size = 8

vocab_size = 5000
max_len = 200
x_trn_raw, y_trn, x_val_raw, y_val, x_tst_raw, y_tst = load_data_words(vocab_size=vocab_size, max_len=max_len)

tokenizer = AutoTokenizer.from_pretrained(model_fullname)
x_trn = tokenizer(x_trn_raw[:100], return_tensors='tf', max_length=max_len, padding="max_length", truncation=True)
x_val = tokenizer(x_val_raw[:100], return_tensors='tf', max_length=max_len, padding="max_length", truncation=True)
x_tst = tokenizer(x_tst_raw[:100], return_tensors='tf', max_length=max_len, padding="max_length", truncation=True)

config = AutoConfig.from_pretrained(
    model_fullname,
    max_length=max_len,
    finetuning_task=True,
    output_hidden_states=False,
    output_attentions=False,
)
BertishLayer = TFAutoModel.from_pretrained(model_fullname, config=config)

# inspect Bertish output
docs = ["let's test it", "see what happens and why it is happening"]
sample = tokenizer(docs, return_tensors='tf', max_length=10, padding="max_length", truncation=True)
out = BertishLayer(**sample)

# build model
input_ids = Input(shape=(max_len,), dtype=tf.int32, name='id')
input_mask = Input(shape=(max_len,), dtype=tf.int32, name='mask')
inputs = [input_ids, input_mask]
hidden_state = BertishLayer(inputs)
hidden_state = Lambda(lambda x: x[0][:, 0, :])(hidden_state)
hidden_state = Dropout(0.2)(hidden_state)
output = Dense(1, activation='sigmoid')(hidden_state)
model = Model(inputs, output)
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.00003), metrics=['accuracy'])
model.summary()

early_stopping = EarlyStopping(
    monitor='val_loss',             # default
    min_delta=1e-5,                 # default=0; minimum change counted as an improvement
    patience=2,                     # default=0; number of epochs with no improvement until stopping
    restore_best_weights=True,      # default=False
    verbose=2,                      # verbosity level
    mode='auto',                    # auto-inferred from 'monitor'
    baseline=None,                  # stop training given no improvement over baseline with 'patience'
)
callbacks = [early_stopping]

model.fit([x_trn['input_ids'], x_trn['attention_mask']], y_trn[:100],
          # validation_data=([x_val['input_ids'], x_val['attention_mask']], y_val),
          batch_size=batch_size,
          epochs=3,
          callbacks=callbacks,
          verbose=2)

model.evaluate([x_tst['input_ids'], x_tst['attention_mask']], y_tst)  # 0.8883




