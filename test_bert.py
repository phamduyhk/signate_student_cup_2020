import torch
from transformers import BertTokenizer, BertModel


import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier

from bert_embedding import BertEmbedding

np.random.seed(42)
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers.recurrent import LSTM
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, CuDNNGRU, Conv1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from keras.backend import argmax
import warnings

from preprocessing import normalize_comment

warnings.filterwarnings('ignore')

import os

os.environ['OMP_NUM_THREADS'] = '4'

EMBEDDING_FILE = 'data/crawl-300d-2M.vec'

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
submission = pd.read_csv('data/submit_sample.csv', header=None)

X_train = train["description"].fillna("fillna").values
y_train = train["jobflag"].values
X_test = test["description"].fillna("fillna").values

onehot_encoder = OneHotEncoder(sparse=False)
reshaped = y_train.reshape(len(y_train), 1)
y_train_onehot = onehot_encoder.fit_transform(reshaped)

# SETTINGS
max_features = 30000
maxlen = 100
embed_size = 300

# Define per-fold score containers <-- these are new
acc_per_fold = []
loss_per_fold = []

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
sentence = X_train[0]
np_normalize = np.vectorize(normalize_comment)
normalized_X_train = np_normalize(X_train)
X_train = tokenizer.texts_to_sequences(normalized_X_train)
X_test = tokenizer.texts_to_sequences(X_test)
print(X_train[0])
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)

print(tokenizer)

print(x_train[0])



model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
input_ids = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True)).unsqueeze(0)
outputs = model(input_ids)
last_hidden_states = outputs[0]

print(last_hidden_states.size())

# input_ids = torch.tensor(tokenizer.encode('The', add_special_tokens=True)).unsqueeze(0)
# outputs = model(input_ids)
# last_hidden_states = outputs[0]
#
# print(last_hidden_states.size())

# bert_embedding = BertEmbedding()
# result = bert_embedding(sentence)
# print(result)
# print(result[0].shape)