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

X_train = train["description"].fillna("fillna").tolist()
y_train = train["jobflag"].values
X_test = test["description"].fillna("fillna").values


onehot_encoder = OneHotEncoder(sparse=False)
reshaped = y_train.reshape(len(y_train), 1)
y_train_onehot = onehot_encoder.fit_transform(reshaped)

# SETTINGS
max_features = 30000
maxlen = 100
embed_size = 300

model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

num_added_toks = tokenizer.add_tokens(X_train)
print('We have added', num_added_toks, 'tokens')
model.resize_token_embeddings(len(tokenizer))

input_ids = torch.tensor(tokenizer.encode(X_train, add_special_tokens=False)).unsqueeze(0)
print(input_ids.shape)
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

