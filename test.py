from imblearn.over_sampling import SMOTE
import datetime
import pandas as pd
import numpy as np
import os
from preprocessing import normalize_comment
import os
import lightgbm as lgb
import warnings
from keras.backend import argmax
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import Callback
from keras.preprocessing import text, sequence
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, CuDNNGRU, Conv1D
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate
from keras.layers.recurrent import LSTM
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, make_scorer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import pandas as pd
import keras
from keras import backend as K
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split

ensemble_files = ["./output/prob_albert-xxlarge-v2_512_10ep.csv", "./output/prob_albert-xxlarge-v2_128_10ep.csv", "output/prob_bert-large-cased_256_10ep.csv",
                  "output/prob_bert-base-cased_512_10ep.csv", "output/prob_bert-base-cased_128_10ep.csv"]


# ensemble_files = ["./output/prob_albert-xxlarge-v2_512_10ep.csv","./output/prob_albert-xxlarge-v2_128_10ep.csv"]

submit = pd.read_csv("./data/submit_sample.csv", header=None)
all_prob = None
for file in ensemble_files:
    df = pd.read_csv(file, header=None)
    array = np.zeros(shape=(len(df), 4))
    for i, item in enumerate(df.iloc[:, 2]):
        arr = eval(item)
        array[i] = arr

    s = 0
    for item in array:
        for v in item:
            if v > 0.6:
                s += 1
                break

    print(s/len(df))

from sklearn.datasets import make_classification

X, y = make_classification(
    n_classes=2, class_sep=1.5, weights=[0.9, 0.1],
    n_informative=3, n_redundant=1, flip_y=0,
    n_features=20, n_clusters_per_class=1,
    n_samples=100, random_state=10
)

print(X.shape)
df = pd.DataFrame(X)
df['target'] = y
df.target.value_counts().plot(kind='bar', title='Count (target)')



pldata = pd.read_csv("./data/train.csv")

label = pldata["jobflag"]
lbcount = {"1": 0, "2": 0, "3": 0, "4": 0}
for item in label:
    for l in ["1", "2", "3", "4"]:
        if str(item) == l:
            lbcount[str(l)] += 1
print(lbcount)

pldata = pd.read_csv("./data/pseudo_train_gru.csv")

label = pldata["jobflag"]
lbcount = {"1": 0, "2": 0, "3": 0, "4": 0}
for item in label:
    for l in ["1", "2", "3", "4"]:
        if str(item) == l:
            lbcount[str(l)] += 1
print(lbcount)


from imblearn.under_sampling import RandomUnderSampler
pldata = pd.read_csv("./data/train.csv")
# Separate input features and target
y_train = pldata["jobflag"]
X_train = pldata["description"]

label1 = y_train[y_train==1].count()
label2 = y_train[y_train==2].count()
label3 = y_train[y_train==3].count()
label4 = y_train[y_train==4].count()
print('label1 count: {}'.format(label1))
print('label2 count: {}'.format(label2))
print('label3 count: {}'.format(label3))
print('label4 count: {}'.format(label4))

maxlen = 256
max_features = 300000

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train))


# for previous version
X_train = tokenizer.texts_to_sequences(X_train)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)



# label1 count: 946
# label2 count: 367
# label3 count: 2296
# label4 count: 748
#### Traindata
#label1 count: 624
# label2 count: 348
# label3 count: 1376
# label4 count: 583

# ランダムにunder-sampling
rus = RandomUnderSampler(random_state=0)
X_train_resampled = rus.fit(x_train, y_train)
print(rus.get_params)


# print(X_train_resampled)
# print(X_train_resampled.shape)
# print(y_train_resampled)


from transformers import BertTokenizer, BertForSequenceClassification
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=4)
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
labels = torch.tensor([3]).unsqueeze(0)  # Batch size 1
outputs = model(**inputs)
print(outputs)
loss, logits = outputs[:2]