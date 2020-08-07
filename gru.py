import numpy as np

np.random.seed(42)
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from keras.backend import argmax
import warnings

warnings.filterwarnings('ignore')

import os

os.environ['OMP_NUM_THREADS'] = '4'

EMBEDDING_FILE = 'data/crawl-300d-2M.vec'

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
submission = pd.read_csv('data/submit_sample.csv',header=None)

X_train = train["description"].fillna("fillna").values
y_train = train["jobflag"].values
X_test = test["description"].fillna("fillna").values

onehot_encoder = OneHotEncoder(sparse=False)
reshaped = y_train.reshape(len(y_train), 1)
y_train_onehot = onehot_encoder.fit_transform(reshaped)

max_features = 30000
maxlen = 100
embed_size = 300

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE, encoding="utf-8"))

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
print(nb_words)
embedding_matrix = np.zeros((nb_words, embed_size))
print(embedding_matrix.shape)
for word, i in word_index.items():
    if i >= nb_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


class Evaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            label = np.argmax(self.y_val, 1)
            predict = np.argmax(y_pred, 1)
            score = f1_score(label, predict)
            print("\n f1 score - epoch: %d - score: %.6f \n" % (epoch + 1, score))


def get_model():
    inp = Input(shape=(maxlen,))
    x = Embedding(nb_words, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    preout = Dense(320, activation='relu')(conc)
    outp = Dense(4, activation="softmax")(preout)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


model = get_model()
print(model.summary())

batch_size = 32
epochs = 20

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train_onehot, train_size=0.95, random_state=233)
eval_score = Evaluation(validation_data=(X_val, y_val), interval=1)

hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                 callbacks=[eval_score], verbose=2)

y_pred = model.predict(x_test, batch_size=1024)
pred = np.argmax(y_pred,1)
submission.ix[:, 1] = pred
submission.to_csv('submission.csv', index=False)
