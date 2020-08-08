import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier

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
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE, encoding="utf-8"))

word_index = tokenizer.word_index
nb_words = max(max_features, len(word_index)+1)
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
            score = f1_score(label, predict, average='macro')
            print("\n f1 score - epoch: %d - score: %.6f \n" % (epoch + 1, score))


def get_gru_model():
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

BATCH_SIZE = 512
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
EPOCHS = 4
MAX_LEN = 220
def get_gru_lstm_model(embedding_matrix):
    words = Input(shape=(None,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(LSTM_UNITS, return_sequences=True))(x)
    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x)

    hidden = concatenate([
        GlobalMaxPooling1D()(x),
        GlobalAveragePooling1D()(x),
    ])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    result = Dense(4, activation='softmax')(hidden)

    model = Model(inputs=words, outputs=result)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

model_type = "gru_lstm"
if model_type is "gru_stlm":
    model = get_gru_lstm_model(embedding_matrix)
    if os.path.exists("grulstm_param.hdf5"):
        model.load_weights("grulstm_param.hdf5")
elif model_type is "gru":
    model = get_gru_model()
    if os.path.exists("gru_param.hdf5"):
        model.load_weights("gru_param.hdf5")

print(model.summary())

batch_size = 32
epochs = 5
num_folds = 5
kf = KFold(n_splits=num_folds)
fold_no = 1
for train, val in kf.split(x_train, y_train_onehot):
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no}/{num_folds} ...')
    # X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train_onehot, train_size=0.95, random_state=233)
    eval_score = Evaluation(validation_data=(x_train[val], y_train_onehot[val]), interval=1)

    hist = model.fit(x_train[train], y_train_onehot[train], batch_size=batch_size, epochs=epochs,
                     validation_data=(x_train[val], y_train_onehot[val]),
                     callbacks=[eval_score], verbose=2)
    fold_no += 1
if model_type is "gru_lstm":
    model.save_weights("grulstm_param.hdf5")
elif model_type is "gru":
    model.save_weights("gru.hdf5")
y_pred = model.predict(x_test, batch_size=1024)
pred = np.argmax(y_pred, 1)
submission.iloc[:, 1] = pred + 1
submission.to_csv('submission.csv', index=False, header=None)
