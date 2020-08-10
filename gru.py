import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier

np.random.seed(42)
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, make_scorer
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers.recurrent import LSTM
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, CuDNNGRU, Conv1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from keras.utils import to_categorical
from keras.backend import argmax
import warnings
import lightgbm as lgb
import os

warnings.filterwarnings(action='ignore', category=DeprecationWarning, module='sklearn')

from preprocessing import normalize_comment

os.environ['OMP_NUM_THREADS'] = '4'

# EMBEDDING_FILE = 'data/crawl-300d-2M.vec'
EMBEDDING_FILE = 'data/wiki-news-300d-1M.vec'

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
submission = pd.read_csv('data/submit_sample.csv', header=None)

X_train = train["description"].fillna("fillna").values
y_train = train["jobflag"].values
X_test = test["description"].fillna("fillna").values

X_tra = []
X_val = []
y_tra, y_val = [], []
label_count = {'1': 0, '2': 0, '3': 0, '4': 0}
# {'1': 624, '2': 348, '3': 1376, '4': 583}
for i, v in enumerate(y_train):
    if v == 1:
        if label_count['1'] <= 500:
            y_tra.append(v)
            X_tra.append(X_train[i])
            label_count['1'] += 1
        else:
            y_val.append(v)
            X_val.append(X_train[i])
            label_count['1'] += 1
    if v == 2:
        if label_count['2'] <= 300:
            y_tra.append(v)
            X_tra.append(X_train[i])
            label_count['2'] += 1
        else:
            y_val.append(v)
            X_val.append(X_train[i])
            label_count['2'] += 1
    if v == 3:
        if label_count['3'] <= 500:
            y_tra.append(v)
            X_tra.append(X_train[i])
            label_count['3'] += 1
        else:
            y_val.append(v)
            X_val.append(X_train[i])
            label_count['3'] += 1
    if v == 4:
        if label_count['4'] <= 500:
            y_tra.append(v)
            X_tra.append(X_train[i])
            label_count['4'] += 1
        else:
            y_val.append(v)
            X_val.append(X_train[i])
            label_count['4'] += 1

y_tra = np.array(y_tra)
y_val = np.array(y_val)
onehot_encoder = OneHotEncoder(sparse=False)
reshaped = y_tra.reshape(len(y_tra), 1)
y_tra = onehot_encoder.fit_transform(reshaped)

onehot_encoder = OneHotEncoder(sparse=False)
y_val_reshaped = y_val.reshape(len(y_val), 1)
y_val = onehot_encoder.fit_transform(y_val_reshaped)

print(y_tra.shape)
print(y_val.shape)
# SETTINGS
max_features = 30000
maxlen = 1700
embed_size = 300

# Define per-fold score containers <-- these are new
acc_per_fold = []
loss_per_fold = []

scaler = StandardScaler()


def z_normalize(data):
    scaler.fit(data)
    return scaler.transform(data)


tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))

X_tra = tokenizer.texts_to_sequences(X_tra)
X_val = tokenizer.texts_to_sequences(X_val)
X_test = tokenizer.texts_to_sequences(X_test)

X_tra = sequence.pad_sequences(X_tra, maxlen=maxlen)
X_val = sequence.pad_sequences(X_val, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)

# for previous version
X_train = tokenizer.texts_to_sequences(X_train)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
reshaped = y_train.reshape(len(y_train), 1)
y_train_onehot = onehot_encoder.fit_transform(reshaped)


# x_train = z_normalize(x_train)
# x_test = z_normalize(x_test)


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE, encoding="utf-8"))

word_index = tokenizer.word_index
nb_words = max(max_features, len(word_index) + 1)
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


model_type = "gru"
load_model = False
if model_type is "gru_stlm":
    model = get_gru_lstm_model(embedding_matrix)
    if load_model is True and os.path.exists("grulstm_param.hdf5"):
        model.load_weights("grulstm_param.hdf5")
        print("loaded weight")
elif model_type is "gru":
    model = get_gru_model()
    if load_model is True and os.path.exists("gru_param.hdf5"):
        model.load_weights("gru_param.hdf5")
        print("loaded weight")

print(model.summary())


def train_with_LGBM():
    scores = []
    param = {
        'metric': 'multi_logloss',
        'num_class': 4,
    }
    stacker = lgb.LGBMClassifier(max_depth=3, n_estimators=125, num_leaves=10, boosting_type="gbdt",
                                 learning_rate=0.1, feature_fraction=0.45, colsample_bytree=0.45, bagging_fraction=0.8,
                                 bagging_freq=5, reg_lambda=0.2)
    score = cross_val_score(stacker, x_train, y_train, cv=5,
                            scoring=make_scorer(f1_score, average='weighted', labels=[4]))
    print("Score:", score)
    scores.append(np.mean(score))
    stacker.fit(x_train, y_train)
    pred = stacker.predict_proba(X_test)[:, 1]
    print(pred)
    print("CV score:", np.mean(scores))


def train_with_cv(batch_size=32, epochs=10, num_folds=5):
    """
    Training with cv
    Returns:

    """
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


def train(batch_size=32, epochs=10):
    """
    Training with out cv
    Returns:

    """
    X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train_onehot, train_size=0.95, random_state=233)
    print(y_tra.shape)
    print(y_val.shape)
    eval_score = Evaluation(validation_data=(X_val, y_val), interval=1)
    hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                     callbacks=[eval_score], verbose=2)

    y_pred = model.predict(x_test, batch_size=1024)
    pred = np.argmax(y_pred, 1)
    submission.iloc[:, 1] = pred + 1
    submission.to_csv('submission.csv', index=False, header=None)
    return hist


batch_size = 32
epochs = 10
train(batch_size=batch_size, epochs=epochs)
# train_with_LGBM()
