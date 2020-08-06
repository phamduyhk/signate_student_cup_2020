# coding: utf-8
"""
Author: Pham Duy
Created date: 2020/08/05
"""
import re
import logging

import numpy as np
import pandas as pd
import spacy
import torch
from joblib import Memory
from torchtext import data
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut

NLP = spacy.load('en')
MAX_CHARS = 20000
LOGGER = logging.getLogger("SIGNATE STUDENT CUP DATASET")
MEMORY = Memory(cachedir="cache/", verbose=1)


def tokenizer(description):
    description = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(description))
    description = re.sub(r"[ ]+", " ", description)
    description = re.sub(r"\!+", "!", description)
    description = re.sub(r"\,+", ",", description)
    description = re.sub(r"\?+", "?", description)
    if (len(description) > MAX_CHARS):
        description = description[:MAX_CHARS]
    return [x.text for x in NLP.tokenizer(description) if x.text != " "]


def prepare_csv():
    df_train = pd.read_csv("data/train_data.csv")
    df_train["description"] = df_train.description.str.replace("\n", " ")
    df_train.to_csv("cache/dataset_train.csv", index=False)
    df_test = pd.read_csv("data/test.csv")
    df_test["description"] = df_test.description.str.replace("\n", " ")
    df_test.to_csv("cache/dataset_test.csv", index=False)


@MEMORY.cache
def read_files(fix_length=100, lower=False, vectors=None):
    if vectors is not None:
        # pretrain vectors only support all lower case
        lower = True
    LOGGER.debug("Preparing CSV files...")
    prepare_csv()
    description = data.Field(
        sequential=True,
        fix_length=fix_length,
        tokenize=tokenizer,
        pad_first=True,
        lower=lower
    )
    LOGGER.debug("Reading train csv file...")
    train = data.TabularDataset(
        path='cache/dataset_train.csv', format='csv', skip_header=True,
        fields=[
            ('id', None),
            ('description', description),
            ('jobflag', data.Field(
                use_vocab=False, sequential=False))
        ])
    LOGGER.debug("Reading test csv file...")
    test = data.TabularDataset(
        path='cache/dataset_test.csv', format='csv', skip_header=True,
        fields=[
            ('id', None),
            ('description', description)
        ])
    LOGGER.debug("Building vocabulary...")
    description.build_vocab(
        train, test,
        max_size=20000,
        min_freq=50,
        vectors=vectors
    )
    LOGGER.debug("Done preparing the datasets")
    return train.examples, test.examples, description


"""
K FOLD USAGE

# # k分割交差検証（k-fold cross-validation）
# kf = KFold(n_splits=5, shuffle=True)
# print(train_val_ds)
# for train_index, test_index in kf.split(train_val_ds):
#     print(train_index, test_index)
#
# # 一つ抜き交差検証（leave-one-out cross-validation）
# loo = LeaveOneOut()
# for train, test in loo.split(train_val_ds):
#     print("train {}, test {}".format(train, test))
#
# # 層化k分割交差検証（stratified k-fold cross-validation）
# skf = StratifiedKFold(n_splits=3)
# for train, test in skf.split(train_val_ds):
#     print("train {}, test {}".format(train, test))
"""


def get_dataset(split_mode="KFold", fix_length=100, lower=False, vectors=None, n_folds=5, seed=999):
    """

    Args:
        split_mode:
            0: KFold
            1: LeaveOneOut
            2: StratifiedKFold
        fix_length:
        lower:
        vectors:
        n_folds:
        seed:

    Returns:
        (train, valid), test
    """
    train_exs, test_exs, description = read_files(
        fix_length=fix_length, lower=lower, vectors=vectors)

    kf = KFold(n_splits=n_folds, random_state=seed)
    loo = LeaveOneOut()
    skf = StratifiedKFold(n_splits=n_folds, random_state=seed)
    if split_mode is "LeaveOneOut":
        splicer = loo
    elif split_mode is "StratifiedKFold":
        splicer = skf
    else:
        splicer = kf

    fields = [
        ('id', None),
        ('description', description),
        ('jobflag', data.Field(
            use_vocab=False, sequential=False))
    ]

    def iter_folds():
        train_exs_arr = np.array(train_exs)
        for train_idx, val_idx in splicer.split(train_exs_arr):
            yield (
                data.Dataset(train_exs_arr[train_idx], fields),
                data.Dataset(train_exs_arr[val_idx], fields),
            )

    test = data.Dataset(test_exs, fields[:2])
    return iter_folds(), test


def get_iterator(dataset, batch_size, device=0, train=True, shuffle=True, repeat=False):
    dataset_iter = data.Iterator(
        dataset, batch_size=batch_size, device=device,
        train=train, shuffle=shuffle, repeat=repeat,
        sort=False
    )
    return dataset_iter


def preprocessing(file_path):
    """

    Args:
        file_path: dir to input file path

    Returns:
        torch formed dataloader
    """
    return None


if __name__ == '__main__':
    train_val_generator, test_dataset = get_dataset(
        fix_length=100, lower=True, vectors="fasttext.en.300d",
        n_folds=2, seed=123
    )
    for fold, (train_dataset, val_dataset) in enumerate(train_val_generator):
        for batch in get_iterator(
                train_dataset, batch_size=32, train=True,
                shuffle=True, repeat=False
        ):
            x = batch.description.data
            y = batch.jobflag
            print(x)
            print(y)
            break
