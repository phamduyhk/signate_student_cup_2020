# Libraries
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch

# preprocessing module
from utils.dataloader import Preprocessing
from preprocessing import get_iterator, get_dataset

# Preliminaries

from torchtext.data import Field, TabularDataset, BucketIterator, Iterator

# Models

import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

# Training

import torch.optim as optim

# Evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import seaborn as sns
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold

preprocessor = Preprocessing()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Model parameter
MAX_SEQ_LEN = 128
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

# Fields

label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                   fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
fields = [('text', text_field), ('label', label_field)]

# SETTINGs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
destination_folder = "./output"
if not os.path.exists(destination_folder):
    os.mkdir(destination_folder)
source_folder = "./data"

# Get data (ver transformer)
vector_list_file = "wiki-news-300d-1M.vec"
max_sequence_length = 512
batch_size = 128


# # TabularDataset Ver
# train_val_ds, test = TabularDataset.splits(path=source_folder, train='train_data.csv',
#                                            test='test_data.csv', format='CSV', fields=fields, skip_header=True)
#
# train, valid = train_val_ds.split(
#     split_ratio=0.7, random_state=random.seed(2395))
#
# # train_iter = BucketIterator(train, batch_size=batch_size, sort_key=lambda x: len(x.text),
# #                             device=device, train=True, sort=True, sort_within_batch=True, shuffle=False)
# # valid_iter = BucketIterator(valid, batch_size=batch_size, sort_key=lambda x: len(x.text),
# #                             device=device, train=True, sort=True, sort_within_batch=True, shuffle=False)
#
#
# train_iter = BucketIterator(train, batch_size=batch_size, device=device, train=True, shuffle=False)
# valid_iter = BucketIterator(valid, batch_size=batch_size,
#                             device=device, train=True, shuffle=False)
# test_iter = Iterator(test, batch_size=batch_size, device=device, train=False, shuffle=False, sort=False)
#
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


class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name, num_labels=4)

    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]

        return loss, text_fea


# Save and Load Functions

def save_checkpoint(save_path, model, valid_loss):
    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    if save_path == None:
        return

    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


# Training Function

def train(model,
          optimizer,
          criterion=nn.CrossEntropyLoss(),
          file_path=destination_folder,
          num_epochs=100,
          n_folds=5,
          best_valid_loss=float("Inf")):
    # initialize running values
    running_loss = 0.0
    running_f1_score = 0.0
    valid_running_loss = 0.0
    valid_running_f1_score = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    valid_f1_score_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):
        train_val_generator, test_dataset = get_dataset(split_mode="KFold",
                                                        fix_length=max_sequence_length, lower=True, vectors="fasttext.en.300d",
                                                        n_folds=n_folds, seed=123
                                                        )
        for fold, (train_dataset, val_dataset) in enumerate(train_val_generator):
            # training step
            for batch in get_iterator(
                    train_dataset, batch_size=batch_size, train=True,
                    shuffle=True, repeat=False
            ):
                text = torch.transpose(batch.description.data, 0, 1)
                labels = batch.jobflag
                labels = labels.type(torch.LongTensor)
                labels = labels.to(device)
                text = text.type(torch.LongTensor)
                text = text.to(device)
                output = model(text, labels)
                loss, pred = output

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update running values
                running_loss += loss.item()
                label_cpu = labels.cpu()
                pred_cpu = torch.argmax(pred, 1).cpu()
                running_f1_score += f1_score(label_cpu, pred_cpu, average='macro')
                global_step += 1

            # evaluation step
            model.eval()
            with torch.no_grad():
                # validation loop
                for batch in get_iterator(
                        val_dataset, batch_size=batch_size, train=True,
                        shuffle=True, repeat=False
                ):
                    text = torch.transpose(batch.description.data, 0, 1)
                    labels = batch.jobflag
                    labels = labels.type(torch.LongTensor)
                    labels = labels.to(device)
                    text = text.type(torch.LongTensor)
                    text = text.to(device)
                    output = model(text, labels)
                    loss, preds = output

                    valid_running_loss += loss.item()
                    labels_cpu = labels.cpu()
                    preds_cpu = torch.argmax(preds, 1).cpu()
                    valid_running_f1_score += f1_score(labels_cpu, preds_cpu, average='macro')

            # evaluation
            average_train_loss = running_loss / len(train_dataset)
            average_f1_score = running_f1_score / len(train_dataset)
            average_valid_loss = valid_running_loss / len(val_dataset)
            average_valid_f1_score = valid_running_f1_score / len(val_dataset)
            train_loss_list.append(average_train_loss)
            valid_loss_list.append(average_valid_loss)
            valid_f1_score_list.append(average_f1_score)
            global_steps_list.append(global_step)

            # resetting running values
            running_loss = 0.0
            running_f1_score = 0.0
            valid_running_loss = 0.0
            valid_running_f1_score = 0.0
            model.train()

            # print progress
            print(
                'Epoch [{}/{}], fold {}/{}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Train F1 Score: {:.4f}, Valid F1 Score: {:.4f}'
                    .format(epoch + 1, num_epochs, fold, n_folds,
                            average_train_loss, average_valid_loss, average_f1_score, average_valid_f1_score))

            # checkpoint
            if best_valid_loss > average_valid_loss:
                best_valid_loss = average_valid_loss
                save_checkpoint(file_path + '/' + 'model.pt', model, best_valid_loss)
                save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list,
                             global_steps_list)

    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')


# Evaluation Function

def evaluate(model, test_loader):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for (text, labels), _ in test_loader:
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            text = text.type(torch.LongTensor)
            text = text.to(device)
            output = model(text, labels)

            _, output = output
            y_pred.extend(torch.argmax(output, 1).tolist())
            y_true.extend(labels.tolist())

    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[3, 2, 1, 0], digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=[3, 2, 1, 0])
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['FAKE', 'REAL'])
    ax.yaxis.set_ticklabels(['FAKE', 'REAL'])


model = BERT().to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-5)

train(model=model, optimizer=optimizer)

best_model = BERT().to(device)

load_checkpoint(destination_folder + '/model.pt', best_model)

train_loss_list, valid_loss_list, global_steps_list = load_metrics(destination_folder + '/metrics.pt')
plt.plot(global_steps_list, train_loss_list, label='Train')
plt.plot(global_steps_list, valid_loss_list, label='Valid')
plt.xlabel('Global Steps')
plt.ylabel('Loss')
plt.legend()
plt.show()
