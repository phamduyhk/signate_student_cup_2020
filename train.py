# Libraries
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
# preprocessing module
from utils.dataloader import Preprocessing

# Preliminaries

from torchtext.data import Field, TabularDataset, BucketIterator, Iterator

# Models

import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
import torch

# Training

import torch.optim as optim

# Evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import seaborn as sns

import preprocessing

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Model parameter
MAX_SEQ_LEN = 256
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

# Fields

label_field = Field(sequential=False, use_vocab=False,
                    batch_first=True, dtype=torch.float)
text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                   fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
                   
train_fields = [  ('id', None), ('description', text_field), ('label', label_field)]
test_fields = [  ('id', None), ('description', text_field)]

# SETTINGs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
destination_folder = "./output"
if not os.path.exists(destination_folder):
    os.mkdir(destination_folder)
source_folder = "./data"

# Get data (ver transformer)
vector_list_file = "wiki-news-300d-1M.vec"
max_sequence_length = 256
batch_size = 128

# train, valid, test, description = preprocessing.read_files()
# TabularDataset Ver
train_val_ds = TabularDataset(path='data/train_data.csv',format='CSV', fields=train_fields, skip_header=True)
test = TabularDataset(path='data/test.csv', format='CSV', fields=test_fields, skip_header=True)

train, valid = train_val_ds.split(
    split_ratio=0.8, random_state=random.seed(2395))
train_iter = BucketIterator(train, batch_size=batch_size, sort_key=lambda x: len(x.description),
                            device=device, train=True, sort=True, sort_within_batch=True, shuffle=False)
valid_iter = BucketIterator(valid, batch_size=batch_size, sort_key=lambda x: len(x.description),
                            device=device, train=True, sort=True, sort_within_batch=True, shuffle=False)
test_iter = Iterator(test, batch_size=batch_size,
                     device=device, train=False, shuffle=False, sort=False)


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers,
                          batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size,
                            self.hidden_dim).zero_().to(device)
        return hidden


class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers,
                            batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.lstm(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden


class BERT(nn.Module):

    def __init__(self, num_labels=4, embedding_dim=768, hidden_dim=128, output_dim=160, n_layers=2, drop_prob=0.2):
        super(BERT, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        options_name = "bert-base-uncased"
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.encoder = BertForSequenceClassification.from_pretrained(
            options_name, num_labels=4)
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers,
                          batch_first=True, dropout=drop_prob)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            batch_first=True, dropout=drop_prob)

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(2*output_dim, num_labels)

        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)


        # freeze_bert
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, text, label):
        embedding = self.model(text)
        x = embedding[0]
        # x = x.permute(0, 2, 1)   # convert to [batch, channels, time]
        # x = F.dropout2d(x, p, training=self.training)
        # x = x.permute(0, 2, 1)   # back to [batch, time, channels]
        gru, h_gru = self.gru(x)
        gru = self.fc(self.relu(gru[:, -1]))
        lstm, h_lstm = self.lstm(x)
        lstm = self.fc(self.relu(lstm[:, -1]))
        concatenate = torch.cat(
            (gru, lstm), 1)
        out = self.linear(self.relu(concatenate))
        return out


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
          loss_function=nn.CrossEntropyLoss(),
          train_loader=train_iter,
          valid_loader=valid_iter,
          num_epochs=100,
          eval_every=len(train_iter) // 2,
          file_path=destination_folder,
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
        for (text, labels), _ in train_loader:
            model.zero_grad()
            labels = labels.type(torch.LongTensor)
            text = text.type(torch.LongTensor)
            output = model(text, labels)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            label_cpu = labels.cpu()
            pred_cpu = torch.argmax(output, 1).cpu()
            running_f1_score += f1_score(label_cpu, pred_cpu, average='macro')
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():

                    # validation loop
                    for (text, labels), _ in valid_loader:
                        labels = labels.type(torch.LongTensor)
                        text = text.type(torch.LongTensor)
                        preds = model(text, labels)
                        loss = loss_function(preds, labels)

                        valid_running_loss += loss.item()
                        labels_cpu = labels.cpu()
                        preds_cpu = torch.argmax(preds, 1).cpu()
                        valid_running_f1_score += f1_score(
                            labels_cpu, preds_cpu, average='macro')

                # evaluation
                average_train_loss = running_loss / eval_every
                average_f1_score = running_f1_score / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                average_valid_f1_score = valid_running_f1_score / \
                    len(valid_loader)
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
                    'Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}, Train F1 Score: {:.4f}, Valid F1 Score: {:.4f}'
                    .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_loader),
                            average_train_loss, average_valid_loss, average_f1_score, average_valid_f1_score))

                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/' + 'model.pt',
                                    model, best_valid_loss)
                    save_metrics(file_path + '/' + 'metrics.pt',
                                 train_loss_list, valid_loss_list, global_steps_list)

    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list,
                 valid_loss_list, global_steps_list)
    print('Finished Training!')


def generate_predict(model, test_loader):
    y_pred = []
    model.eval()
    with torch.no_grad():
        for text, _ in test_loader:
            text = text.type(torch.LongTensor)
            output = model(text)
            y_pred.extend(torch.argmax(output, 1).tolist())
            y_true.extend(labels.tolist())
    return y_pred

# TRAINING
model = BERT()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

train(model=model, optimizer=optimizer)
predict = generate_predict(model, test_iter)
submission = pd.read_csv("data/submit_sample.csv", header=None)
submission.iloc[:, 1] = predict + 1
submission.to_csv('submission.csv', index=False, header=None)

# best_model = BERT()

# load_checkpoint(destination_folder + '/model.pt', best_model)

# train_loss_list, valid_loss_list, global_steps_list = load_metrics(
#     destination_folder + '/metrics.pt')
# plt.plot(global_steps_list, train_loss_list, label='Train')
# plt.plot(global_steps_list, valid_loss_list, label='Valid')
# plt.xlabel('Global Steps')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
