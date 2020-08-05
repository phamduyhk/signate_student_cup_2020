# coding: utf-8
"""
Author: Pham Duy
Created date: 2020/08/05
"""
# coding: utf-8
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

import torchtext
import pandas as pd
import datetime
import os
import sys

from utils.EarlyStopping import EarlyStopping
from utils.transformer import TransformerClassification
from utils.dataloader import Preprocessing

preprocessing = Preprocessing()
es = EarlyStopping(patience=10)
sigmoid = nn.Sigmoid()


def train(data_path, train_file, test_file, vector_list,
          max_sequence_length=512, num_epochs=15, learning_rate=3e-5,
          device=None, train_mode=True,
          load_trained=False,
          early_stop=False):
    if device is None:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    train_dl, val_dl, test_dl, TEXT = preprocessing.get_data(path=data_path, train_file=train_file, test_file=test_file,
                                                             vectors=vector_list, max_length=max_sequence_length,
                                                             batch_size=1024)

    dataloaders_dict = {"train": train_dl, "val": val_dl}

    # define output dataframe
    sample = pd.read_csv("./data/submit_sample.csv")

    label_cols = ['jobflag']

    num_labels = 4

    if load_trained is True:
        net = torch.load("net_trained_transformer.weights",
                         map_location=device)
    else:
        net = TransformerClassification(
            text_embedding_vectors=TEXT.vocab.vectors, d_model=300, max_seq_len=max_sequence_length,
            output_dim=num_labels,
            device=device)

    net.train()

    net.net3_1.apply(weights_init)
    net.net3_2.apply(weights_init)

    print('done setup network')

    print("running mode: {}".format("training" if train_mode else "predict"))

    # Define loss function
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCELoss()

    """or"""
    # criterion = nn.MultiLabelSoftMarginLoss()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    if not os.path.exists("./transformers"):
        os.mkdir("./transformers")

    net_trained_save_path = os.path.join("./transformers", "net_trained_transformer.weights")

    if train_mode:
        net_trained = train_model(net, dataloaders_dict,
                                  criterion, optimizer, num_epochs=num_epochs, label_cols=label_cols, device=device,
                                  early_stop=early_stop)

        # net_trainedを保存
        torch.save(net_trained, net_trained_save_path)

    else:
        net_trained = net

    net_trained.eval()
    net_trained.to(device)

    pred_probs = np.array([]).reshape(0, num_labels)
    raw_pred = np.array([]).reshape(0, num_labels)

    for batch in (test_dl):
        inputs = batch.Text[0].to(device)

        with torch.set_grad_enabled(False):
            input_pad = 1
            input_mask = (inputs != input_pad)

            outputs, _, _ = net_trained(inputs, input_mask)
            raw_output = outputs.cpu()
            raw_pred = np.vstack([raw_pred, raw_output])
            preds = (outputs.sigmoid() > 0.5) * 1
            preds = preds.cpu()
            pred_probs = np.vstack([pred_probs, preds])
    print(raw_pred)
    df = pd.DataFrame()
    raw_pred = raw_pred.reshape(raw_pred.shape[1], raw_pred.shape[0])
    for index, label in enumerate(label_cols):
        df[label] = raw_pred[index]
    df.to_csv("transformer_raw_pred.csv", index=False)
    print(pred_probs)
    # predicts = np.round(pred_probs)
    predicts = pred_probs.reshape(pred_probs.shape[1], pred_probs.shape[0])
    for index, label in enumerate(label_cols):
        sample[label] = predicts[index]

    # save predictions
    if not os.path.exists("./submission"):
        os.mkdir("./submission")
    sample.to_csv("./submission/submission_Transformer_{}_{}ep.csv".format(
        datetime.datetime.now().date(), num_epochs), index=False)


def roc_auc_score_FIXED(y_true, y_pred):
    try:
        score = roc_auc_score(y_true, y_pred)
    except ValueError:
        score = accuracy_score(y_true, np.rint(y_pred))
    return score


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs, label_cols, device="cpu", early_stop=False):
    print("using device: ", device)

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     net = nn.DataParallel(net)

    net.to(device)

    torch.backends.cudnn.benchmark = True

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_metrics = 0

            for batch in (dataloaders_dict[phase]):
                inputs = batch.Text[0].to(device)
                y_true = torch.cat([getattr(batch, feat).unsqueeze(1)
                                    for feat in label_cols], dim=1).float()
                y_true = y_true.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    input_pad = 1
                    input_mask = (inputs != input_pad)

                    outputs, _, _ = net(inputs, input_mask)
                    loss = criterion(outputs, y_true)
                    preds = torch.argmax(outputs, 1)

                    # training mode
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # validation mode
                    epoch_loss += loss.item() * inputs.size(0)
                    y_true = y_true.data.cpu()
                    preds = preds.cpu()
                    print("y_true {}, y_pred {}".format(y_true, preds))
                    epoch_metrics += f1_score(y_true, preds, average='macro')

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_eval = epoch_metrics / len(dataloaders_dict[phase])

            print('Epoch {}/{} | {:^5} |  Loss: {:.4f} F1 Score: {:.4f}'.format(epoch + 1, num_epochs,
                                                                                phase, epoch_loss, epoch_eval))

        if early_stop:
            if es.step(torch.tensor(epoch_eval)):
                print("Early stoped at epoch: {}".format(num_epochs))
                break  # early stop criterion is met, we can stop now

    return net


if __name__ == '__main__':
    path = "./data/"
    train_file = "train.csv"
    test_file = "test.csv"
    vector_list_file = "wiki-news-300d-1M.vec"
    train(data_path=path, train_file=train_file, test_file=test_file, vector_list=vector_list_file)
