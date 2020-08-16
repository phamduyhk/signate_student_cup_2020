import os
import random
import collections

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformers import AutoTokenizer, AutoModel, AdamW
import nlp

SEED = 42


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    current_device = torch.cuda.current_device()
    print("Device:", torch.cuda.get_device_name(current_device))

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRAIN_FILE = "./data/train.csv"
TEST_FILE = "./data/test.csv"
MODELS_DIR = "./models/"
MODEL_NAME = 'albert-xxlarge-v2'
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 128
NUM_CLASSES = 4
EPOCHS = 5
NUM_SPLITS = 10
MAX_LENGTH = 256
LEARNING_RATE = 2e-5

if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)


def preprocessing_text(df, is_train=True):
    remove_list = [';', '-', '+',  '1', '2','3', '4', '5', '6', '7','8', '9', '0', '&', '%', ':', '!', '/', '#','/', '#', ')', '(', '.', '"',  "'"]

    for i, line in enumerate(df['description']):
        for r in remove_list:
            df['description'][i] = df['description'][i].replace(r, "")
    # remove duplicated rows
    if is_train:
        df = df.drop_duplicates(subset=['description'])

    return df

def make_folded_df(csv_file, num_splits=5):
    df = pd.read_csv(csv_file)
    df = preprocessing_text(df)
    df["jobflag"] = df["jobflag"] - 1
    df["kfold"] = np.nan
    df = df.rename(columns={'jobflag': 'labels'})
    label = df["labels"].tolist()

    skfold = StratifiedKFold(num_splits, shuffle=True, random_state=SEED)
    for fold, (_, valid_indexes) in enumerate(skfold.split(range(len(label)), label)):
        for i in valid_indexes:
            df.iat[i, 3] = fold
    return df

def make_dataset(df, tokenizer, device):
    dataset = nlp.Dataset.from_pandas(df)
    dataset = dataset.map(
        lambda example: tokenizer(example["description"],
                                  padding="max_length",
                                  truncation=True,
                                  max_length=MAX_LENGTH))
    dataset.set_format(type='torch',
                       columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
                       device=device)
    return dataset


class Classifier(nn.Module):
    def __init__(self, model_name, num_classes=4):
        super().__init__()
        embedding_dim = 4096
        hidden_dim = 160
        n_layers = 2
        output_dim = 128
        drop_prob = 0.2
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(drop_prob)
        # for bert only
        self.linear = nn.Linear(embedding_dim, num_classes)

        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers,
                          batch_first=False, dropout=drop_prob)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            batch_first=False, dropout=drop_prob)

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        # for bert-gru-lstm
        # self.linear = nn.Linear(2*output_dim, num_classes)

        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output, _ = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)

        output = output[:, 0, :]

        # x = self.dropout(output)
        # gru, h_gru = self.gru(x)
        # gru = self.fc(self.relu(gru[:, -1]))
        # lstm, h_lstm = self.lstm(x)
        # lstm = self.fc(self.relu(lstm[:, -1]))
        # concatenate = torch.cat(
        #     (gru, lstm), 1)
        # output = self.linear(self.relu(concatenate))

        #   # for bert only
        output = self.dropout(output)
        output = self.linear(output)
        return output


def train_fn(dataloader, model, criterion, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    total_corrects = 0
    all_labels = []
    all_preds = []

    progress = tqdm(dataloader, total=len(dataloader))

    for i, batch in enumerate(progress):
        progress.set_description(f"<Train> Epoch{epoch + 1}")

        attention_mask, input_ids, labels, token_type_ids = batch.values()
        del batch

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask, token_type_ids)
        del input_ids, attention_mask, token_type_ids
        loss = criterion(outputs, labels)  # 損失を計算
        _, preds = torch.max(outputs, 1)  # ラベルを予測
        del outputs

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        del loss
        total_corrects += torch.sum(preds == labels)

        all_labels += labels.tolist()
        all_preds += preds.tolist()
        del labels, preds

        progress.set_postfix(loss=total_loss / (i + 1), f1=f1_score(all_labels, all_preds, average="macro"))

    train_loss = total_loss / len(dataloader)
    train_acc = total_corrects.double().cpu().detach().numpy() / len(dataloader.dataset)
    train_f1 = f1_score(all_labels, all_preds, average="macro")

    return train_loss, train_acc, train_f1


def eval_fn(dataloader, model, criterion, device, epoch):
    model.eval()
    total_loss = 0
    total_corrects = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        progress = tqdm(dataloader, total=len(dataloader))

        for i, batch in enumerate(progress):
            progress.set_description(f"<Valid> Epoch{epoch + 1}")

            attention_mask, input_ids, labels, token_type_ids = batch.values()
            del batch

            outputs = model(input_ids, attention_mask, token_type_ids)
            del input_ids, attention_mask, token_type_ids
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            del outputs

            total_loss += loss.item()
            del loss
            total_corrects += torch.sum(preds == labels)

            all_labels += labels.tolist()
            all_preds += preds.tolist()
            del labels, preds

            progress.set_postfix(loss=total_loss / (i + 1), f1=f1_score(all_labels, all_preds, average="macro"))

    valid_loss = total_loss / len(dataloader)
    valid_acc = total_corrects.double().cpu().detach().numpy() / len(dataloader.dataset)

    valid_f1 = f1_score(all_labels, all_preds, average="macro")

    return valid_loss, valid_acc, valid_f1


def plot_training(train_losses, train_accs, train_f1s,
                  valid_losses, valid_accs, valid_f1s,
                  epoch, fold):
    loss_df = pd.DataFrame({"Train": train_losses,
                            "Valid": valid_losses},
                           index=range(1, epoch + 2))
    loss_ax = sns.lineplot(data=loss_df).get_figure()
    loss_ax.savefig(f"./figures/loss_plot_fold={fold}.png", dpi=300)
    loss_ax.clf()

    acc_df = pd.DataFrame({"Train": train_accs,
                           "Valid": valid_accs},
                          index=range(1, epoch + 2))
    acc_ax = sns.lineplot(data=acc_df).get_figure()
    acc_ax.savefig(f"./figures/acc_plot_fold={fold}.png", dpi=300)
    acc_ax.clf()

    f1_df = pd.DataFrame({"Train": train_f1s,
                          "Valid": valid_f1s},
                         index=range(1, epoch + 2))
    f1_ax = sns.lineplot(data=f1_df).get_figure()
    f1_ax.savefig(f"./figures/f1_plot_fold={fold}.png", dpi=300)
    f1_ax.clf()


def trainer(fold, df):
    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    train_dataset = make_dataset(train_df, tokenizer, DEVICE)
    valid_dataset = make_dataset(valid_df, tokenizer, DEVICE)


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False
    )

    model = Classifier(MODEL_NAME, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=1.0)
    # ダミーのスケジューラー

    train_losses = []
    train_accs = []
    train_f1s = []
    valid_losses = []
    valid_accs = []
    valid_f1s = []

    best_loss = np.inf
    best_acc = 0
    best_f1 = 0

    for epoch in range(EPOCHS):
        train_loss, train_acc, train_f1 = train_fn(train_dataloader, model, criterion, optimizer, scheduler, DEVICE,
                                                   epoch)
        valid_loss, valid_acc, valid_f1 = eval_fn(valid_dataloader, model, criterion, DEVICE, epoch)
        print(f"Loss: {valid_loss}  Acc: {valid_acc}  f1: {valid_f1}  ", end="")

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_f1s.append(train_f1)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        valid_f1s.append(valid_f1)

        # plot_training(train_losses, train_accs, train_f1s,
        #               valid_losses, valid_accs, valid_f1s,
        #               epoch, fold)

        best_loss = valid_loss if valid_loss < best_loss else best_loss
        besl_acc = valid_acc if valid_acc > best_acc else best_acc
        if valid_f1 > best_f1:
            best_f1 = valid_f1
            print("model saving!", end="")
            torch.save(model.state_dict(), MODELS_DIR + f"best_{MODEL_NAME}_{fold}.pth")
        print("\n")

    return best_f1


# TRAINING
df = make_folded_df(TRAIN_FILE, NUM_SPLITS)
f1_scores = []
for fold in range(NUM_SPLITS):
    print(f"fold {fold}", "=" * 80)
    f1 = trainer(fold, df)
    f1_scores.append(f1)
    print(f"<fold={fold}> best score: {f1}\n")

cv = sum(f1_scores) / len(f1_scores)
print(f"CV: {cv}")

lines = ""
for i, f1 in enumerate(f1_scores):
    line = f"fold={i}: {f1}\n"
    lines += line
lines += f"CV    : {cv}"
if not os.path.exists("./result"):
    os.mkdir("./result")
with open(f"./result/{MODEL_NAME}_result.txt", mode='w') as f:
    f.write(lines)

models = []
for fold in range(NUM_SPLITS):
    model = Classifier(MODEL_NAME)
    model.load_state_dict(torch.load(MODELS_DIR + f"best_{MODEL_NAME}_{fold}.pth"))
    model.to(DEVICE)
    model.eval()
    models.append(model)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
test_df = pd.read_csv(TEST_FILE)
test_df = preprocessing_text(test_df, is_train=False)
test_df["labels"] = -1
test_dataset = make_dataset(test_df, tokenizer, DEVICE)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)

final_prob = []
final_output = []
with torch.no_grad():
    progress = tqdm(test_dataloader, total=len(test_dataloader))
 
    for batch in progress:
        progress.set_description("<Test>")

        attention_mask, input_ids, labels, token_type_ids = batch.values()

        outputs = []
        for model in models:
            output = model(input_ids, attention_mask, token_type_ids)
            outputs.append(output)

        outputs = sum(outputs) / len(outputs)
        outputs = torch.softmax(outputs, dim=1).cpu().detach().tolist()
        final_prob.extend(outputs)
        outputs = np.argmax(outputs, axis=1)
        
        final_output.extend(outputs)

submit = pd.read_csv("./data/submit_sample.csv", names=["id", "labels"])
submit["labels"] = final_output
submit["labels"] = submit["labels"] + 1
submit["probs"] = final_prob
if not os.path.exists("./output"):
    os.mkdir("./output")
try:
    submit.to_csv("./output/prob_{}_{}_{}cv_{}ep.csv".format(str(MODEL_NAME),str(MAX_LENGTH),str(NUM_SPLITS),str(EPOCHS)), index=False, header=False)
except NameError:
    submit.to_csv("./output/submission.csv", index=False, header=False)
submit.head()
