import pandas as pd
import torch.utils.data as Data
import torch
from pytorch_transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset
import argparse
import os
import copy
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from tqdm import tqdm


def rmse(preds, labels):
    return sqrt(mean_squared_error(preds, labels))


def right_pad(array, n=500):
    """Right padding."""
    current_len = len(array)
    if current_len > n:
        array[n - 1] = 102
        return array[: n]
    extra = n - current_len
    return array + ([0] * extra)


class CustomDataset(Dataset):
    """Configurable SST Dataset."""

    def __init__(self, train_x, train_y, max_length=500, bert="bert-base-uncased"):
        """Initializes the dataset with given configuration."""
        tokenizer = BertTokenizer.from_pretrained(bert)
        self.max_length = max_length

        sentences = train_x
        labels = train_y

        self.data = [
            (
                right_pad(
                    tokenizer.encode("[CLS] " + sentence + " [SEP]"), self.max_length
                ),
                label,
            )
            for sentence, label in zip(sentences, labels)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index]
        x = torch.tensor(x)
        y = torch.tensor(y, dtype=torch.float)
        return x, y


class RegressionBert(torch.nn.Module):
    def __init__(self, bert, freeze_bert=False):
        super(RegressionBert, self).__init__()
        self.bert = BertModel.from_pretrained(bert)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.fc0 = torch.nn.Linear(768, 1)

    def forward(self, x, att=None):
        x = self.bert(x, attention_mask=att)[0]
        x = x[:, 0, :]
        x = self.fc0(x)
        x = x.flatten()
        return x


def train_one_epoch(model, optimizer, dataset, f_loss, batch_size=32, device="cpu"):
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model.train()
    train_loss, train_acc = 0.0, 0.0
    for batch, labels in tqdm(generator):
        batch, labels = batch.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(batch)
        loss = f_loss(preds, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += rmse(preds.cpu().detach(), labels.cpu().detach())
    train_loss /= len(dataset)
    train_acc /= len(dataset)
    return train_loss, train_acc


def evaluate_one_epoch(model, dataset, f_loss, batch_size=32, device="cpu"):
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model.eval()
    val_loss, val_acc = 0.0, 0.0
    with torch.no_grad():
        for batch, labels in tqdm(generator):
            batch, labels = batch.to(device), labels.to(device)
            preds = model(batch)
            loss = f_loss(preds, labels)
            loss.backward()
            val_loss += loss.item()
            preds = preds.cpu().detach()
            val_acc += rmse(preds, labels.cpu().detach())
    val_loss /= len(dataset)
    val_acc /= len(dataset)
    return val_loss, val_acc


def predict_one_epoch(model, dataset, batch_size=32, device="cpu"):
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch, labels in tqdm(generator):
            batch = batch.to(device)
            preds = model(batch)
            preds = preds.cpu().detach()
            predictions += preds
    return predictions


def bert_regression(args):

    path_data = args.path_data
    path_val = args.path_val
    path_test = args.path_test
    epochs = args.epochs
    bert = args.bert
    batch_size = args.batch_size
    max_length = args.max_length
    device = args.device
    test = args.test
    val = args.val
    limit_rows = args.limit_rows
    save_model = args.save_model
    load_model = args.load_model
    freeze_bert = args.freeze_bert

    df = pd.read_csv(path_data, encoding='ISO-8859-1', )
    train_y = df["price"].values.astype(float)
    train_x = df["review"].values
    if limit_rows is not None:
        train_x = train_x[:limit_rows]
        train_y = train_y[:limit_rows]

    if val and os.path.exists(path_val):
        df = pd.read_csv(path_val, encoding='ISO-8859-1')
        val_y = df["price"].values.astype(float)
        val_x = df["review"].values
    else:
        val = False

    if test and os.path.exists(path_test):
        df = pd.read_csv(path_test, encoding='ISO-8859-1')
        test_x = df["review"].values
    else:
        test = False

    train_set = CustomDataset(train_x, train_y, max_length, bert)
    if val:
        dev_set = CustomDataset(val_x, val_y, max_length, bert)
    if test:
        test_set = CustomDataset(test_x, np.zeros([test_x.shape[0]]), max_length, bert)

    model = RegressionBert(bert, freeze_bert).to(device)
    if load_model and os.path.exists("regression_bert.pth"):
        model.load_state_dict(torch.load("regression_bert.pth"))
        print("Model loaded")
    else:
        load_model = False

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    f_loss = torch.nn.MSELoss()

    if val:
        best_model = copy.deepcopy(model.state_dict())
        best_val_loss = np.inf

    if not load_model:
        print("Start training...")
        for epoch in range(0, epochs):

            train_loss, train_acc = train_one_epoch(model, optimizer, train_set, f_loss, batch_size, device)
            print("Epoch {}, Train loss: {}, Train RMSE: {}".format(epoch+1, train_loss, train_acc))

            if val:
                val_loss, val_acc = evaluate_one_epoch(model, dev_set, f_loss, batch_size, device)
                print("Epoch {}, Val loss: {}, Val RMSE: {}".format(epoch+1, val_loss, val_acc))

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = copy.deepcopy(model.state_dict())
        print("Finish training")

    if val:
        model.load_state_dict(best_model)
        print("Loaded best model")
    if test:
        print("Making predictions...")
        predictions = predict_one_epoch(model, test_set, batch_size, device)
        predictions = predictions[:test_x.shape[0]]
        predictions = np.array(predictions).astype(int)
        pd.DataFrame(predictions).to_csv("bert_regression.csv", index=False, header=False)
        print("Predictions saved")
    if save_model:
        torch.save(model.state_dict(), "regression_bert.pth")
        print("Model saved")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_data', type=str, default='training_set.csv')
    parser.add_argument('--path_val', type=str, default='validation_set.csv')
    parser.add_argument('--path_test', type=str, default='test_set.csv')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--bert', type=str, default="bert-base-uncased")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--limit_rows', type=int, default=None)
    parser.add_argument('--max_length', type=int, default=300)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--test', default=True, action='store_false')
    parser.add_argument('--val', default=True, action='store_false')
    parser.add_argument('--save_model', default=True, action='store_false')
    parser.add_argument('--load_model', default=True, action='store_false')
    parser.add_argument('--freeze_bert', default=False, action='store_true')
    args = parser.parse_known_args()[0]
    return args


# python bert_regression.py
# python bert_regression.py --epochs 1000 --freeze_bert
if __name__ == '__main__':
    args = get_args()
    bert_regression(args)
