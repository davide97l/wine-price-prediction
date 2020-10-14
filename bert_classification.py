import pandas as pd
import torch.utils.data as Data
import torch
from pytorch_transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset
import argparse
import os
import copy
import numpy as np
from tqdm import tqdm


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
        y = torch.tensor(y)
        return x, y


def train_one_epoch(model, optimizer, dataset, batch_size=32, device="cpu"):
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model.train()
    train_loss, train_acc = 0.0, 0.0
    for batch, labels in tqdm(generator):
        batch, labels = batch.to(device), labels.to(device)
        optimizer.zero_grad()
        loss, logits = model(batch, labels=labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred_labels = torch.argmax(logits, dim=1)
        train_acc += (pred_labels == labels).sum().item()
    train_loss /= len(dataset)
    train_acc /= len(dataset)
    return train_loss, train_acc


def evaluate_one_epoch(model, f_loss, dataset, batch_size=32, device="cpu"):
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model.eval()
    loss, acc = 0.0, 0.0
    with torch.no_grad():
        for batch, labels in tqdm(generator):
            batch, labels = batch.to(device), labels.to(device)
            logits = model(batch)[0]
            err = f_loss(logits, labels)
            loss += err.item()
            pred_labels = torch.argmax(logits, dim=1)
            acc += (pred_labels == labels).sum().item()
    loss /= len(dataset)
    acc /= len(dataset)
    return loss, acc


def predict_one_epoch(model, dataset, batch_size=32, device="cpu"):
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch, labels in tqdm(generator):
            batch = batch.to(device)
            logits = model(batch)[0]
            pred_labels = torch.argmax(logits, dim=1)
            pred_labels = pred_labels.cpu().detach()
            predictions += pred_labels
    return predictions


def bert_classification(args):

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
    num_labels = args.num_labels
    text_column = args.text_column

    df = pd.read_csv(path_data, encoding='ISO-8859-1')
    train_y = df["price"].values.astype(int)
    train_x = df[text_column].values
    if limit_rows is not None:
        train_x = train_x[:limit_rows]
        train_y = train_y[:limit_rows]

    if val and os.path.exists(path_val):
        df = pd.read_csv(path_val, encoding='ISO-8859-1')
        val_y = df["price"].values.astype(int)
        val_x = df[text_column].values
    else:
        val = False

    if test and os.path.exists(path_test):
        df = pd.read_csv(path_test, encoding='ISO-8859-1')
        test_x = df[text_column].values
    else:
        test = False

    train_set = CustomDataset(train_x, train_y, max_length, bert)
    if val:
        dev_set = CustomDataset(val_x, val_y, max_length, bert)
    if test:
        test_set = CustomDataset(test_x, np.zeros([test_x.shape[0]]), max_length, bert)

    model = BertForSequenceClassification.from_pretrained(bert, num_labels=num_labels).to(device)
    if load_model and os.path.exists("classification_bert.pth"):
        model.load_state_dict(torch.load("classification_bert.pth"))
        print("Model loaded")
    else:
        load_model = False

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    f_loss = torch.nn.CrossEntropyLoss()

    if val:
        best_model = copy.deepcopy(model.state_dict())
        best_val_loss = np.inf

    if not load_model:
        print("Start training")
        for epoch in range(0, epochs):

            train_loss, train_acc = train_one_epoch(model, optimizer, train_set, batch_size, device)
            print("Epoch {}, Train loss: {}, Train Acc: {}".format(epoch+1, train_loss, train_acc))

            if val:
                val_loss, val_acc = evaluate_one_epoch(model, f_loss, dev_set, batch_size, device)
                print("Epoch {}, Val loss: {}, Val Acc: {}".format(epoch+1, val_loss, val_acc))

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
        pd.DataFrame(predictions).to_csv("bert_classification_" + str(num_labels) + ".csv", index=False, header=False)
        print("Predictions saved")
    if save_model:
        torch.save(model.state_dict(), "classification_bert_" + str(num_labels) + ".pth")
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
    parser.add_argument('--num_labels', type=int, default=5)
    parser.add_argument('--text_column', type=str, default="content")
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--test', default=True, action='store_false')
    parser.add_argument('--val', default=True, action='store_false')
    parser.add_argument('--save_model', default=True, action='store_false')
    parser.add_argument('--load_model', default=True, action='store_false')
    args = parser.parse_known_args()[0]
    return args


# python bert_classification.py
# python bert_classification.py --epochs 1000 --path_data training_set_discrete_labels.csv
# python bert_classification.py --epochs 1 --path_data training_set_discrete_labels.csv --path_test test_set_discrete_labels.csv --num_labels 20 --batch_size 64 --text_column content --max_length 13
# python bert_classification.py --epochs 1 --path_data training_set_discrete_labels.csv --path_test test_set_discrete_labels.csv --num_labels 20 --batch_size 64 --text_column content --max_length 13 --limit_rows 1000
if __name__ == '__main__':
    args = get_args()
    bert_classification(args)
