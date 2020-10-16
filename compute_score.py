import pandas as pd
import argparse
from math import sqrt
from sklearn.metrics import mean_squared_error


def rmse(preds, labels):
    return sqrt(mean_squared_error(preds, labels))


def compute_score(args):
    path_data = args.path_data
    path_validation = args.path_val

    df1 = pd.read_csv(path_data, encoding='ISO-8859-1')["price"].values
    df2 = pd.read_csv(path_validation, encoding='ISO-8859-1', header=None).values
    score = rmse(df1, df2)
    print("rmse:", score)
    print("done!")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_data', type=str)
    parser.add_argument('--path_val', type=str)
    args = parser.parse_known_args()[0]
    return args


# python compute_score.py --path_data "training_set.csv" --path_val "ensemble_predictions.csv"
if __name__ == '__main__':
    args = get_args()
    compute_score(args)
