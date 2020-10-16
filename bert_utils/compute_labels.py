import pandas as pd

if __name__ == '__main__':
    path_data = "bert_classification_20.csv"
    label_range = 5

    df = pd.read_csv(path_data, encoding='ISO-8859-1', header=None)

    def regress(x):
        x = x.values[0]
        return int(x * label_range - label_range / 2 + label_range)

    df = df.apply(regress, axis=1)

    df.to_csv("bert_regression_20.csv", index=False, header=False)
    print("done!")
