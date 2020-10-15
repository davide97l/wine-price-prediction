import pandas as pd

if __name__ == '__main__':
    path_data = "training_set.csv"

    data = pd.read_csv(path_data, encoding='ISO-8859-1')
    features = data.head()

    print(data.describe())

    print("---number categories--")
    for f in features:
        nulls = data[f].isna().sum() + len(data.loc[data[f] == "NA"])
        print(f + ": " + str(len(data[f].unique())) + " | null: " + str(nulls))

    print("---features average value---")
    col = "country"
    dict_avg = {}
    values = pd.unique(data[col])
    for i, v in enumerate(values):
        df1 = data.loc[data[col] == v]
        mean = df1["price"].mean()
        dict_avg[v] = mean
    dict_avg = {k: v for k, v in sorted(dict_avg.items(), key=lambda item: item[1])}
    for k, v in dict_avg.items():
        print(k, v)

