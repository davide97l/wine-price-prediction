import pandas as pd

if __name__ == '__main__':
    path_data = "training_set.csv"

    data = pd.read_csv(path_data, encoding='ISO-8859-1')
    features = data.head()

    print(data.describe())

    print("Number categories")
    for f in features:
        nulls = data[f].isna().sum() + len(data.loc[data[f] == "NA"])
        print(f + ": " + str(len(data[f].unique())) + " | null: " + str(nulls))


