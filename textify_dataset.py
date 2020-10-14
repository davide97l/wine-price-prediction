import pandas as pd

if __name__ == '__main__':
    test_set = True
    path_data = "test_set.csv"
    n_labels = 20
    label_range = 5
    labels = [i for i in range(n_labels)]
    values = [i * label_range + label_range for i in range(n_labels)]

    text_features = ["country", "province", "winery", "region_1", "designation", "variety", "review_score"]
    y_label = "price"
    df = pd.read_csv(path_data, encoding='ISO-8859-1')

    def make_text(x):
        text = ""
        for f in text_features:
            text += str(x[f]) + " "
        text = text.strip()
        return text

    def discretize(x):
        for i, l in enumerate(labels[:-1]):
            if x < values[i]:
                return i
        return n_labels - 1

    df["content"] = df.apply(make_text, axis=1)
    if not test_set:
        df[y_label] = df[y_label].apply(discretize)
        df = df[["content", "review", y_label]]
    else:
        df = df[["content", "review"]]

    if not test_set:
        df1 = pd.Series(df["price"].values).value_counts().reset_index().sort_values('index').reset_index(drop=True)
        df1.columns = ['price_label', 'frequency']
        print(df1)

    if not test_set:
        df.to_csv("training_set_discrete_labels.csv", index=False)
    else:
        df.to_csv("test_set_discrete_labels.csv", index=False)
    print("done!")
