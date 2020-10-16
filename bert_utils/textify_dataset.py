import pandas as pd
import argparse


def textify(args):
    make_discrete = args.make_discrete
    path_data = args.path_data
    n_labels = args.num_labels
    label_range = args.label_range
    labels = [i for i in range(n_labels)]
    values = [i * label_range + label_range for i in range(n_labels)]

    text_features = ["country", "province", "winery", "region_1", "designation", "variety", "review_score"]
    y_label = "price"
    df = pd.read_csv(path_data, encoding='ISO-8859-1')

    test_set = False
    if y_label not in df.head():
        test_set = True

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
        if make_discrete:
            df[y_label] = df[y_label].apply(discretize)
        df = df[["content", "review", y_label]]
        df1 = pd.Series(df["price"].values).value_counts().reset_index().sort_values('index').reset_index(drop=True)
        df1.columns = ['price_label', 'frequency']
        print(df1)
    else:
        df = df[["content", "review"]]

    output_path = path_data.split(".csv")[0] + "_text"
    if make_discrete:
        output_path += "_discrete_" + str(n_labels)
    output_path += ".csv"
    df.to_csv(output_path, index=False)
    print("created file:", output_path)
    return df


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_data', type=str, default='training_set.csv')
    parser.add_argument('--num_labels', type=int, default=20)
    parser.add_argument('--label_range', type=int, default=5)
    parser.add_argument('--make_discrete', default=False, action='store_true')
    args = parser.parse_known_args()[0]
    return args


# python bert_utils/textify_dataset.py --path_data "training_set.csv" --make_discrete
if __name__ == '__main__':
    args = get_args()
    textify(args)

