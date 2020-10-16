import pandas as pd
import argparse
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV


def order_cat_columns(df, col):
    values = pd.unique(df[col])
    dict_avg = {}
    for i, v in enumerate(values):
        df1 = df.loc[df[col] == v]
        mean = df1["price"].mean()
        dict_avg[v] = mean
    dict_avg = {k: v for k, v in sorted(dict_avg.items(), key=lambda item: item[1])}
    ordered_values = dict_avg.keys()
    dict_values = {}
    for i, v in enumerate(ordered_values):
        dict_values[v] = i
    return dict_values


def preprocess_cat_columns(df, columns, order_columns):
    if not hasattr(preprocess_cat_columns, "dict_values"):
        preprocess_cat_columns.dict_values = {}

    def get_value(x):
        if x in preprocess_cat_columns.dict_values[col].keys():
            v = preprocess_cat_columns.dict_values[col][x]
        else:
            v = len(preprocess_cat_columns.dict_values[col])
            preprocess_cat_columns.dict_values[col][x] = v
        return v

    for col in columns:
        if len(pd.unique(df[col])) <= 5:
            one_hot = pd.get_dummies(df[col])
            df = df.join(one_hot)
            df = df.drop([col], axis=1)
        else:
            if col not in order_columns:
                le = preprocessing.LabelEncoder()
                le.fit(df[col])
                df[col] = le.transform(df[col])
            else:
                if order_columns is not None and col not in preprocess_cat_columns.dict_values.keys():
                    preprocess_cat_columns.dict_values[col] = order_cat_columns(df, col)
                df[col] = df[col].apply(get_value)
    return df


def normalize(df, columns):
    x = df[columns].values
    x_scaled = preprocessing.normalize(x)
    df_temp = pd.DataFrame(x_scaled, columns=columns, index=df.index)
    df[columns] = df_temp
    return df


def remove_outliers(train_x, train_y):
    iso = IsolationForest(contamination='auto')
    yhat = iso.fit_predict(train_x)
    print("Removed " + str(yhat.sum()) + " outliers")
    mask = yhat != -1
    train_x, train_y = train_x[mask, :], train_y[mask]
    return train_x, train_y


def process_data(df, drop_columns=None, categorical_columns=None, normalize_columns=None, order_columns=None):
    if drop_columns:
        df = df.drop(drop_columns, axis=1)
    if categorical_columns:
        df = preprocess_cat_columns(df, categorical_columns, order_columns)
    if normalize_columns:
        df = normalize(df, normalize_columns)
    print(df)
    return df


def search_best_model(model, train_x, train_y):

    parameters_for_testing = {
        'n_estimators': [100, 300],
        'max_features': ['auto', 'sqrt', 'log2'],
        'min_samples_split': [2, 4, 6],
        'min_impurity_decrease': [0., 0.2],
        'max_depth': [None, 3],
    }

    # 'max_depth': None, 'max_features': 'sqrt', 'min_impurity_decrease': 0.0, 'min_samples_split': 4, 'n_estimators': 300

    model = GridSearchCV(estimator=model, param_grid=parameters_for_testing,
                         n_jobs=32, verbose=10, scoring='neg_mean_squared_error')
    model.fit(train_x, train_y)
    print('best params')
    print(model.best_params_)
    print('best score')
    print(model.best_score_)

    return model


def random_forest(args):
    cat_columns = ["country", "province", "winery", "region_1", "variety", "designation"]
    drop_columns = ["region_2", "review"]
    normalize_columns = None  # ["country", "province", "winery", "region_1", "variety", "designation", "review_score"]
    order_columns = None  # ["country", "province", "winery", "region_1", "variety", "designation"]
    label = "price"

    train_dataset = pd.read_csv(args.path_data, encoding='ISO-8859-1')
    test_dataset = pd.read_csv(args.path_test, encoding='ISO-8859-1')
    train_dataset = process_data(train_dataset, drop_columns, cat_columns, normalize_columns, order_columns)
    test_dataset = process_data(test_dataset, drop_columns, cat_columns, normalize_columns, order_columns)
    train_x = train_dataset.drop([label], axis=1).values
    train_y = train_dataset[label].values
    if args.remove_outliers:
        train_x, train_y = remove_outliers(train_x, train_y)

    rf_regr = RandomForestRegressor(max_depth=None, n_estimators=300, max_features='sqrt', min_impurity_decrease=0.,
                                    min_samples_split=4)
    if not args.grid_search:
        rf_regr = search_best_model(rf_regr, train_x, train_y)
    else:
        rf_regr.fit(train_x, train_y)
    rf_preds = rf_regr.predict(test_dataset.values)
    rf_preds = rf_preds.astype(int)
    pd.DataFrame(rf_preds).to_csv(args.path_save, index=False, header=False)

    print("done!")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_data', type=str, default='training_set.csv')
    parser.add_argument('--path_test', type=str, default='test_set.csv')
    parser.add_argument('--grid_search', default=False, action='store_false')
    parser.add_argument('--remove_outliers', default=False, action='store_true')
    parser.add_argument('--path_save', type=str, default="rf_predictions.csv")
    args = parser.parse_known_args()[0]
    return args


# python random_forest.py --grid_search
if __name__ == '__main__':
    args = get_args()
    random_forest(args)
