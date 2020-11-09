import pandas as pd
import argparse
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import sys
import numpy
from math import sqrt
from sklearn.metrics import mean_squared_error
import numpy as np
numpy.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import os
from sklearn.metrics import mean_absolute_error


def rmse(preds, labels):
    return sqrt(mean_squared_error(preds, labels))


def preprocess_cat_columns(df, columns, one_hot_threshold=10):

    for col in columns:
        if len(pd.unique(df[col])) <= one_hot_threshold:
            one_hot = pd.get_dummies(df[col])
            df = df.join(one_hot)
            df = df.drop([col], axis=1)
        else:
            le = preprocessing.LabelEncoder()
            le.fit(df[col])
            df[col] = le.transform(df[col])
    return df


def remove_outliers(train_x, train_y):
    iso = IsolationForest(contamination='auto')
    yhat = iso.fit_predict(train_x)
    print("Removed " + str(yhat.sum()) + " outliers")
    mask = yhat != -1
    train_x, train_y = train_x[mask, :], train_y[mask]
    return train_x, train_y


def make_number(df, dollar_columns=None, numerical_columns=None):
    def f(x):
        if str(x) != "nan":
            x = int(x.replace("$", "").replace(",", ""))
            return x
    if dollar_columns:
        for col in dollar_columns:
            df[col] = df[col].apply(f)
    if numerical_columns:
        for col in numerical_columns:
            df[col] = df[col].fillna(df[col].mean())
    return df


def process_data(df, drop_columns=None, categorical_columns=None,
                 dollar_columns=None, numerical_columns=None, one_hot_threshold=10):
    if categorical_columns:
        df = preprocess_cat_columns(df, categorical_columns, one_hot_threshold)
    if dollar_columns or numerical_columns:
        df = make_number(df, dollar_columns, numerical_columns)
    if drop_columns:
        df = df.drop(drop_columns, axis=1)
    return df


def scale_data(df):
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    return df, scaler


def search_best_model_rf(model, train_x, train_y):

    parameters_for_testing = {
        'n_estimators': [100, 300],
        'max_features': ['auto', 'sqrt', 'log2'],
        'min_samples_split': [2, 4, 6],
        'min_impurity_decrease': [0., 0.2],
        'max_depth': [None, 3],
    }

    model = GridSearchCV(estimator=model, param_grid=parameters_for_testing,
                         n_jobs=32, verbose=10, scoring='neg_mean_squared_error')
    model.fit(train_x, train_y)
    print('best params')
    print(model.best_params_)

    return model


def search_best_model_svm(model, train_x, train_y):
    # defining parameter range
    parameters_for_testing = {'C': [0.1, 1, 10, 100, 1000],
                              'gamma': [1, 0.1, 0.01, 0.001, 0.0001, "scale"],
                              'kernel': ['rbf', 'poly']}

    model = GridSearchCV(estimator=model, param_grid=parameters_for_testing,
                         n_jobs=32, verbose=10, scoring='neg_mean_squared_error')
    model.fit(train_x, train_y)
    print('best params')
    print(model.best_params_)

    return model


def main(args):
    # categorical columns to be one-hot encoded or make numerical
    cat_columns = ["Year", "Production Country", "Genre", "Source", "Production Companies",
                   "Release Month", "Leading Cast 1", "Director 1", "Theatrical Distributor"]
    # columns you want to convert from dollars to number ($400.000 -> 400000)
    dollar_columns = ["Production Budget", "Worldwide Box Office", "Opening Weekend Revenue",
                      "Domestic Box Office", "Marketing Budget", "International Box office",]
    # columns with numerical features (except label)
    numerical_columns = ["Production Budget", "Marketing Budget"]
    label = "Worldwide Box Office"
    #label = "Mao Yan Grade"
    # ... write the name of the column you want to predict here

    # start

    one_hot_threshold = args.one_hot_threshold
    train_dataset = pd.read_csv(args.path_data, encoding='ISO-8859-1')
    drop_columns = [col for col in train_dataset.columns if col not in cat_columns + numerical_columns + [label]]
    train_dataset = process_data(train_dataset, drop_columns, cat_columns, dollar_columns,
                                 numerical_columns, one_hot_threshold)
    train_x = train_dataset.drop([label], axis=1).values
    train_y = train_dataset[label].values
    if args.scale_output:
        train_y, y_scaler = scale_data(train_y.reshape(-1, 1))
    train_x, _ = scale_data(train_x)
    if args.remove_outliers:
        train_x, train_y = remove_outliers(train_x, train_y)
    train_x_, train_y_ = train_x[:-4], train_y[:-4]
    test_x, test_y = train_x[-4:], train_y[-4:]
    train_x, train_y = train_x_, train_y_

    if args.model == "rf":
        model = RandomForestRegressor(max_depth=None, n_estimators=100, max_features='sqrt', min_impurity_decrease=0.2,
                                      min_samples_split=2)
    if args.model == "svr" or args.model == "svm":
        model = SVR(kernel='rbf', C=1, gamma=0.01, epsilon=.1)

    if args.grid_search:
        if args.model == "rf":
            model = search_best_model_rf(model, train_x, train_y)
        if args.model == "svr" or args.model == "svm":
            model = search_best_model_svm(model, train_x, train_y)
    else:
        model.fit(train_x, train_y)

    preds = model.predict(test_x)
    if args.scale_output:
        preds = y_scaler.inverse_transform(preds)
    preds = preds.astype(int)
    if args.save_predictions:
        pd.DataFrame(preds).to_csv(os.path.join(args.path_save, args.model + "_predictions_test.csv"), index=False, header=False)
    print("Predictions on test set")
    print(preds)

    preds = model.predict(train_x)
    if args.scale_output:
        preds = y_scaler.inverse_transform(preds)
        train_y = y_scaler.inverse_transform(train_y)
    preds = preds.astype(int)
    if args.save_predictions:
        pd.DataFrame(preds).to_csv(os.path.join(args.path_save, args.model + "_predictions_train.csv"), index=False, header=False)
    score = rmse(preds, train_y)
    print("rmse train:", score)
    print("mae train:", mean_absolute_error(train_y, preds))
    print("average box worldwide box office", train_y.mean())
    print("done!")

    if not args.grid_search and one_hot_threshold == 0:
        features = ["Year", "Production Country", "Release Month", "Genre", "Source", "Cast1", "Director1",
                    "Theatrical distribution", "Production companies", "Production budget", "Market Budget"]
        if args.model == "rf":
            importances = model.feature_importances_
            std = np.std([tree.feature_importances_ for tree in model.estimators_],
                         axis=0)
            indices = np.argsort(importances)[::-1]
            # Print the feature ranking
            print("Feature ranking:")
            for f in range(train_x.shape[1]):
                print("%d. (%s) feature %d (%f)" % (f + 1, features[indices[f]], indices[f], importances[indices[f]]))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_data', type=str, default='train_ver_2.csv')
    parser.add_argument('--grid_search', default=False, action='store_true')
    parser.add_argument('--remove_outliers', default=False, action='store_true')
    parser.add_argument('--path_save', type=str, default="")
    parser.add_argument('--model', type=str, default='rf')  # rf, svr
    parser.add_argument('--one_hot_threshold', type=int, default=20)  # put 0 to compute feature importance
    parser.add_argument('--scale_output', default=False, action='store_true')
    parser.add_argument('--save_predictions', default=False, action='store_true')
    args = parser.parse_known_args()[0]
    return args


# python box_office_prediction.py --model rf --grid_search --save_predictions RANDOM FOREST
# python box_office_prediction.py --model svr --scale_output --grid_search --save_predictions SVR
# python box_office_prediction.py --model rf --one_hot_threshold 0 FEATURE IMPORTANCE
if __name__ == '__main__':
    args = get_args()
    main(args)
