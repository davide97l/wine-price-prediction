import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
import argparse
from sklearn import preprocessing
import xgboost
import matplotlib.pyplot as plt


def preprocess_cat_columns(df, columns):
    for col in columns:
        if len(pd.unique(df[col])) <= 5:
            one_hot = pd.get_dummies(df[col])
            df = df.join(one_hot)
            df = df.drop([col], axis=1)
        else:
            le = preprocessing.LabelEncoder()
            le.fit(df[col])
            df[col] = le.transform(df[col])
    return df


def normalize(df, columns):
    x = df[columns].values
    x_scaled = preprocessing.normalize(x)
    df_temp = pd.DataFrame(x_scaled, columns=columns, index=df.index)
    df[columns] = df_temp
    return df


def process_data(df, drop_columns=None, categorical_columns=None, normalize_columns=None):
    if drop_columns:
        df = df.drop(drop_columns, axis=1)
    if categorical_columns:
        df = preprocess_cat_columns(df, categorical_columns)
    if normalize_columns:
        df = normalize(df, normalize_columns)
    return df


def feature_importance(model):
    # Get the booster from the xgbmodel
    booster = model.get_booster()
    # Get the importance dictionary (by gain) from the booster
    importance = booster.get_score(importance_type="gain")
    # make your changes
    for key in importance.keys():
        importance[key] = round(importance[key], 2)
    return importance


def scatter_plot(data, x, y):
    plt.scatter(data[x], data[y], c="blue", marker="s")
    plt.title("Scatter Plot")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()


def search_best_model(model, train_x, train_y):

    parameters_for_testing = {
        'colsample_bytree': [0.5, 1],
        'gamma': [0., 0.3],
        'min_child_weight': [1, 4],
        'max_depth': [3, 5],
        'n_estimators': [10000],
        'reg_lambda': [1],
        'subsample': [0.5, 0.95]
    }

    #{'colsample_bytree': 0.5, 'gamma': 0.3, 'max_depth': 3, 'min_child_weight': 4, 'n_estimators': 10000,
    # 'reg_lambda': 1, 'subsample': 0.95}

    model = GridSearchCV(estimator=model, param_grid=parameters_for_testing,
                         n_jobs=32, verbose=10, scoring='neg_mean_squared_error')
    model.fit(train_x, train_y)
    print('best params')
    print(model.best_params_)
    print('best score')
    print(model.best_score_)

    return model


def xgboost_regressor(args):
    cat_columns = ["country", "province", "winery", "region_1", "designation"]
    drop_columns = ["region_2", "review", "variety"]
    normalize_columns = None  # ["country", "province", "winery", "region_1", "variety", "designation", "review_score"]
    label = "price"

    train_dataset = pd.read_csv(args.path_data, encoding='ISO-8859-1')
    test_dataset = pd.read_csv(args.path_test, encoding='ISO-8859-1')
    train_dataset = process_data(train_dataset, drop_columns, cat_columns, normalize_columns)
    test_dataset = process_data(test_dataset, drop_columns, cat_columns, normalize_columns)
    train_x = train_dataset.drop([label], axis=1)
    train_y = train_dataset[label]

    model = xgboost.XGBRegressor(colsample_bytree=0.5,
                                 gamma=0.3,
                                 max_depth=3,
                                 min_child_weight=4,
                                 n_estimators=10000,
                                 reg_lambda=1.,
                                 subsample=0.95,
                                 seed=42)
    if not args.grid_search:
        model.fit(train_x, train_y)
    else:
        model = search_best_model(model, train_x, train_y)

    print("feature importance: xgb")
    print(feature_importance(model))

    print("predicting")
    xg_preds = np.array(model.predict(test_dataset)).astype(int)
    pd.DataFrame(xg_preds).to_csv(args.path_save, index=False, header=False)
    print("done!")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_data', type=str, default='training_set.csv')
    parser.add_argument('--path_test', type=str, default='test_set.csv')
    parser.add_argument('--grid_search', default=False, action='store_true')
    parser.add_argument('--path_save', type=str, default="xg_predictions.csv")
    args = parser.parse_known_args()[0]
    return args


# python xgboost_regressor.py --path_save "xg_predictions.csv" --grid_search
if __name__ == '__main__':
    args = get_args()
    xgboost_regressor(args)
