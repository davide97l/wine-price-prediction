import pandas as pd

if __name__ == '__main__':
    files = ["results_train/rf_predictions.csv", "results_train/xg_predictions.csv"]
    assert len(files) > 1

    data = pd.read_csv(files[0], header=None).values
    for f in files[1:]:
        df = pd.read_csv(f, header=None).values
        data += df
    data //= len(files)
    pd.DataFrame(data).to_csv("ensemble_predictions.csv", index=False, header=False)
    print("done!")
