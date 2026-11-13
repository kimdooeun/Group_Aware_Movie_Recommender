"""Tune MF on a validation split of the training data, then report RMSE/MAE on ratings_test_eval.csv.

    python run_mf.py tune       # grid search -> results/mf_tuning.csv
    python run_mf.py final      # retrain on full train with best params -> results/mf_final.csv, mf_curve.csv
    python run_mf.py export     # all user x movie scores -> results/mf_scores.csv
"""
import itertools
import json
import os
import sys

import pandas as pd

from mf import MatrixFactorization, load_ratings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE_DIR, "results")  # preprocessed CSVs and MF outputs
BEST_PARAMS = os.path.join(RESULTS, "mf_best_params.json")
GRID = {
    "k": [20, 50],
    "lr": [0.005, 0.01],
    "reg": [0.02, 0.05, 0.1],
}
EPOCHS = 20


def split_validation(train, frac=0.1, seed=42):
    val = train.sample(frac=frac, random_state=seed)
    return train.drop(val.index), val


def tune():
    train = load_ratings(os.path.join(RESULTS, "ratings_train.csv"))
    sub_train, val = split_validation(train)  # test_eval is never used for tuning
    rows = []
    for k, lr, reg in itertools.product(GRID["k"], GRID["lr"], GRID["reg"]):
        print("== k=%d lr=%g reg=%g" % (k, lr, reg))
        model = MatrixFactorization(k=k, lr=lr, reg=reg, epochs=EPOCHS)
        hist = model.fit(sub_train, val, verbose=False)
        best = hist.loc[hist["val_rmse"].idxmin()]
        print("   best epoch %d  val_rmse=%.4f val_mae=%.4f"
              % (best["epoch"], best["val_rmse"], best["val_mae"]))
        rows.append({"k": k, "lr": lr, "reg": reg, "best_epoch": int(best["epoch"]),
                     "val_rmse": best["val_rmse"], "val_mae": best["val_mae"]})
    res = pd.DataFrame(rows).sort_values("val_rmse")
    os.makedirs(RESULTS, exist_ok=True)
    res.to_csv(os.path.join(RESULTS, "mf_tuning.csv"), index=False)
    top = res.iloc[0]
    with open(BEST_PARAMS, "w") as f:
        json.dump({"k": int(top["k"]), "lr": top["lr"], "reg": top["reg"],
                   "epochs": int(top["best_epoch"])}, f)
    print(res.head(10).to_string(index=False))


def final():
    with open(BEST_PARAMS) as f:
        p = json.load(f)
    train = load_ratings(os.path.join(RESULTS, "ratings_train.csv"))
    test = load_ratings(os.path.join(RESULTS, "ratings_test_eval.csv"))
    model = MatrixFactorization(**p)
    hist = model.fit(train, test)
    os.makedirs(RESULTS, exist_ok=True)
    hist.to_csv(os.path.join(RESULTS, "mf_curve.csv"), index=False)
    out = pd.DataFrame([{**p, **model.evaluate(train), **{"test_" + k: v for k, v in model.evaluate(test).items()}}])
    out.to_csv(os.path.join(RESULTS, "mf_final.csv"), index=False)
    print(out.to_string(index=False))
    return model, train


def export():
    model, train = final()
    path = os.path.join(RESULTS, "mf_scores.csv")
    for n, chunk in enumerate(model.score_all(exclude_seen=train)):
        chunk.to_csv(path, mode="w" if n == 0 else "a", header=(n == 0), index=False)
    print("saved", path)


if __name__ == "__main__":
    {"tune": tune, "final": final, "export": export}[sys.argv[1]]()
