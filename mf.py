"""Matrix Factorization (biased MF) trained with plain SGD.

    r_hat(u, i) = mu + b_u[u] + b_i[i] + P[u] . Q[i]

Loss per rating (L2 regularization):
    e^2 + reg * (b_u^2 + b_i^2 + |P_u|^2 + |Q_i|^2)
"""
import time

import numpy as np
import pandas as pd

try:  # optional speed-up; the pure Python loop is identical
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        return lambda f: f

MIN_RATING, MAX_RATING = 1.0, 5.0


def load_ratings(path):
    return pd.read_csv(path, encoding="utf-8-sig")


@njit(cache=True)
def _sgd_epoch(users, items, ratings, order, mu, bu, bi, P, Q, lr, reg):
    for idx in order:
        u, i, r = users[idx], items[idx], ratings[idx]
        pu, qi = P[u].copy(), Q[i].copy()
        err = r - (mu + bu[u] + bi[i] + np.dot(pu, qi))
        bu[u] += lr * (err - reg * bu[u])
        bi[i] += lr * (err - reg * bi[i])
        P[u] += lr * (err * qi - reg * pu)
        Q[i] += lr * (err * pu - reg * qi)


class MatrixFactorization:
    def __init__(self, k=20, lr=0.01, reg=0.05, epochs=20, lr_decay=1.0, seed=42):
        self.k, self.lr, self.reg = k, lr, reg
        self.epochs, self.lr_decay, self.seed = epochs, lr_decay, seed

    def fit(self, train, val=None, verbose=True):
        """train/val: DataFrames with userId, movieId, rating. Returns per-epoch history."""
        rng = np.random.RandomState(self.seed)
        self.user_ids = np.sort(train["userId"].unique())
        self.item_ids = np.sort(train["movieId"].unique())
        self.u_idx = {u: n for n, u in enumerate(self.user_ids)}
        self.i_idx = {i: n for n, i in enumerate(self.item_ids)}

        users = train["userId"].map(self.u_idx).values.astype(np.int64)
        items = train["movieId"].map(self.i_idx).values.astype(np.int64)
        ratings = train["rating"].values.astype(np.float64)

        self.mu = ratings.mean()
        self.bu = np.zeros(len(self.user_ids))
        self.bi = np.zeros(len(self.item_ids))
        self.P = rng.normal(0, 0.1, (len(self.user_ids), self.k))
        self.Q = rng.normal(0, 0.1, (len(self.item_ids), self.k))

        history, lr = [], self.lr
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            order = rng.permutation(len(ratings))
            _sgd_epoch(users, items, ratings, order, self.mu,
                       self.bu, self.bi, self.P, self.Q, lr, self.reg)
            row = {"epoch": epoch,
                   "train_rmse": self.evaluate(train)["rmse"]}
            if val is not None:
                row.update({"val_" + k: v for k, v in self.evaluate(val).items()})
            history.append(row)
            if verbose:
                print("  ".join("%s=%.4f" % (k, v) if k != "epoch" else "epoch %2d" % v
                                for k, v in row.items()) + "  (%.1fs)" % (time.time() - t0))
            lr *= self.lr_decay
        return pd.DataFrame(history)

    def predict(self, user_ids, item_ids):
        """Predicted ratings clipped to [1, 5]; unknown users/items fall back to the bias terms."""
        u = np.array([self.u_idx.get(x, -1) for x in user_ids])
        i = np.array([self.i_idx.get(x, -1) for x in item_ids])
        pred = np.full(len(u), self.mu)
        known_u, known_i = u >= 0, i >= 0
        pred[known_u] += self.bu[u[known_u]]
        pred[known_i] += self.bi[i[known_i]]
        both = known_u & known_i
        pred[both] += np.sum(self.P[u[both]] * self.Q[i[both]], axis=1)
        return np.clip(pred, MIN_RATING, MAX_RATING)

    def evaluate(self, df):
        err = df["rating"].values - self.predict(df["userId"].values, df["movieId"].values)
        return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err)))}

    def score_all(self, exclude_seen=None, chunk=500):
        """Yield DataFrames (userId, movieId, score) for every user x candidate movie.

        exclude_seen: ratings DataFrame whose (user, movie) pairs are skipped.
        """
        seen_keys = None
        if exclude_seen is not None:
            seen_keys = (exclude_seen["userId"].values.astype(np.int64) * 100000
                         + exclude_seen["movieId"].values)
        for start in range(0, len(self.user_ids), chunk):
            uidx = np.arange(start, min(start + chunk, len(self.user_ids)))
            scores = (self.mu + self.bu[uidx, None] + self.bi[None, :]
                      + self.P[uidx] @ self.Q.T)
            scores = np.clip(scores, MIN_RATING, MAX_RATING)
            df = pd.DataFrame({
                "userId": np.repeat(self.user_ids[uidx], len(self.item_ids)),
                "movieId": np.tile(self.item_ids, len(uidx)),
                "score": scores.ravel().round(4),
            })
            if seen_keys is not None:
                key = df["userId"].values.astype(np.int64) * 100000 + df["movieId"].values
                df = df[~np.isin(key, seen_keys)]
            yield df
