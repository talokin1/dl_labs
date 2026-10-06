import numpy as np
from sklearn.datasets import load_iris


def standardize(X, mean, std):
    return (X - mean) / std


def prepare_data():
    dataset = load_iris()
    X = dataset.data.astype(np.float64)
    y = dataset.target
    rng = np.random.default_rng(0)
    train_idx, test_idx = [], []
    for class_ in [0, 1, 2]:
        idx = rng.permutation(np.where(y == class_)[0])
        train_idx.extend(idx[:35])
        test_idx.extend(idx[35:])
    train_idx, test_idx = np.array(train_idx), np.array(test_idx)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=0)
    assert np.all(std > 0)
    X_train = standardize(X_train, mean, std)
    X_test = standardize(X_test, mean, std)
    assert np.array_equal(np.bincount(y_train), [35, 35, 35])
    assert np.array_equal(np.bincount(y_test), [15, 15, 15])
    assert np.intersect1d(train_idx, test_idx).size == 0
    assert np.unique(np.r_[train_idx, test_idx]).size == 150
    assert np.allclose(X_train.mean(axis=0), 0, atol=1e-14)
    assert np.allclose(X_train.std(axis=0), 1)
    return X_train, y_train, X_test, y_test


def initialize():
    rng = np.random.default_rng(0)
    W1 = rng.normal(0, np.sqrt(2 / 4), size=(4, 8))
    W2 = rng.normal(0, np.sqrt(2 / (8 + 3)), size=(8, 3))
    return dict(W1=W1, b1=np.zeros(8), W2=W2, b2=np.zeros(3))


def forward(X, params):
    Z1 = X @ params['W1'] + params['b1']
    A1 = np.maximum(Z1, 0)
    Z2 = A1 @ params['W2'] + params['b2']
    cache = dict(X=X, Z1=Z1, A1=A1)
    return Z2, cache


def cross_entropy(logits, y):
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    loss = -log_probs[np.arange(len(y)), y].mean()
    return loss, np.exp(log_probs)


def backward(params, cache, probs, y, bug=False):
    dZ2 = probs.copy()
    dZ2[np.arange(len(y)), y] -= 1
    if not bug:
        dZ2 /= len(y)
    dW2 = cache['A1'].T @ dZ2
    db2 = dZ2.sum(axis=0)
    dA1 = dZ2 @ params['W2'].T
    dZ1 = dA1 * (cache['Z1'] > 0)
    dW1 = cache['X'].T @ dZ1
    db1 = dZ1.sum(axis=0)
    return dict(W1=dW1, b1=db1, W2=dW2, b2=db2)
