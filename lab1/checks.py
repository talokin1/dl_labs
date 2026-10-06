import numpy as np
import torch
from torch import nn
from model import forward, cross_entropy

POINTS = [('W1', (0, 0)), ('b1', (0,)), ('W2', (0, 0)), ('b2', (0,))]


def torch_reference(X, y, params):
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3)).double()
    with torch.no_grad():
        for layer, W, b in [(model[0], 'W1', 'b1'), (model[2], 'W2', 'b2')]:
            layer.weight.copy_(torch.from_numpy(params[W].T))
            layer.bias.copy_(torch.from_numpy(params[b]))
    logits = model(torch.from_numpy(X))
    loss = nn.CrossEntropyLoss(reduction='mean')(logits, torch.from_numpy(y).long())
    loss.backward()
    grads = {}
    for layer, W, b in [(model[0], 'W1', 'b1'), (model[2], 'W2', 'b2')]:
        grads[W] = layer.weight.grad.numpy().T.copy()
        grads[b] = layer.bias.grad.numpy().copy()
    return loss.item(), grads


def compare_torch(loss, grads, ref_loss, ref_grads):
    rows = []
    for name in ['loss', 'W1', 'b1', 'W2', 'b2']:
        a, b = (loss, ref_loss) if name == 'loss' else (grads[name], ref_grads[name])
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
        diff = float(np.max(np.abs(a - b)))
        rows.append(dict(name=name, max_abs_diff=diff, passed=finite and diff <= 1e-12))
    return rows


def numerical_check(X, y, params, grads, eps=1e-6):
    rows = []
    for name, idx in POINTS:
        original = params[name][idx].copy()
        try:
            params[name][idx] = original + eps
            logits, plus_cache = forward(X, params)
            plus, _ = cross_entropy(logits, y)
            params[name][idx] = original - eps
            logits, minus_cache = forward(X, params)
            minus, _ = cross_entropy(logits, y)
        finally:
            params[name][idx] = original
        numerical = (plus - minus) / (2 * eps)
        manual = grads[name][idx]
        diff = abs(numerical - manual)
        finite = np.isfinite([plus, minus, numerical, manual]).all()
        same_mask = np.array_equal(plus_cache['Z1'] > 0, minus_cache['Z1'] > 0)
        rows.append(dict(name=f'{name}{list(idx)}', manual=float(manual),
                         numerical=float(numerical), abs_diff=float(diff),
                         relu_mask_unchanged=bool(same_mask),
                         passed=bool(finite and diff <= 1e-7)))
    return rows
