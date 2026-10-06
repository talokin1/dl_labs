import argparse
import json
import platform
import numpy as np
import sklearn
import torch
from model import prepare_data, initialize, forward, cross_entropy, backward
from checks import torch_reference, compare_torch, numerical_check

PREDICTION = ('Без /N втрата не зміниться. Кожен ненульовий градієнт стане у 105 разів '
              'більшим. Порівняння втрати пройде, перевірки градієнтів PyTorch і '
              'чисельними різницями виявлять помилку. Нульові градієнти залишаться нульовими.')


def run(bug=False):
    X, y, X_test, y_test = prepare_data()
    params = initialize()
    initial = {name: value.copy() for name, value in params.items()}
    if bug:
        print('Прогноз перед дослідом:', PREDICTION)
    logits, cache = forward(X, params)
    loss, probs = cross_entropy(logits, y)
    grads = backward(params, cache, probs, y, bug=bug)
    ref_loss, ref_grads = torch_reference(X, y, params)
    torch_rows = compare_torch(loss, grads, ref_loss, ref_grads)
    numerical_rows = numerical_check(X, y, params, grads)
    assert all(np.array_equal(params[k], initial[k]) for k in params)
    assert all(v.dtype == np.float64 for v in [X, X_test, logits, probs, *params.values(), *grads.values()])
    assert all(np.isfinite(v).all() for v in [logits, probs, *grads.values()])
    assert np.allclose(probs.sum(axis=1), 1)
    scaling_error = None
    if bug:
        correct = backward(params, cache, probs, y)
        scaling_error = max(float(np.max(np.abs(grads[k] - len(y) * correct[k]))) for k in grads)
        assert all(np.allclose(grads[k], len(y) * correct[k], rtol=1e-13, atol=1e-13) for k in grads)
        assert torch_rows[0]['passed']
        assert all(not r['passed'] for r in torch_rows[1:] + numerical_rows)
        assert all(r['passed'] for r in compare_torch(loss, correct, ref_loss, ref_grads))
    else:
        assert all(r['passed'] for r in torch_rows + numerical_rows)
    result = dict(mode='bug' if bug else 'correct', loss_numpy=float(loss), loss_torch=ref_loss,
                  torch_comparison=torch_rows, numerical_check=numerical_rows,
                  prediction=PREDICTION if bug else None, scaling_error=scaling_error,
                  train_shape=list(X.shape), test_shape=list(X_test.shape),
                  versions=dict(python=platform.python_version(), numpy=np.__version__,
                                sklearn=sklearn.__version__, torch=torch.__version__))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bug', action='store_true', help='Пропустити /N у backward')
    args = parser.parse_args()
    run(args.bug)
