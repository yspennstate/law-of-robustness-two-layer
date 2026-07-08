"""Adversarial check of the localized projection floor at width one.

Theorem genact2 predicts: for every rank-2 projector P (the factorization
space of a width-1 network plus skip), some opposite-label pair satisfies
||P(x_i - x_j)|| <= C sqrt(log(nd))/n, so Lip >= c n/sqrt(log(nd)).
The adversary below maximizes the minimum opposite-label projected gap over
P by soft-min gradient ascent with restarts. The floor predicts g_opt * n
stays bounded (a constant times sqrt(log)) as n grows; a sqrt(n)-type floor
would instead let g_opt * n grow like sqrt(n).
"""

import numpy as np

rng = np.random.default_rng(20260708)


def opt_min_gap(n, d, restarts=12, steps=150, lr=0.15, tau=200.0):
    X = rng.standard_normal((n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = rng.choice([-1.0, 1.0], size=n)
    ip, im = np.where(y > 0)[0], np.where(y < 0)[0]
    pairs = [(i, j) for i in ip for j in im]
    if len(pairs) > 20000:
        idx = rng.choice(len(pairs), 20000, replace=False)
        pairs = [pairs[k] for k in idx]
    D = np.array([X[i] - X[j] for i, j in pairs])

    best = 0.0
    for _ in range(restarts):
        W = rng.standard_normal((d, 2))
        for _ in range(steps):
            Q, _ = np.linalg.qr(W)
            G = D @ Q
            g2 = (G * G).sum(axis=1) + 1e-12
            w = np.exp(-tau * g2)
            w /= w.sum()
            grad = 2 * (D.T * w[None, :] * g2[None, :] ** 0) @ G
            W = Q + lr * grad
        Q, _ = np.linalg.qr(W)
        g = np.sqrt(((D @ Q) ** 2).sum(axis=1)).min()
        best = max(best, g)
    return best


print(f"{'n':>5} {'d':>4} {'g_opt*n':>9} {'sqrt(n)':>8} {'L>=2/(3g)':>10} {'n/sqrt(log nd)':>15}")
for n, d in [(100, 20), (200, 20), (400, 20), (400, 60)]:
    g = opt_min_gap(n, d)
    lam = np.log(n * d)
    print(f"{n:>5} {d:>4} {g * n:9.2f} {np.sqrt(n):8.1f} {2 / (3 * g):10.1f} {n / np.sqrt(lam):15.1f}")
print("floor confirmed if g_opt*n stays flat (does not grow like sqrt(n))")
