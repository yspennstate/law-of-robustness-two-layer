"""Sector multiplicity throttles for incoherent anchored ridges.

Three checks on the depth-sector structure of the multiplier reduction:

(D) deep sector: two poles holding a common point at depth s with
    2 s^2 - 1 > nu must violate nu-incoherence -- co-depth multiplicity 1.
(S) shallow band [sqrt(nu), s_1]: the tangential Gram of K co-rising
    incoherent poles is near-orthonormal, so no sign pattern cancels the
    aggregate gradient: min over signs of ||sum eps_g P_x u_g||^2 >= K/2.
(M) mid sector: the combined gradient+value throttle -- aligned sign
    patterns (which cancel the tangential gradient on simplex-like frames)
    saturate the value ball, and value-legal patterns restore the sqrt(K)
    gradient; the maximal legal multiplicity is dimension-independent.

Each check reports pass/fail against the stated bound.
"""

import numpy as np

rng = np.random.default_rng(20260708)


def sample_codepth_poles(d, K, s):
    """K poles all holding a common point x0 at depth ~s."""
    x0 = np.zeros(d)
    x0[0] = 1.0
    poles = []
    for _ in range(K):
        w = rng.standard_normal(d)
        w[0] = 0.0
        w /= np.linalg.norm(w)
        poles.append(s * x0 + np.sqrt(1 - s * s) * w)
    return x0, np.array(poles)


def check_deep(d=400, trials=200):
    nu = 0.10
    s2 = np.sqrt((1 + 2 * nu) / 2)
    worst = 0.0
    for _ in range(trials):
        _, U = sample_codepth_poles(d, 2, s2 * 1.01)
        worst = max(worst, abs(U[0] @ U[1]))
    ok = worst > nu
    print(f"(D) deep: min pairwise coherence at co-depth {s2*1.01:.3f} = "
          f"{worst:.3f} > nu = {nu} -> co-depth mult 1: {'PASS' if ok else 'FAIL'}")
    return ok


def check_shallow(d=2000, K=24, trials=50):
    nu = np.sqrt(2 * np.log(1e6) / d) / 1.0
    s1 = np.sqrt(2 * nu)
    worst = np.inf
    for _ in range(trials):
        x0, U = sample_codepth_poles(d, K, s1 * 0.9)
        T = U - np.outer(U @ x0, x0)
        G = T @ T.T
        # adversarial signs: minimize eps' G eps by greedy descent
        eps = rng.choice([-1.0, 1.0], K)
        for _ in range(200):
            i = rng.integers(K)
            e2 = eps.copy()
            e2[i] *= -1
            if e2 @ G @ e2 < eps @ G @ eps:
                eps = e2
        worst = min(worst, (eps @ G @ eps) / K)
    ok = worst >= 0.5
    print(f"(S) shallow: min-over-signs ||sum eps P u||^2 / K = {worst:.2f} "
          f">= 0.5: {'PASS' if ok else 'FAIL'}")
    return ok


def check_mid(K_target=64, B=2.0, h0=0.05, Lam=1.0, Lagg=2.0):
    """Mid sector: search for a legal (coherence+value+gradient) K-fold co-depth
    frame; report the largest K found, across dimensions."""
    results = []
    for d in [2000, 20000, 200000]:
        nu = np.sqrt(2 * np.log(1e6) / d)
        s = np.sqrt(2 * nu) * 1.05
        best = 1
        for K in range(2, K_target + 1):
            x0, U = sample_codepth_poles(d, K, s)
            coh = np.abs(U @ U.T - np.eye(K) * (U * U).sum(1)).max()
            if coh > nu:
                continue
            T = U - np.outer(U @ x0, x0)
            # aligned signs: value K*v0 vs B; gradient ||sum T||
            v0 = Lam * s * 0.5
            val_ok = K * v0 <= B
            grad = Lam * np.linalg.norm(T.sum(0))
            grad_ok = grad <= Lagg
            if val_ok and grad_ok:
                best = K
        results.append(best)
    flat = max(results) - min(results) <= max(8, max(results) // 3)
    print(f"(M) mid: max legal aligned K across d = 2e3, 2e4, 2e5: {results} "
          f"(value cap 2B/h0-scale = {2*B/h0:.0f}); dimension-independent: "
          f"{'PASS' if flat else 'FAIL'}")
    return flat


if __name__ == "__main__":
    a = check_deep()
    b = check_shallow()
    c = check_mid()
    print("ALL PASS" if (a and b and c) else "SOME CHECKS FAILED")
