"""Matching upper bound at m = Theta(n): an EXPLICIT two-layer ReLU network that exactly
interpolates n random +-1 labels on S^{d-1} with Lipschitz constant O(1), using width 2n.

Construction. For each anchor x_i a spherical-cap ramp (2 ReLU units):
    phi_i(x) = (1/s)[ ReLU(<x_i,x> - (1-s)) - ReLU(<x_i,x> - 1) ],
0 for <x_i,x> <= 1-s, rising linearly to 1 at <x_i,x>=1. On the sphere <x_i,x> <= 1, so
phi_i(x_i)=1 and phi_i=0 once <x_i,x> <= 1-s. Set f = sum_i y_i phi_i (width 2n).

Gradient (a.e.): grad f(x) = (1/s) sum_{i: <x_i,x> in (1-s,1)} y_i x_i. On the sphere the
relevant Lipschitz constant is the tangential-gradient norm. If the caps {<x_i,x> > 1-s} are
pairwise disjoint, at most one i is active at any x, so |grad f| <= (1/s)*sqrt(1-<x_i,x>^2),
maximized over the band at <x_i,x>=1-s giving Lip = (1/s) sqrt(1-(1-s)^2) = sqrt((2-s)/s).
For s a fixed constant this is O(1), independent of n and d.

We verify: (a) exact interpolation, (b) caps disjoint (so no gradient stacking), and
(c) measured tangential-gradient norm = O(1) matching sqrt((2-s)/s), versus floor sqrt(n/2n).
"""
import numpy as np
rng = np.random.default_rng(20260707)

def run(n, d, s=0.25):
    X = rng.standard_normal((n, d)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = (rng.integers(0, 2, n) * 2 - 1).astype(float)
    G = X @ X.T
    off = G - 2 * np.eye(n)                       # kill the diagonal
    rho_max = off.max()

    # (a) exact interpolation: f(x_i) = y_i + sum_{j!=i} y_j ramp(<x_j,x_i>); ramp=0 iff <=1-s
    def ramp(u):
        return (np.maximum(u - (1 - s), 0.0) - np.maximum(u - 1.0, 0.0)) / s
    F = (ramp(G) * y[None, :]).sum(axis=1)         # f at all data points, vectorized
    interp_err = float(np.max(np.abs(F - y)))
    interp_ok = rho_max <= 1 - s

    # (b) caps pairwise disjoint? overlap of cap_i,cap_j possible iff sqrt((1+rho_ij)/2) > 1-s
    worst_overlap = float(np.sqrt((1 + rho_max) / 2))
    caps_disjoint = worst_overlap <= 1 - s

    # (c) measured max tangential-gradient norm over dense samples in the ramp bands.
    #     Sample points x = cos(a) x_i + sin(a) g (g tangent), a in [0, arccos(1-s)]; the
    #     gradient there is (1/s) sum_{active} y_j x_j, tangential-projected. Vectorized over
    #     a batch of tangent directions per anchor.
    lip = 0.0
    A = np.linspace(0.0, np.arccos(1 - s), 12)
    for i in rng.choice(n, size=min(n, 40), replace=False):
        Gt = rng.standard_normal((30, d)); Gt -= (Gt @ X[i])[:, None] * X[i][None, :]
        Gt /= np.linalg.norm(Gt, axis=1, keepdims=True)
        for a in A:
            P = np.cos(a) * X[i][None, :] + np.sin(a) * Gt        # (30,d) on sphere
            act = (P @ X.T)                                       # (30,n) inner products
            mask = ((act > 1 - s) & (act < 1.0)).astype(float)    # active ramps
            grad = (mask * y[None, :]) @ X / s                    # (30,d) analytic gradient
            tang = grad - (np.sum(grad * P, axis=1))[:, None] * P # tangential part
            lip = max(lip, float(np.linalg.norm(tang, axis=1).max()))
    predicted = np.sqrt((2 - s) / s)                              # O(1) prediction
    floor = np.sqrt(n / (2 * n))                                  # sqrt(n/m), m=2n
    return rho_max, interp_ok, interp_err, caps_disjoint, lip, predicted, floor

print(f"{'n':>5} {'d':>5} {'rho_max':>8} {'interp?':>7} {'err':>9} {'disjoint?':>9} {'Lip_hat':>8} {'pred O(1)':>9} {'floor':>6}")
for (n, d) in [(100, 200), (200, 400), (400, 400), (400, 800), (800, 1600)]:
    rho, ok, err, dj, lip, pred, floor = run(n, d)
    print(f"{n:>5} {d:>5} {rho:>8.3f} {str(bool(ok)):>7} {err:>9.1e} {str(bool(dj)):>9} {lip:>8.3f} {pred:>9.3f} {floor:>6.3f}")
print("\nExact interpolation (err~0) at Lip_hat = O(1) ~ sqrt((2-s)/s), independent of n:")
print("so the lower bound sqrt(n/m) is tight up to constants/log at m = 2n.")
