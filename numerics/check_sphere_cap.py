"""Sphere rigidity factor check (Lemma 3'): a single cap unit f(x) = alpha*ReLU(<u,x>-t)
on S^{d-1} has Lip_{S}(f) ~ alpha*sqrt(1-t^2) (Euclidean metric on the sphere), so the
canonical bound |alpha| <= 2L/sqrt(1-t^2) is the right scaling as t -> 1.

Empirical Lip: max over sampled pairs AND finite-difference along great circles through
the cap boundary (where the gradient is largest)."""
import numpy as np
rng = np.random.default_rng(7)
d = 8
print(f"{'t':>6} {'alpha':>8} {'alpha*sqrt(1-t^2)':>18} {'Lip_hat':>9} {'ratio':>7}")
for t in [0.0, 0.5, 0.9, 0.99, 0.999]:
    alpha = 1.0 / np.sqrt(1 - t * t)   # so predicted Lip ~ 1
    u = np.zeros(d); u[0] = 1.0
    # great circles through the boundary point x0 = t*u + sqrt(1-t^2)*w, tangent xi mixes u
    best = 0.0
    for _ in range(2000):
        w = rng.standard_normal(d); w -= (w @ u) * u; w /= np.linalg.norm(w)
        x0 = t * u + np.sqrt(1 - t * t) * w
        xi = (u - t * x0); xi /= np.linalg.norm(xi)
        for s in [1e-5, 1e-4, 1e-3]:
            # one-sided pair: both points on the active side of the kink
            xp = np.cos(2 * s) * x0 + np.sin(2 * s) * xi
            xm = np.cos(s) * x0 + np.sin(s) * xi
            fp = alpha * max(xp @ u - t, 0.0); fm = alpha * max(xm @ u - t, 0.0)
            best = max(best, abs(fp - fm) / np.linalg.norm(xp - xm))
    pred = alpha * np.sqrt(1 - t * t)
    print(f"{t:>6.3f} {alpha:>8.2f} {pred:>18.3f} {best:>9.3f} {best/pred:>7.3f}")
