"""Static profile rigidity for single-cluster networks (exact form).

For f(x) = <v,x> + c + sum_k alpha_k relu(<u0,x> - t_k) on the sphere, with
v = beta u0 + v_perp and B(s) = beta + sum_{t_k < s} alpha_k, the claim is

    |B(s)| sqrt(1-s^2) <= Lip_S(f)   and   ||v_perp|| <= Lip_S(f)

at every generic s. The spherical Lipschitz constant is computed exactly:
on the level set <u0,x> = s the tangential gradient norm squared is

    B^2 + ||v_perp||^2 - min_phi ( B s + sqrt(1-s^2) ||v_perp|| cos phi )^2,

with the minimum 0 if |B s| <= sqrt(1-s^2) ||v_perp||, else the squared gap.
Lip_S(f) is the max over segments between consecutive kinks (B is constant
per segment; the expression is evaluated on a fine s-grid per segment).
"""

import numpy as np

rng = np.random.default_rng(20260707)


def exact_lip_and_rigidity(d, m):
    t = np.sort(rng.uniform(0.0, 0.9, size=m))
    alpha = rng.standard_normal(m) * rng.choice([0.2, 1.0, 5.0])
    beta = rng.standard_normal() * 2.0
    vperp_norm = abs(rng.standard_normal()) * 2.0

    knots = np.concatenate([[-1.0], t, [1.0]])
    lip2 = 0.0
    rig = 0.0
    for j in range(len(knots) - 1):
        lo, hi = knots[j] + 1e-9, knots[j + 1] - 1e-9
        if hi <= lo:
            continue
        B = beta + alpha[t < (lo + hi) / 2].sum()
        s = np.linspace(lo, hi, 400)
        root = np.sqrt(np.maximum(1 - s * s, 0.0))
        inner = np.abs(B * s)
        reach = root * vperp_norm
        gap = np.where(inner <= reach, 0.0, (inner - reach) ** 2)
        g2 = B * B + vperp_norm ** 2 - gap
        lip2 = max(lip2, g2.max())
        rig = max(rig, (np.abs(B) * root).max())
    lip = np.sqrt(lip2)
    return rig / lip, vperp_norm / lip


worst_r, worst_v = 0.0, 0.0
trials = 2000
for _ in range(trials):
    d = rng.integers(3, 61)
    m = rng.integers(1, 61)
    r, v = exact_lip_and_rigidity(d, m)
    worst_r, worst_v = max(worst_r, r), max(worst_v, v)

print(f"trials: {trials}")
print(f"max |B(s)| sqrt(1-s^2) / Lip_S : {worst_r:.4f}  (claim: <= 1)")
print(f"max ||v_perp|| / Lip_S         : {worst_v:.4f}  (claim: <= 1)")
assert worst_r <= 1.0 + 1e-9 and worst_v <= 1.0 + 1e-9
print("PASS: static profile rigidity holds with the exact constant 1")
