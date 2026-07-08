"""Sanity check of the rigidity lemma (Lemma 3): after canonicalization, every kink
coefficient satisfies |alpha_j| <= 2 * Lip_{B_R}(f), even when the raw parameterization
uses enormous cancelling weights.

Construction: random ReLU nets in R^d, width m, with (i) generic neurons, (ii) planted
pairs of same-hyperplane neurons with huge opposite coefficients (cancellation), and
(iii) planted near-parallel pairs (near-cancellation, distinct hyperplanes - these do NOT
merge and their individual alphas MUST still obey |alpha| <= 2L).

Lipschitz constant of a CPWL f on B_R: max over activation regions of ||v + sum_active
alpha_j u_j||. We sample regions by sampling many points in B_R (each point's active set
gives its region gradient) -- a lower bound on Lip that converges quickly for small m.
For the check we need Lip exactly enough: we also include, for every kink j, points just
on both sides of H_j near a random point of H_j cap B_R (so both regions adjacent to each
kink are sampled -- these realize the rigidity argument and make the sampled max exact for
the inequality tested).

Run: python check_rigidity.py  (seed fixed; prints PASS/FAIL table)
"""
import numpy as np

rng = np.random.default_rng(20260707)
R = 2.0

def canonicalize(alpha, U, T):
    """Merge neurons with identical (as sets) hyperplanes. U rows unit. Returns merged
    (alpha, U, T) and the absorbed affine part (v_extra, c_extra) from orientation flips.
    Convention: flip (u,t) -> (-u,-t) if u's first nonzero coord is negative, using
    ReLU(-z) = ReLU(z) - z  =>  a*ReLU(<u,x>-t) = a*ReLU(<-u,x>+t) + a*(<u,x>-t)."""
    m, d = U.shape
    v_extra = np.zeros(d); c_extra = 0.0
    A2, U2, T2 = [], [], []
    for j in range(m):
        u, t, a = U[j].copy(), float(T[j]), float(alpha[j])
        nz = np.nonzero(np.abs(u) > 1e-12)[0]
        if len(nz) == 0:
            c_extra += a * max(-t, 0.0); continue
        if u[nz[0]] < 0:
            # a*ReLU(<u,x>-t): substitute z = <u,x>-t, ReLU(z) = ReLU(-z) + z
            v_extra += a * u; c_extra += -a * t
            u, t, a = -u, -t, a  # now a*ReLU(<u',x>-t') with u'=-u,t'=-t
        A2.append(a); U2.append(u); T2.append(t)
    A2, U2, T2 = np.array(A2), np.array(U2), np.array(T2)
    # merge identical hyperplanes (round to tolerance)
    keys = {}
    for j in range(len(A2)):
        k = (tuple(np.round(U2[j], 9)), round(float(T2[j]), 9))
        keys.setdefault(k, []).append(j)
    alpha_m, U_m, T_m = [], [], []
    for k, idx in keys.items():
        a = float(A2[idx].sum())
        if abs(a) < 1e-12: continue
        alpha_m.append(a); U_m.append(U2[idx[0]]); T_m.append(T2[idx[0]])
    return (np.array(alpha_m), np.array(U_m), np.array(T_m), v_extra, c_extra)

def sampled_lip(alpha, U, T, v, npts=40000):
    """Max region-gradient norm over sampled points in B_R + forced two-side kink samples."""
    m, d = U.shape if len(U) else (0, len(v))
    X = rng.standard_normal((npts, d))
    X = X / np.linalg.norm(X, axis=1, keepdims=True) * (R * rng.random((npts, 1)) ** (1/d))
    pts = [X]
    for j in range(m):
        # random point on H_j cap B_{0.9R}: x0 = t*u + w, w perp u, |w| <= sqrt((0.9R)^2-t^2)
        t, u = T[j], U[j]
        if abs(t) >= 0.9 * R: continue
        for _ in range(40):
            w = rng.standard_normal(d); w -= (w @ u) * u
            nw = np.linalg.norm(w)
            if nw < 1e-12: continue
            rad = np.sqrt(max((0.9*R)**2 - t*t, 0.0)) * rng.random()
            x0 = t * u + w / nw * rad
            for s in (+1e-7, -1e-7):
                pts.append((x0 + s * u)[None, :])
    P = np.vstack(pts)
    P = P[np.linalg.norm(P, axis=1) <= R]
    act = (P @ U.T - T[None, :]) > 0 if m else np.zeros((len(P), 0), bool)
    G = v[None, :] + act @ (alpha[:, None] * U)
    return float(np.max(np.linalg.norm(G, axis=1)))

def one_trial(d, m, planted):
    U = rng.standard_normal((m, d)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    T = rng.uniform(-1.5, 1.5, m)
    alpha = rng.standard_normal(m)
    v = rng.standard_normal(d) * 0.3
    if planted == "same-hyperplane-cancel":
        # neurons 0,1: same hyperplane, coefficients +M, -M + delta
        M = 1e6
        U[1] = U[0]; T[1] = T[0]
        alpha[0], alpha[1] = M, -M + 0.7
    elif planted == "near-parallel":
        # neurons 0,1: nearly parallel hyperplanes, huge opposite coefficients:
        # the FUNCTION then has huge Lipschitz constant near the gap - rigidity says
        # |alpha| <= 2L must STILL hold (L will be large). Testing the inequality, not smallness.
        M = 1e4
        U[1] = U[0]; T[1] = T[0] + 1e-3
        alpha[0], alpha[1] = M, -M
    a2, U2, T2, v_extra, _ = canonicalize(alpha, U, T)
    v2 = v + v_extra
    L = sampled_lip(a2, U2, T2, v2)
    ok = np.all(np.abs(a2) <= 2 * L + 1e-6 * max(1, L))
    worst = float(np.max(np.abs(a2)) / (2 * L)) if len(a2) else 0.0
    return ok, worst, len(a2), L

print(f"{'case':>28} {'d':>3} {'m':>3} {'merged m0':>9} {'L':>12} {'max|a|/2L':>10} {'PASS':>5}")
allok = True
for planted in ["generic", "same-hyperplane-cancel", "near-parallel"]:
    for (d, m) in [(3, 6), (5, 10), (10, 20)]:
        for _ in range(3):
            ok, worst, m0, L = one_trial(d, m, planted)
            allok &= ok
            print(f"{planted:>28} {d:>3} {m:>3} {m0:>9} {L:>12.4g} {worst:>10.4f} {str(ok):>5}")
print("\nALL PASS" if allok else "\nFAILURES PRESENT")
