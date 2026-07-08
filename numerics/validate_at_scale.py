"""Large-scale validation of the two novel results, run on the Azure CPU box (pure numpy).

(A) Rigidity lemma (Lemma 3.1): for random canonical two-layer ReLU nets in dimension up to
    d=40 with up to m=80 units -- including planted cancellations -- every canonical kink
    coefficient obeys |alpha_j| <= 2*Lip_hat, where Lip_hat is a (conservative, sampled)
    LOWER bound on the true Lipschitz constant. Passing with an underestimate of Lip is a
    strictly harder test.
(B) Matching upper bound (Prop.): the width-2n spherical-ramp interpolant at n up to 2000,
    d = ceil(400*log n): exact interpolation, pairwise-disjoint caps, and Lip = sqrt(7).
"""
import numpy as np
rng = np.random.default_rng(20260707)
R = 2.0

# ---------- (A) rigidity at scale ----------
def canonicalize(alpha, U, T):
    m, d = U.shape
    v_extra = np.zeros(d); A2, U2, T2 = [], [], []
    for j in range(m):
        u, t, a = U[j].copy(), float(T[j]), float(alpha[j])
        nz = np.nonzero(np.abs(u) > 1e-12)[0]
        if len(nz) == 0:
            continue
        if u[nz[0]] < 0:
            v_extra += a * u; u, t, a = -u, -t, a
        A2.append(a); U2.append(u); T2.append(t)
    A2, U2, T2 = np.array(A2), np.array(U2), np.array(T2)
    keys = {}
    for j in range(len(A2)):
        k = (tuple(np.round(U2[j], 9)), round(float(T2[j]), 9))
        keys.setdefault(k, []).append(j)
    am, Um, Tm = [], [], []
    for k, idx in keys.items():
        a = float(A2[idx].sum())
        if abs(a) < 1e-12: continue
        am.append(a); Um.append(U2[idx[0]]); Tm.append(T2[idx[0]])
    return np.array(am), np.array(Um), np.array(Tm), v_extra

def sampled_lip(alpha, U, T, v, npts=120000):
    m, d = U.shape
    X = rng.standard_normal((npts, d)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    X *= (R * rng.random((npts, 1)) ** (1.0 / d))
    pts = [X]
    # two-sided probes at each kink hyperplane
    for j in range(m):
        t, u = T[j], U[j]
        if abs(t) >= 0.95 * R: continue
        W = rng.standard_normal((8, d)); W -= (W @ u)[:, None] * u
        W /= np.linalg.norm(W, axis=1, keepdims=True)
        rad = np.sqrt(max((0.9 * R) ** 2 - t * t, 0.0)) * rng.random((8, 1))
        base = t * u[None, :] + W * rad
        pts.append(base + 1e-6 * u); pts.append(base - 1e-6 * u)
    P = np.vstack(pts); P = P[np.linalg.norm(P, axis=1) <= R]
    act = (P @ U.T - T[None, :]) > 0
    G = v[None, :] + act @ (alpha[:, None] * U)
    return float(np.max(np.linalg.norm(G, axis=1)))

print("=== (A) rigidity at scale: check max|alpha_j| <= 2*Lip_hat ===")
print(f"{'case':>22} {'d':>4} {'m':>4} {'m0':>4} {'Lip_hat':>9} {'max|a|/2L':>10} {'PASS':>5}")
allok = True
cases = [("generic", 20, 40), ("generic", 30, 60), ("generic", 40, 80),
         ("plant-cancel", 20, 40), ("plant-cancel", 40, 80),
         ("near-parallel", 20, 40), ("near-parallel", 40, 80)]
for name, d, m in cases:
    for _ in range(4):
        U = rng.standard_normal((m, d)); U /= np.linalg.norm(U, axis=1, keepdims=True)
        T = rng.uniform(-1.5, 1.5, m); alpha = rng.standard_normal(m); v = 0.3 * rng.standard_normal(d)
        if name == "plant-cancel":
            U[1] = U[0]; T[1] = T[0]; alpha[0], alpha[1] = 1e6, -1e6 + 0.7
        elif name == "near-parallel":
            U[1] = U[0]; T[1] = T[0] + 1e-3; alpha[0], alpha[1] = 1e4, -1e4
        a2, U2, T2, ve = canonicalize(alpha, U, T)
        L = sampled_lip(a2, U2, T2, v + ve)
        ratio = float(np.max(np.abs(a2)) / (2 * L)) if len(a2) else 0.0
        ok = ratio <= 1.0 + 1e-6
        allok &= ok
        print(f"{name:>22} {d:>4} {m:>4} {len(a2):>4} {L:>9.3g} {ratio:>10.4f} {str(ok):>5}")
print("(A)", "ALL PASS" if allok else "FAILURE")

# ---------- (B) matching upper bound at scale ----------
print("\n=== (B) matching upper bound: exact interp, disjoint caps, Lip=sqrt(7) ===")
print(f"{'n':>6} {'d':>6} {'rho_max':>8} {'interp_err':>11} {'disjoint':>9} {'Lip(analytic)':>13}")
s = 0.25
ramp = lambda u: (np.maximum(u - (1 - s), 0.0) - np.maximum(u - 1.0, 0.0)) / s
ubok = True
for n in [500, 1000, 2000]:
    d = int(np.ceil(400 * np.log(n)))
    X = rng.standard_normal((n, d)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = (rng.integers(0, 2, n) * 2 - 1).astype(float)
    G = X @ X.T; off = G - 2 * np.eye(n); rho = float(off.max())
    F = (ramp(G) * y[None, :]).sum(1); err = float(np.max(np.abs(F - y)))
    disjoint = float(np.sqrt((1 + rho) / 2)) <= 1 - s
    lip = float(np.sqrt((2 - s) / s))
    ubok &= (err < 1e-9) and disjoint
    print(f"{n:>6} {d:>6} {rho:>8.3f} {err:>11.2e} {str(disjoint):>9} {lip:>13.3f}")
print("(B)", "ALL PASS" if ubok else "FAILURE")
print("\nSUMMARY:", "BOTH PASS" if (allok and ubok) else "SOMETHING FAILED")
