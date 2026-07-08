"""d = 2: coefficient rigidity is GENUINELY FALSE on the circle (Remark in the paper).

Construction (even-cycle cancellation). On S^1, unit j is alpha_j*ReLU(<u_j,x>-t_j) with
u_j at angle theta_j and t_j = cos(a_j); its two kink points are at angles theta_j +- a_j,
and by convexity of ReLU it contributes a one-sided-derivative jump of the SAME sign,
magnitude |alpha_j|*sin(a_j) = |alpha_j|*sqrt(1-t_j^2), at BOTH points. Choose four units
whose eight kink points collapse to four shared angles forming an even cycle, and
alternating coefficients alpha_j = kappa*(-1)^(j+1)/sin(a_j): every jump cancels exactly,
the function is C^1 on S^1, each |alpha_j|*sqrt(1-t_j^2) = kappa, yet Lip(f) << kappa.
So no bound |alpha_j|*sqrt(1-t_j^2) <= 2*Lip can hold at d = 2 (it does hold for d >= 3:
Lemma 3.3 of the paper).

theta = (0, 0.05, 0.04, -0.01), a = (0.10, 0.05, 0.04, 0.09):
kink angles: {+-0.10}, {0, 0.10}, {0, 0.08}, {-0.10, 0.08} -> shared angles
-0.10:(1,4), 0.00:(2,3), 0.08:(3,4), 0.10:(1,2) — a 4-cycle.
"""
import numpy as np

kappa = 1000.0
theta = np.array([0.0, 0.05, 0.04, -0.01])
a = np.array([0.10, 0.05, 0.04, 0.09])
t = np.cos(a)
alpha = kappa * np.array([1, -1, 1, -1]) / np.sin(a)

def f(phi):
    return sum(alpha[j] * np.maximum(np.cos(phi - theta[j]) - t[j], 0.0) for j in range(4))

# per-unit canonical quantity
per_unit = np.abs(alpha) * np.sqrt(1 - t ** 2)

# jumps of f' at the four shared angles (finite differences of the derivative)
def fprime(phi, h=1e-9):
    return (f(phi + h) - f(phi - h)) / (2 * h)
jumps = []
for ang in [-0.10, 0.00, 0.08, 0.10]:
    jumps.append(fprime(ang + 1e-6) - fprime(ang - 1e-6))

# Lipschitz constant on S^1 (arc-length derivative = d f / d phi; chord vs arc equal in the limit)
phi = np.linspace(-np.pi, np.pi, 4_000_001)
vals = f(phi)
lip = np.max(np.abs(np.diff(vals))) / (phi[1] - phi[0])

print("per-unit |alpha_j| sqrt(1-t_j^2):", per_unit)
print("derivative jumps at shared angles (should be ~0):", [f"{j:.3e}" for j in jumps])
print(f"Lip(f) on S^1 ~ {lip:.1f};  kappa = {kappa:.0f};  ratio kappa/(2 Lip) = {kappa/(2*lip):.2f}")
print("rigidity would require ratio <= 1; ratio > 1 => rigidity FALSE at d=2")
