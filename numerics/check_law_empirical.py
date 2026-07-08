"""Empirical check of the law of robustness (lower-bound direction).

Sphere data x_i ~ Unif(S^{d-1}), labels y_i iid +-1 (pure noise, sigma^2 = 1). For each
width m, an UNCONSTRAINED-weight network f(x)=sum_k a_k ReLU(<w_k,x>+b_k)+<v,x>+c is trained
to mean squared error <= 1/2 (= sigma^2 - eps, eps=1/2) while its path norm
    P(f) = sum_k |a_k| ||w_k|| + ||v||
is penalized. P(f) is an exact upper bound on Lip(f) for a two-layer ReLU net, so driving it
down searches for the least-Lipschitz fitting network -- the adversarial direction for the
theorem. We then MEASURE the true Lipschitz constant by sampling gradients. The measured Lip
can never fall below the theoretical floor sqrt(n/m) if the law holds, and tracks it up to
log factors if the law is near-tight.

d=32, n=256, m in {16,64,256,1024}. CPU torch, single-backward penalty, ~1 min.
"""
import os, math, time
os.environ["OMP_NUM_THREADS"] = "4"
import torch
torch.set_num_threads(4)
torch.manual_seed(20260707)

d, n = 32, 256
X = torch.randn(n, d); X = X / X.norm(dim=1, keepdim=True)
y = (torch.randint(0, 2, (n,)) * 2 - 1).float()

class Net(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(m, d) / math.sqrt(d))
        self.b = torch.nn.Parameter(torch.zeros(m))
        self.a = torch.nn.Parameter(torch.randn(m) / math.sqrt(m))
        self.v = torch.nn.Parameter(torch.zeros(d))
        self.c = torch.nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return torch.relu(x @ self.W.T + self.b) @ self.a + x @ self.v + self.c
    def path_norm(self):
        return (self.a.abs() * self.W.norm(dim=1)).sum() + self.v.norm()

def measure_lip(net, npts=20000, batch=5000):
    mx = 0.0
    for i in range(0, npts, batch):
        Z = torch.randn(batch, d); Z = Z / Z.norm(dim=1, keepdim=True)
        Z.requires_grad_(True)
        g = torch.autograd.grad(net(Z).sum(), Z)[0]
        gt = g - (g * Z.detach()).sum(1, keepdim=True) * Z.detach()   # tangential to sphere
        mx = max(mx, float(gt.norm(dim=1).max()))
    Z = X.clone().requires_grad_(True)
    g = torch.autograd.grad(net(Z).sum(), Z)[0]
    gt = g - (g * X).sum(1, keepdim=True) * X
    return max(mx, float(gt.norm(dim=1).max()))

print(f"{'m':>5} {'MSE':>8} {'pathnorm':>9} {'Lip_hat':>9} {'sqrt(n/m)':>10} {'ratio':>7} {'sec':>6}", flush=True)
for m in [16, 64, 256, 1024]:
    t0 = time.time()
    net = Net(m)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    STEPS = 6000
    for step in range(STEPS):
        opt.zero_grad()
        mse = ((net(X) - y) ** 2).mean()
        if step < STEPS // 3:
            loss = mse
        else:
            # push path norm (hence Lipschitz) down while holding the fit under 1/2
            over = torch.clamp(mse - 0.45, min=0.0)
            loss = 20.0 * over + 3e-3 * net.path_norm() + 0.2 * mse
        loss.backward(); opt.step()
    mse_f = float(((net(X) - y) ** 2).mean().detach())
    pn = float(net.path_norm().detach())
    L = measure_lip(net)
    floor = math.sqrt(n / m)
    print(f"{m:>5} {mse_f:>8.4f} {pn:>9.2f} {L:>9.3f} {floor:>10.3f} {L/floor:>7.3f} {time.time()-t0:>6.1f}", flush=True)

print("\nLaw: measured Lip must exceed floor sqrt(n/m) (ratio > 0) at every m; near-tightness => ratio ~ O(polylog).", flush=True)
