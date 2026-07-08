"""Fast minimal check of the law of robustness (lower-bound direction), for a loaded CPU.

Same setup as check_law_empirical.py but smaller: sphere data, d=24, n=192, iid +-1 labels,
widths m in {24, 96, 384}. Train unconstrained-weight two-layer ReLU nets to MSE <= 1/2 while
penalizing the path norm sum_k |a_k| ||w_k|| + ||v|| (an exact upper bound on Lip for a
two-layer ReLU net), then measure the true sphere-Lipschitz constant. The measured Lip must
exceed the floor sqrt(n/m) if the law holds; near-tightness => ratio ~ O(polylog)."""
import os, math, time
os.environ["OMP_NUM_THREADS"] = "4"
import torch
torch.set_num_threads(4)
torch.manual_seed(20260707)

d, n = 24, 192
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

def measure_lip(net, npts=6000, batch=3000):
    mx = 0.0
    for _ in range(0, npts, batch):
        Z = torch.randn(batch, d); Z = Z / Z.norm(dim=1, keepdim=True)
        Z.requires_grad_(True)
        g = torch.autograd.grad(net(Z).sum(), Z)[0]
        gt = g - (g * Z.detach()).sum(1, keepdim=True) * Z.detach()
        mx = max(mx, float(gt.norm(dim=1).max()))
    Z = X.clone().requires_grad_(True)
    g = torch.autograd.grad(net(Z).sum(), Z)[0]
    gt = g - (g * X).sum(1, keepdim=True) * X
    return max(mx, float(gt.norm(dim=1).max()))

print(f"{'m':>5} {'MSE':>8} {'Lip_hat':>9} {'sqrt(n/m)':>10} {'ratio':>7} {'sec':>6}", flush=True)
for m in [24, 96, 384]:
    t0 = time.time()
    net = Net(m)
    opt = torch.optim.Adam(net.parameters(), lr=8e-3)
    STEPS = 1500
    for step in range(STEPS):
        opt.zero_grad()
        mse = ((net(X) - y) ** 2).mean()
        if step < STEPS // 3:
            loss = mse
        else:
            over = torch.clamp(mse - 0.45, min=0.0)
            loss = 20.0 * over + 3e-3 * net.path_norm() + 0.2 * mse
        loss.backward(); opt.step()
    mse_f = float(((net(X) - y) ** 2).mean().detach())
    L = measure_lip(net)
    floor = math.sqrt(n / m)
    print(f"{m:>5} {mse_f:>8.4f} {L:>9.3f} {floor:>10.3f} {L/floor:>7.3f} {time.time()-t0:>6.1f}", flush=True)
