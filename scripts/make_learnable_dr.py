"""
Replace the PCA initialization of the collaborative item embedding with LEARNABLE
dimensionality-reduction inits.

The collaborative branch's 64-d item embedding is normally initialized from a PCA of the
frozen 1536-d LLM item embeddings (pca64_itm_emb_np.pkl). An established finding in this
project is that this PCA init is load-bearing (random init hurts substantially). Here we
ask whether a *learnable* DR gives an equal/better init:

  - ae64  : NONLINEAR autoencoder bottleneck (1536->512->ReLU->64 ->512->ReLU->1536)
  - lae64 : LINEAR  autoencoder bottleneck   (1536->64 ->1536)  -- control: a gradient-trained
            linear DR recovers the PCA subspace, so this isolates whether *nonlinearity* in the
            DR helps (vs PCA / linear).

Both code matrices are scale-matched to pca64's global std (0.0341) so the comparison
isolates DR *geometry* from scale (a prior finding: magnitude, not geometry, is what broke
over-scaled embeddings). Outputs go to a worktree-local dir (NOT the handled/ symlink).
"""
import os, pickle, time
import numpy as np
import torch
import torch.nn as nn

WT = "/gpfs/work5/0/prjs2120/groups/group_05/workspace/LLM-ESR-ext"
HANDLED = os.path.join(WT, "data/yelp/handled")     # symlink into main repo -> READ ONLY
OUT = os.path.join(WT, "data/yelp/handled_dr")       # worktree-local artifacts
os.makedirs(OUT, exist_ok=True)
TARGET_STD = 0.0341                                  # pca64_itm_emb_np.pkl global std

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", dev, flush=True)

X = pickle.load(open(os.path.join(HANDLED, "itm_emb_np.pkl"), "rb")).astype(np.float32)
N, D = X.shape
print("X (LLM item emb):", X.shape, flush=True)
mu = X.mean(0, keepdims=True)
sd = X.std(0, keepdims=True) + 1e-8
Xs = torch.tensor((X - mu) / sd, device=dev)         # per-dim standardized


def train_ae(nonlinear, epochs=600, lr=1e-3, bottleneck=64, hidden=512, seed=42):
    torch.manual_seed(seed)
    if nonlinear:
        enc = nn.Sequential(nn.Linear(D, hidden), nn.ReLU(), nn.Linear(hidden, bottleneck))
        dec = nn.Sequential(nn.Linear(bottleneck, hidden), nn.ReLU(), nn.Linear(hidden, D))
    else:
        enc = nn.Sequential(nn.Linear(D, bottleneck))
        dec = nn.Sequential(nn.Linear(bottleneck, D))
    enc, dec = enc.to(dev), dec.to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)
    lossf = nn.MSELoss()
    for ep in range(epochs):
        opt.zero_grad()
        loss = lossf(dec(enc(Xs)), Xs)
        loss.backward()
        opt.step()
        if (ep + 1) % 150 == 0:
            print("  [%s] epoch %d/%d  mse %.5f" % ("nonlin" if nonlinear else "linear",
                                                     ep + 1, epochs, loss.item()), flush=True)
    enc.eval()
    with torch.no_grad():
        Z = enc(Xs).cpu().numpy()
        final = lossf(dec(enc(Xs)), Xs).item()
    return Z, final


for name, nl in [("ae", True), ("lae", False)]:
    t0 = time.time()
    Z, mse = train_ae(nl)
    Z = Z * (TARGET_STD / (Z.std() + 1e-12))         # scale-match to pca64
    Z = Z.astype(np.float32)
    assert Z.shape == (N, 64), "bad shape %s (expected (%d, 64))" % (Z.shape, N)
    p = os.path.join(OUT, "%s64_itm_emb_np.pkl" % name)
    pickle.dump(Z, open(p, "wb"))
    print("%-4s saved %s  shape %s  recon_mse %.5f  std %.4f  (%.1fs)"
          % (name, p, Z.shape, mse, Z.std(), time.time() - t0), flush=True)

# sanity: reload + shape check (afterok gate for the dependent runs)
for name in ("ae", "lae"):
    z = pickle.load(open(os.path.join(OUT, "%s64_itm_emb_np.pkl" % name), "rb"))
    assert z.shape == (N, 64), (name, z.shape)
print("VERIFY OK: ae64 + lae64 both (%d, 64)" % N, flush=True)
print("DONE", flush=True)
