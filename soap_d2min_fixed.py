# %% [markdown]
# # SOAP -> future D2min, corrected pipeline (LDA -> HDA, TIP4P/2005, 80 K, 20 MPa/ns)
#
# Fixes compared with the earlier notebooks:
#   1. Real box length from each fort file (no hard-coded const = 50).
#   2. Optional time-averaged positions to remove thermal vibration before SOAP.
#   3. Labels are defined PER FRAME (top / bottom fraction of D2min in that frame),
#      so a model cannot score well just by knowing the compression stage.
#   4. SOAP is centered per frame (removes the global density/structure drift).
#   5. Train/test split by blocks of frames with a gap >= dp (no overlapping windows,
#      no "leftover" particles from training frames in the test set).
#   6. Baseline: AUC from "frame only"; must be beaten to claim structural signal.
#   7. Evaluation as AUC per frame and as a function of prediction horizon (lag).
#
# Run it as cells in Jupyter / VS Code (# %% markers), or as a script.

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from dscribe.descriptors import SOAP
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import spearmanr

# %%
# ---------------- parameters ----------------
n_particles = 4000
dp = 10                         # D2min window length in frames
lower_frame_bound = 200
upper_frame_bound = 700
d2min_file = f"/media/gadha/Googok/Sarika_Mohit_Project/nonaffine/d2min_delf_{dp}_scaled.dat"
traj_dir = "/media/gadha/Googok/Sarika_Mohit_Project/data/"

# run_const_nn_to_sir.sh uses ref = f, current = f + dp, so row k is the window
# [f, f+dp] with f = lower_frame_bound + k. SOAP must come from the START of the window.
# Still verify that fort.(f+1000) is the same frame as .gro frame f (1-based) - no off-by-one.
# Note: D2min is in reduced units (positions scaled by rho^(1/3)); 1 unit ~ 9.6 A^2.
d2min_row_is_window_start = True

use_real_box = True             # read the box from line 1 of each fort file
time_avg_half_width = 2         # average positions over f-w..f+w (0 = off); IS are better still

r_cut, n_max, l_max, sigma = 6.0, 6, 6, 0.4
mobile_frac = 0.10              # top 10 % of D2min in a frame -> label 1
immobile_frac = 0.50            # bottom 50 % in a frame -> label 0, the rest unused
coarse_grain_shell = 3.5        # average SOAP over neighbours within this distance (0 = off)

# %%
# ---------------- D2min ----------------
n_rows = upper_frame_bound - lower_frame_bound - (dp - 1)


def read_d2min(filename):
    d2 = np.zeros((n_rows, n_particles))
    with open(filename) as f:
        for k in range(n_rows):
            for i in range(n_particles):
                d2[k, i] = float(f.readline().split()[1])
    return d2


d2min = read_d2min(d2min_file)
row_frames = lower_frame_bound + np.arange(n_rows)
# frame whose structure should be used to predict row k
soap_frame_of_row = row_frames if d2min_row_is_window_start else row_frames - dp

# %%
# ---------------- coordinates ----------------
def read_fort(frame):
    path = os.path.join(traj_dir, f"fort.{frame + 1000}")
    with open(path) as f:
        box = float(f.readline())
        frac = np.loadtxt(f, max_rows=n_particles)
    # The old notebooks multiplied by 50 regardless of the box: under compression
    # the real box shrinks by several percent, so every SOAP distance was wrong.
    # If line 1 is NOT the box length in Angstrom, fix this here.
    if not use_real_box:
        box = 50.0
    return frac, box


def averaged_atoms(frame, w):
    frac0, box0 = read_fort(frame)
    if w == 0:
        return Atoms(f"Si{n_particles}", positions=frac0 * box0,
                     cell=[box0] * 3, pbc=True)
    acc = np.zeros_like(frac0)
    boxes = []
    for g in range(frame - w, frame + w + 1):
        frac, box = read_fort(g)
        d = frac - frac0
        d -= np.round(d)                # unwrap relative to the central frame
        acc += frac0 + d
        boxes.append(box)
    frac_avg = (acc / (2 * w + 1)) % 1.0
    box = np.mean(boxes)
    return Atoms(f"Si{n_particles}", positions=frac_avg * box, cell=[box] * 3, pbc=True)


soap = SOAP(species=["Si"], periodic=True, r_cut=r_cut, n_max=n_max,
            l_max=l_max, sigma=sigma, rbf="gto")


def coarse_grain(desc, atoms, rc):
    if rc <= 0:
        return desc
    from ase.neighborlist import neighbor_list
    i, j = neighbor_list("ij", atoms, rc)
    out = desc.copy()
    np.add.at(out, i, desc[j])
    counts = np.bincount(i, minlength=len(atoms)) + 1
    return out / counts[:, None]


features = {}                   # frame -> (n_particles, n_features)
for f in np.unique(soap_frame_of_row):
    atoms = averaged_atoms(int(f), time_avg_half_width)
    x = soap.create(atoms)
    x = np.hstack([x, coarse_grain(x, atoms, coarse_grain_shell)])
    x -= x.mean(axis=0)         # per-frame centering: remove the global drift
    features[int(f)] = x.astype(np.float32)
print("SOAP feature length:", next(iter(features.values())).shape[1])

# %%
# ---------------- per-frame labels ----------------
def per_frame_labels(d2_row):
    lab = np.full(n_particles, -1)
    order = np.argsort(d2_row)
    lab[order[: int(immobile_frac * n_particles)]] = 0
    lab[order[-int(mobile_frac * n_particles):]] = 1
    return lab


labels = np.array([per_frame_labels(r) for r in d2min])

# %%
# ---------------- splits by frame blocks with a gap ----------------
# Alternate blocks of 40 frames between train and test and drop dp frames at every
# boundary, so no train window overlaps a test window.
block = 40
blk = (row_frames - lower_frame_bound) // block
is_train = blk % 2 == 0
pos_in_block = (row_frames - lower_frame_bound) % block
keep = (pos_in_block >= dp) & (pos_in_block < block - dp)
train_rows = np.where(is_train & keep)[0]
test_rows = np.where(~is_train & keep)[0]


def stack(rows, lag=0):
    X, y, fr = [], [], []
    for k in rows:
        k_target = k + lag
        if k_target >= n_rows:
            continue
        lab = labels[k_target]
        m = lab != -1
        X.append(features[int(soap_frame_of_row[k])][m])
        y.append(lab[m])
        fr.append(np.full(m.sum(), k))
    return np.vstack(X), np.concatenate(y), np.concatenate(fr)


X_tr, y_tr, f_tr = stack(train_rows)
X_te, y_te, f_te = stack(test_rows)
print("train", X_tr.shape, y_tr.mean(), "| test", X_te.shape, y_te.mean())

# %%
# ---------------- models ----------------
# Class balancing is done with sample weights so this works on any scikit-learn
# version (HistGradientBoostingClassifier only got class_weight in 1.2).
def make_hgb():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)


def fit_balanced(model, X, y):
    w = compute_sample_weight("balanced", y)
    if hasattr(model, "steps"):                    # Pipeline: weight goes to the last step
        model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": w})
    else:
        model.fit(X, y, sample_weight=w)
    return model


models = {
    "LogReg (PCA 30)": make_pipeline(StandardScaler(), PCA(30),
                                     LogisticRegression(max_iter=2000)),
    "HGB": make_hgb(),
}
proba = {}
for name, m in models.items():
    fit_balanced(m, X_tr, y_tr)
    proba[name] = m.predict_proba(X_te)[:, 1]
    print(f"{name:16s} pooled test AUC = {roc_auc_score(y_te, proba[name]):.3f}")

# Frame-only baseline: with per-frame labels the class ratio is identical in every
# frame, so this should sit at 0.5. If it does not, labels still leak the frame.
frame_rate = {k: labels[k][labels[k] != -1].mean() for k in np.unique(f_te)}
print("frame-only baseline AUC =", roc_auc_score(y_te, [frame_rate[k] for k in f_te]))

# %%
# ---------------- AUC per frame (where in the compression does structure predict?) ----------------
name = "HGB"
fr_u = np.unique(f_te)
auc_f = [roc_auc_score(y_te[f_te == k], proba[name][f_te == k]) for k in fr_u]
plt.figure(figsize=(9, 4))
plt.plot(row_frames[fr_u], auc_f, "o", ms=3)
plt.axhline(0.5, color="gray", ls="--")
plt.xlabel("frame (start of D2min window)")
plt.ylabel("per-frame AUC")
plt.title(f"{name}: SOAP(f) -> top {int(mobile_frac*100)}% D2min in [f, f+{dp}]")
plt.tight_layout()
plt.show()

# %%
# ---------------- prediction horizon ----------------
# SOAP at frame f predicting D2min in the window starting at f + lag.
# This is the real test of "can I tell from the structure that a change will happen there".
lags = [0, 10, 20, 40, 80]
for lag in lags:
    Xa, ya, _ = stack(train_rows, lag)
    Xb, yb, _ = stack(test_rows, lag)
    m = fit_balanced(make_hgb(), Xa, ya)
    print(f"lag {lag:3d} frames: test AUC = {roc_auc_score(yb, m.predict_proba(Xb)[:, 1]):.3f}")

# %%
# ---------------- sanity check: is D2min itself persistent? ----------------
# Upper bound on what any structural descriptor can do: how well does the PAST
# window's D2min rank the NEXT window's D2min for the same particle?
for lag in [dp, 2 * dp, 5 * dp]:
    rho = np.mean([spearmanr(d2min[k], d2min[k + lag])[0]
                   for k in range(0, n_rows - lag, 5)])
    print(f"Spearman(D2min[f], D2min[f+{lag}]) averaged over frames = {rho:.3f}")

# %% [markdown]
# ## Next steps
#
# Results so far (conf1): pooled test AUC 0.61 (HGB) / 0.56 (LogReg), frame-only
# baseline 0.50. Per-frame AUC is about 0.62-0.65 from frame 250 to 500, drops to
# about 0.59 near frame 575 and about 0.52 near frame 660 (HDA, nothing left to predict).
#
# The limiting factor is the label, not SOAP: a molecule's D2min in one 10-frame
# window has Spearman 0.22 with its D2min in the next window, and 0.06 two windows
# later. A label that barely predicts itself cannot be predicted well from structure.
#
# The cells below (a) build a less noisy target, cumulative D2min over a longer
# horizon H, and (b) compare SOAP with four simple local-structure descriptors.

# %%
# ---------------- (a) cumulative D2min over a longer horizon ----------------
# target_H[k] = sum of the non-overlapping dp-windows that cover [f, f+H].
# Summing several windows averages out the thermal part, which is uncorrelated
# from window to window, while real rearrangements add up.
def cumulative_target(H):
    n_win = H // dp
    n_ok = n_rows - (n_win - 1) * dp
    return np.array([d2min[k:k + n_win * dp:dp].sum(axis=0) for k in range(n_ok)])


def block_split(n_ok, gap, train_len=40):
    # blocks: [gap | train_len usable | gap], alternate train / test
    blen = train_len + 2 * gap
    pos = np.arange(n_ok) % blen
    blk = np.arange(n_ok) // blen
    keep = (pos >= gap) & (pos < blen - gap)
    return np.where(keep & (blk % 2 == 0))[0], np.where(keep & (blk % 2 == 1))[0]


def stack_target(rows, lab_arr, feat=None):
    feat = features if feat is None else feat
    X, y, fr = [], [], []
    for k in rows:
        m = lab_arr[k] != -1
        X.append(feat[int(soap_frame_of_row[k])][m])
        y.append(lab_arr[k][m])
        fr.append(np.full(m.sum(), k))
    return np.vstack(X), np.concatenate(y), np.concatenate(fr)


results_H = {}
for H in [10, 30, 50, 80]:
    T = cumulative_target(H)
    # persistence of this target: rank correlation with the NEXT, non-overlapping horizon
    pers = np.mean([spearmanr(T[k], T[k + H])[0] for k in range(0, len(T) - H, 5)])
    lab_H = np.array([per_frame_labels(r) for r in T])
    tr, te = block_split(len(T), gap=max(H, dp))
    Xa, ya, _ = stack_target(tr, lab_H)
    Xb, yb, fb = stack_target(te, lab_H)
    m = fit_balanced(make_hgb(), Xa, ya)
    pb = m.predict_proba(Xb)[:, 1]
    results_H[H] = (pb, yb, fb)
    print(f"H = {H:3d} frames: target persistence rho = {pers:.3f} | "
          f"SOAP test AUC = {roc_auc_score(yb, pb):.3f}  "
          f"(train frames {len(tr)}, test frames {len(te)})")

# %%
# per-frame AUC for the longest horizon
H = max(results_H)
pb, yb, fb = results_H[H]
fu = np.unique(fb)
plt.figure(figsize=(9, 4))
plt.plot(row_frames[fu], [roc_auc_score(yb[fb == k], pb[fb == k]) for k in fu], "o", ms=3)
plt.axhline(0.5, color="gray", ls="--")
plt.xlabel("frame f")
plt.ylabel("per-frame AUC")
plt.title(f"HGB: SOAP(f) -> top {int(mobile_frac*100)}% cumulative D2min over [f, f+{H}]")
plt.tight_layout()
plt.show()

# %%
# ---------------- (b) simple descriptors: d5, LSI, q_tet, local density ----------------
# If these do as well as SOAP, the signal is the familiar LDA/HDA one (interstitial
# 5th neighbour). If neither beats ~0.6, the limit is the label noise.
from scipy.spatial import cKDTree


def simple_descriptors(atoms, lsi_cut=3.7, dens_cut=3.5, k=16):
    L = atoms.cell.lengths()[0]
    pos = atoms.positions % L
    tree = cKDTree(pos, boxsize=L)
    dist, idx = tree.query(pos, k=k + 1)
    dist, idx = dist[:, 1:], idx[:, 1:]                     # drop self
    d5 = dist[:, 4]
    # LSI: variance of gaps between consecutive neighbour distances below lsi_cut
    lsi = np.zeros(len(pos))
    for i in range(len(pos)):
        n = np.searchsorted(dist[i], lsi_cut)
        gaps = np.diff(dist[i, :n + 1])
        lsi[i] = gaps.var() if len(gaps) > 0 else 0.0
    # q_tet from the 4 nearest neighbours
    v = pos[idx[:, :4]] - pos[:, None, :]
    v -= L * np.round(v / L)
    v /= np.linalg.norm(v, axis=2, keepdims=True)
    cos = np.einsum("nid,njd->nij", v, v)
    iu = np.triu_indices(4, 1)
    qtet = 1 - 3 / 8 * ((cos[:, iu[0], iu[1]] + 1 / 3) ** 2).sum(axis=1)
    n_dens = (dist < dens_cut).sum(axis=1)
    return np.column_stack([d5, lsi, qtet, n_dens]).astype(np.float32)


simple_feats = {}
for f in np.unique(soap_frame_of_row):
    x = simple_descriptors(averaged_atoms(int(f), time_avg_half_width))
    simple_feats[int(f)] = x - x.mean(axis=0)              # same per-frame centering

for H in [10, max(results_H)]:
    T = cumulative_target(H)
    lab_H = np.array([per_frame_labels(r) for r in T])
    tr, te = block_split(len(T), gap=max(H, dp))
    for name, feat in [("SOAP", features), ("d5/LSI/qtet/n", simple_feats)]:
        Xa, ya, _ = stack_target(tr, lab_H, feat)
        Xb, yb, _ = stack_target(te, lab_H, feat)
        m = fit_balanced(make_hgb(), Xa, ya)
        print(f"H = {H:3d}  {name:14s} test AUC = {roc_auc_score(yb, m.predict_proba(Xb)[:, 1]):.3f}")

# %%
# single-descriptor AUCs (sign-free), pooled over test frames, H = 10
T = cumulative_target(10)
lab_H = np.array([per_frame_labels(r) for r in T])
_, te = block_split(len(T), gap=dp)
Xb, yb, _ = stack_target(te, lab_H, simple_feats)
for j, nm in enumerate(["d5", "LSI", "q_tet", "n(r<3.5)"]):
    a = roc_auc_score(yb, Xb[:, j])
    print(f"{nm:9s} AUC = {max(a, 1 - a):.3f}  ({'high' if a > 0.5 else 'low'} value -> mobile)")

# %% [markdown]
# ## Round 3: same frames for every target, and less noisy targets
#
# Notes on the round-2 output:
# * The H comparison above is NOT fair: with the frame-block split each H was tested on
#   different frames (H=50 only on frames 390-429, H=80 trained on 280-319 and tested on
#   480-519). Below, train and test are separated in SPACE instead (two slabs of the box
#   with a ~9 A buffer), so every frame is used and every target is scored on the same frames.
# * Neighbouring D2min windows share one frame ([f, f+10] and [f+10, f+20] both use f+10),
#   so thermal noise in that frame inflates the 0.216 persistence. Windows 20 frames apart
#   give 0.056. Likewise SOAP(f) and D2min[f, f+10] share frame f. So every target below
#   starts at f + dp: nothing is shared between the input structure and the label.
# * Targets compared:
#     1. D2min[f+dp, f+2dp]                       (single window)
#     2. cumulative D2min over H after f+dp        (time-averaged)
#     3. coarse-grained D2min (averaged over the 3.5 A neighbourhood)  (space-averaged)
#     4. LDA -> HDA conversion: an LDA-like molecule at f (large d5) becomes HDA-like (small d5)
#        by f+H. This is closest to "will this region change", and it is irreversible,
#        so it should be much less noisy than D2min.

# %%
# ---------------- per-frame geometry: slab coordinate, raw d5, neighbour pairs ----------------
from scipy.spatial import cKDTree

frac_x, d5_raw, nbr_pairs = {}, {}, {}
for f in np.unique(soap_frame_of_row):
    at = averaged_atoms(int(f), time_avg_half_width)
    L = at.cell.lengths()[0]
    pos = at.positions % L
    pos[pos >= L] = 0.0
    tree = cKDTree(pos, boxsize=L)
    dist, _ = tree.query(pos, k=6)
    d5_raw[int(f)] = dist[:, 5].astype(np.float32)          # column 0 is the molecule itself
    frac_x[int(f)] = pos[:, 0] / L
    nbr_pairs[int(f)] = tree.query_pairs(coarse_grain_shell, output_type="ndarray")

for f in [250, 400, 550, 650]:
    if f in d5_raw:
        q = np.percentile(d5_raw[f], [10, 50, 90])
        print(f"frame {f}: d5 10/50/90 percentiles = {q.round(2)}")

# %%
# ---------------- spatial-split evaluation ----------------
TRAIN_SLAB = (0.05, 0.35)     # fractional x; the gaps 0.35-0.55 and 0.85-1.05 are buffers
TEST_SLAB = (0.55, 0.85)


def cg_row(vals, pairs):
    s = vals.astype(float).copy()
    c = np.ones_like(s)
    np.add.at(s, pairs[:, 0], vals[pairs[:, 1]])
    np.add.at(s, pairs[:, 1], vals[pairs[:, 0]])
    np.add.at(c, pairs[:, 0], 1)
    np.add.at(c, pairs[:, 1], 1)
    return s / c


def eval_target(get_label, name, stride=2, show=True):
    """get_label(k) -> labels (-1/0/1) for all molecules, or None if undefined for row k."""
    Xa, ya, Xb, yb, fb = [], [], [], [], []
    for n_, k in enumerate(range(0, n_rows, stride)):
        lab = get_label(k)
        if lab is None:
            continue
        f = int(soap_frame_of_row[k])
        x = frac_x[f]
        tr = (lab != -1) & (x >= TRAIN_SLAB[0]) & (x < TRAIN_SLAB[1])
        te = (lab != -1) & (x >= TEST_SLAB[0]) & (x < TEST_SLAB[1])
        if n_ % 2 == 0:
            Xa.append(features[f][tr]); ya.append(lab[tr])
        Xb.append(features[f][te]); yb.append(lab[te]); fb.append(np.full(te.sum(), k))
    ya_all = np.concatenate(ya) if ya else np.array([])
    if len(np.unique(ya_all)) < 2:
        print(f"{name:38s} not enough labelled molecules of both classes")
        return None
    m = fit_balanced(make_hgb(), np.vstack(Xa), ya_all)
    Xb, yb, fb = np.vstack(Xb), np.concatenate(yb), np.concatenate(fb)
    p = m.predict_proba(Xb)[:, 1]
    per = {}
    for k in np.unique(fb):
        s = fb == k
        if 0 < yb[s].sum() < s.sum():
            per[k] = roc_auc_score(yb[s], p[s])
    mean_auc = np.mean(list(per.values()))
    print(f"{name:38s} mean per-frame AUC = {mean_auc:.3f}   "
          f"(frames {len(per)}, test molecules {len(yb)}, positives {yb.mean():.2f})")
    if show:
        ks = np.array(sorted(per))
        plt.plot(row_frames[ks], [per[k] for k in ks], "o", ms=3, label=name)
    return per


def label_rows(rowvals):
    return per_frame_labels(rowvals)


plt.figure(figsize=(10, 4.5))
res = {}

# 1. single window, starting at f + dp
res["D2min [f+dp, f+2dp]"] = eval_target(
    lambda k: label_rows(d2min[k + dp]) if k + dp < n_rows else None, "D2min [f+dp, f+2dp]")

# 2. cumulative over H after f + dp
for H in [30, 50]:
    T = cumulative_target(H)
    res[f"cumulative H={H}"] = eval_target(
        lambda k, T=T: label_rows(T[k + dp]) if k + dp < len(T) else None, f"cumulative D2min H={H}")

# 3. coarse-grained single window and coarse-grained H=50
res["CG D2min [f+dp, f+2dp]"] = eval_target(
    lambda k: label_rows(cg_row(d2min[k + dp], nbr_pairs[int(row_frames[k + dp])]))
    if k + dp < n_rows else None, "CG D2min [f+dp, f+2dp]")
T50 = cumulative_target(50)
res["CG cumulative H=50"] = eval_target(
    lambda k: label_rows(cg_row(T50[k + dp], nbr_pairs[int(row_frames[k + dp])]))
    if k + dp < len(T50) else None, "CG cumulative D2min H=50")

plt.axhline(0.5, color="gray", ls="--")
plt.xlabel("frame f (structure)")
plt.ylabel("per-frame AUC (test slab)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %%
# persistence of each target (windows that share no frame): rank correlation between
# the target starting at f and the same target starting at f + H + dp
def persistence(rows_fn, gap):
    vals = [spearmanr(rows_fn(k), rows_fn(k + gap))[0] for k in range(0, n_rows - gap - 60, 5)]
    return np.mean(vals)

print("raw D2min, 20 frames apart          :", round(persistence(lambda k: d2min[k], 2 * dp), 3))
print("CG  D2min, 20 frames apart          :", round(persistence(
    lambda k: cg_row(d2min[k], nbr_pairs[int(row_frames[k])]), 2 * dp), 3))
print("cumulative H=50, 60 frames apart    :", round(persistence(lambda k: T50[k], 60), 3))
print("CG cumulative H=50, 60 frames apart :", round(persistence(
    lambda k: cg_row(T50[k], nbr_pairs[int(row_frames[k])]), 60), 3))

# %%
# ---------------- 4. LDA -> HDA conversion target ----------------
# Eligible: LDA-like at f (d5 > d5_lda). Positive: HDA-like at f+H (d5 < d5_hda).
# Check the d5 percentiles printed above and adjust the two thresholds if needed
# (TIP4P/2005: LDA d5 roughly 3.5-4 A, HDA d5 roughly 3.0-3.3 A).
d5_lda, d5_hda = 3.5, 3.3

plt.figure(figsize=(10, 4.5))
for H in [20, 50]:
    def conv_label(k, H=H):
        f, f2 = int(soap_frame_of_row[k]), int(soap_frame_of_row[k]) + H
        if f2 not in d5_raw:
            return None
        lab = np.full(n_particles, -1)
        elig = d5_raw[f] > d5_lda
        lab[elig] = (d5_raw[f2][elig] < d5_hda).astype(int)
        return lab
    res[f"conversion H={H}"] = eval_target(conv_label, f"LDA->HDA conversion within H={H}")

    # baseline: d5(f) alone on the same test molecules (larger d5 = further from HDA)
    aucs = []
    for k in range(0, n_rows, 2):
        lab = conv_label(k)
        if lab is None:
            continue
        f = int(soap_frame_of_row[k])
        x = frac_x[f]
        s = (lab != -1) & (x >= TEST_SLAB[0]) & (x < TEST_SLAB[1])
        if 0 < lab[s].sum() < s.sum():
            aucs.append(roc_auc_score(lab[s], -d5_raw[f][s]))
    if aucs:
        print(f"{'   baseline: d5(f) alone':38s} mean per-frame AUC = {np.mean(aucs):.3f}")
plt.axhline(0.5, color="gray", ls="--")
plt.xlabel("frame f (structure)")
plt.ylabel("per-frame AUC (test slab)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()
