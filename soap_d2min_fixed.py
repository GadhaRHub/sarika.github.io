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
models = {
    "LogReg (PCA 30)": make_pipeline(StandardScaler(), PCA(30),
                                     LogisticRegression(max_iter=2000, class_weight="balanced")),
    "HGB": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                          class_weight="balanced", random_state=0),
}
proba = {}
for name, m in models.items():
    m.fit(X_tr, y_tr)
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
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                       class_weight="balanced", random_state=0).fit(Xa, ya)
    print(f"lag {lag:3d} frames: test AUC = {roc_auc_score(yb, m.predict_proba(Xb)[:, 1]):.3f}")

# %%
# ---------------- sanity check: is D2min itself persistent? ----------------
# Upper bound on what any structural descriptor can do: how well does the PAST
# window's D2min rank the NEXT window's D2min for the same particle?
for lag in [dp, 2 * dp, 5 * dp]:
    rho = np.mean([spearmanr(d2min[k], d2min[k + lag])[0]
                   for k in range(0, n_rows - lag, 5)])
    print(f"Spearman(D2min[f], D2min[f+{lag}]) averaged over frames = {rho:.3f}")
