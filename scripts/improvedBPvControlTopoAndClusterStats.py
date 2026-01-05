# Button-press ASMR (any press) vs Control: group topomaps + cluster stats
# Compatible with older MNE plot_topomap API (vlim tuple), avoids layout issues.
# Exports figures and a CSV of per-subject cluster metrics
"""

import os
import pickle
import numpy as np
import mne
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_1samp_test
from mne.channels import find_ch_adjacency
import csv

# ---------- PATHS (edit these if needed) ----------
BASE = "/Users/jrf521/Documents/EEGdata/ASMRconverted"
BP_PKL = os.path.join(BASE, "BPgroupASMRout_anyPress", "per_subject_fooof_peaks_ASMR_BP_anyPress.pkl")
CTRL_PKL = os.path.join(BASE, "AllASMRvControl_FOOOFout", "per_subject_fooof_peaks_ASMR_allTrials.pkl")
OUT_DIR = os.path.join(BASE, "BPasmrVsControlStats_anyPress_FOOOFup")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- PARAMS ----------
N_PERM    = 2000
ALPHA     = 0.05
TAIL      = 0
THRESHOLD = None
FIGSIZE   = (7, 6)
DPI       = 300

# ---------- LOAD ----------
def load_pickle(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        return pickle.load(f)

BP = load_pickle(BP_PKL)       # expected keys: peakAmpASMR_BP, band_names, info, subjects, ch_names
CT = load_pickle(CTRL_PKL)     # expected keys: peakAmpControl, band_names, info, subjects, ch_names

# Pull arrays
amp_bp    = BP["peakAmpASMR_BP"]     # (n_subj_bp, n_bands, n_ch)
amp_ctrl  = CT["peakAmpControl"]     # (n_subj_ctrl, n_bands, n_ch)
bands_bp  = BP["band_names"]
bands_ct  = CT["band_names"]
info_bp   = BP.get("info", None)
info_ct   = CT.get("info", None)
chs_bp    = BP.get("ch_names", None)
chs_ct    = CT.get("ch_names", None)
subs_bp   = BP.get("subjects", None)
subs_ct   = CT.get("subjects", None)

# ---------- SANITY ----------
if bands_bp != bands_ct:
    raise ValueError("Band name mismatch between BP and Control pickles.")
band_names = bands_bp
n_bands = len(band_names)

if amp_bp.shape[2] != amp_ctrl.shape[2]:
    # if channel counts differ, we’ll try to reindex by names
    if (chs_bp is None) or (chs_ct is None):
        raise ValueError("Channel counts differ and ch_names not available for reindex.")
    # reindex ctrl to bp order (or vice versa)
    idx_map = [chs_ct.index(ch) for ch in chs_bp]
    amp_ctrl = amp_ctrl[:, :, idx_map]
    chs_ct_reindexed = [chs_ct[i] for i in idx_map]
    assert chs_ct_reindexed == chs_bp, "Failed to align channel orders."

# Choose info for plotting/adjacency
info = info_bp if info_bp is not None else info_ct
if info is None:
    raise RuntimeError("No MNE Info in pickles; cannot compute adjacency/topos.")
adjacency, _ = find_ch_adjacency(info, ch_type="eeg")
print(f"Adjacency: {adjacency.shape}")

# ---------- ALIGN SUBJECTS BY ID ----------
if (subs_bp is not None) and (subs_ct is not None):
    common = [s for s in subs_bp if s in subs_ct]
    if len(common) == 0:
        raise ValueError("No overlapping subjects between BP and Control pickles.")
    # index arrays in the order of 'common'
    idx_bp = [subs_bp.index(s) for s in common]
    idx_ct = [subs_ct.index(s) for s in common]
    amp_bp = amp_bp[idx_bp, :, :]
    amp_ctrl = amp_ctrl[idx_ct, :, :]
    subjects = common
else:
    # fall back to min length with a warning
    n = min(amp_bp.shape[0], amp_ctrl.shape[0])
    subjects = [f"S{i+1}" for i in range(n)]
    amp_bp = amp_bp[:n]
    amp_ctrl = amp_ctrl[:n]
    print("WARNING: subjects not stored; truncated to min(n_bp, n_ctrl).")

n_subj, _, n_ch = amp_bp.shape
print(f"Aligned: n_subj={n_subj}, n_bands={n_bands}, n_ch={n_ch}, bands={band_names}")

# ---------- DIFF: BP – Control ----------
diff_amp = amp_bp - amp_ctrl   # (n_subj, n_bands, n_ch)

# ---------- HELPERS ----------
def symmetric_limits(x, min_floor=1e-12):
    amax = float(np.nanmax(np.abs(x))) if np.isfinite(x).any() else 1.0
    amax = max(amax, min_floor)
    return (-amax, amax)

def save_topomap(data_1d, title, fname, cmap="RdBu_r", symmetric=True):
    data_1d = np.asarray(data_1d, dtype=float)
    vmin, vmax = symmetric_limits(data_1d) if symmetric else (float(np.nanmin(data_1d)), float(np.nanmax(data_1d)))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    # Old MNE prefers vlim; try that first
    try:
        im, _ = mne.viz.plot_topomap(
            data_1d, pos=info, ch_type="eeg",
            axes=ax, show=False, cmap=cmap,
            outlines="head", sensors=True, contours=0,
            vlim=(vmin, vmax)
        )
    except TypeError:
        im, _ = mne.viz.plot_topomap(
            data_1d, pos=info, ch_type="eeg",
            axes=ax, show=False, cmap=cmap,
            outlines="head", sensors=True, contours=0,
            vmin=vmin, vmax=vmax
        )
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Value")
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=DPI, bbox_inches="tight")
    plt.close(fig)

def chans_from_mask(mask):
    return np.where(mask[:, 0])[0]

# ---------- RUN PER BAND ----------
summary_lines = []
sig_clusters_by_band = {}

for bi, band in enumerate(band_names):
    print(f"\n=== Band: {band} ===")
    X = diff_amp[:, bi, :]                 # (n_subj, n_ch)
    mean_diff = np.nanmean(X, axis=0)      # (n_ch,)
    save_topomap(mean_diff, f"ASMR-BP - Control (mean diff) — {band}", f"bp_mean_diff_{band}.png")

    # cluster test needs (n_obs, n_ch, n_times=1)
    X3 = X[:, :, np.newaxis]
    T_obs, clusters, p_vals, H0 = permutation_cluster_1samp_test(
        X3,
        n_permutations=N_PERM,
        adjacency=adjacency,
        tail=TAIL,
        threshold=THRESHOLD,
        out_type='mask',
        verbose=False
    )
    T_ch = T_obs[:, 0]
    sig_idx = np.where(p_vals < ALPHA)[0]
    print(f"Clusters found: {len(clusters)} | significant: {len(sig_idx)}")
    summary_lines.append(f"[{band}] clusters={len(clusters)}, significant={len(sig_idx)}")

    # save sig t-map + sig-masked mean
    sig_mask = np.zeros(n_ch, dtype=bool)
    band_clusters = []
    for k in sig_idx:
        ch_inds = chans_from_mask(clusters[k])
        sig_mask[ch_inds] = True
        band_clusters.append({"p": float(p_vals[k]), "channels": ch_inds.tolist()})
        ch_list = [ (info.ch_names[i] if hasattr(info, "ch_names") else str(i)) for i in ch_inds ]
        summary_lines.append(f"  - cluster #{k}  p={p_vals[k]:.4f}  n_ch={len(ch_inds)}  chans={ch_list}")

    sig_clusters_by_band[band] = band_clusters

    t_sig = np.zeros_like(T_ch); t_sig[sig_mask] = T_ch[sig_mask]
    save_topomap(t_sig, f"T-values (sig clusters) — {band}", f"bp_tmap_sigclusters_{band}.png")

    diff_masked = np.zeros_like(mean_diff); diff_masked[sig_mask] = mean_diff[sig_mask]
    save_topomap(diff_masked, f"ASMR-BP - Control (mean diff, sig-only) — {band}", f"bp_sigmask_diff_{band}.png")

# ---------- EXPORT per-subject metrics for β band (and any other sig band) ----------
csv_path = os.path.join(OUT_DIR, "bp_cluster_subject_metrics.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    # header
    header = ["subject_id"]
    for band in band_names:
        header += [f"{band}_cluster_mean_diff", f"{band}_cluster_mean_BP", f"{band}_cluster_mean_CTRL", f"{band}_nch_cluster", f"{band}_pmin"]
    w.writerow(header)

    # build indices for each band (union of all sig clusters in that band)
    band_sig_masks = {}
    for band in band_names:
        clusters = sig_clusters_by_band.get(band, [])
        if not clusters:
            band_sig_masks[band] = None
        else:
            mask = np.zeros(n_ch, dtype=bool)
            pmin = 1.0
            for c in clusters:
                mask[np.array(c["channels"], dtype=int)] = True
                pmin = min(pmin, c["p"])
            band_sig_masks[band] = (mask, pmin, int(mask.sum()))

    # per subject rows
    for si, sid in enumerate(subjects):
        row = [sid]
        for bi, band in enumerate(band_names):
            pack = band_sig_masks[band]
            if pack is None:
                row += [np.nan, np.nan, np.nan, 0, np.nan]
            else:
                mask, pmin, nch = pack
                bp_mean   = float(np.nanmean(amp_bp[si, bi, mask])) if nch > 0 else np.nan
                ctrl_mean = float(np.nanmean(amp_ctrl[si, bi, mask])) if nch > 0 else np.nan
                row += [bp_mean - ctrl_mean, bp_mean, ctrl_mean, nch, pmin]
        w.writerow(row)

print(f"\nSaved figures + cluster summary + CSV to: {OUT_DIR}")

# write a human-readable summary too
with open(os.path.join(OUT_DIR, "bp_cluster_summary.txt"), "w") as f:
    f.write("Button-press ASMR vs Control (FOOOF peak amplitudes)\n")
    f.write(f"subjects (n={len(subjects)}): {subjects}\n")
    f.write(f"alpha={ALPHA}, permutations={N_PERM}, tail={TAIL}, threshold={THRESHOLD}\n")
    f.write("\n".join(summary_lines))
