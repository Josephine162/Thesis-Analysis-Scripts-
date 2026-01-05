#Group-level topographies + cluster stats for ASMR vs Control (FOOOF peak amplitudes).

# Outputs in: <AllASMRvControl_FOOOFout>/GroupFigures/
"""

import os
import pickle
import numpy as np
import mne
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_1samp_test
from mne.channels import find_ch_adjacency

# ----------------- paths -----------------
DATA_DIR = "/Users/jrf521/Documents/EEGdata/ASMRconverted/AllASMRvControl_FOOOFout"
IN_PKL   = os.path.join(DATA_DIR, "per_subject_fooof_peaks_ASMR_allTrials.pkl")
OUT_DIR  = os.path.join(DATA_DIR, "GroupFigures")
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------- params ----------------
N_PERM    = 2000
ALPHA     = 0.05
TAIL      = 0
THRESHOLD = None  # distribution-based threshold

FIGSIZE   = (7, 6)
DPI       = 300

# ----------------- load ------------------
with open(IN_PKL, "rb") as f:
    D = pickle.load(f)

subjects       = D["subjects"]
band_names     = D["band_names"]
ch_names       = D["ch_names"]
info           = D["info"]
peakAmpASMR    = D["peakAmpASMR"]      # (n_subj, n_bands, n_ch)
peakAmpControl = D["peakAmpControl"]   # (n_subj, n_bands, n_ch)

assert peakAmpASMR.shape == peakAmpControl.shape, "ASMR/Control shapes differ!"
n_subj, n_bands, n_ch = peakAmpASMR.shape
print(f"Loaded: n_subj={n_subj}, n_bands={n_bands}, n_ch={n_ch}, bands={band_names}")

# ------------- differences ---------------
diff_amp = peakAmpASMR - peakAmpControl   # (n_subj, n_bands, n_ch)

# ------------- adjacency -----------------
if info is None:
    raise RuntimeError("No MNE 'info' saved in pickle; cannot derive EEG adjacency.")
adjacency, _ = find_ch_adjacency(info, ch_type='eeg')
print(f"Adjacency shape: {adjacency.shape}")

# ------------- helpers -------------------
def symmetric_limits(x, min_floor=1e-12):
    """Return symmetric (vmin, vmax) around zero using max abs value."""
    amax = float(np.nanmax(np.abs(x))) if np.isfinite(x).any() else 1.0
    amax = max(amax, min_floor)
    return (-amax, amax)

def save_topomap(data_1d, title, fname, cmap="RdBu_r", symmetric=True):
    """Plot a topomap from a 1D channel vector on an explicit Axes (no tiny figures)."""
    data_1d = np.asarray(data_1d, dtype=float)
    if symmetric:
        vmin, vmax = symmetric_limits(data_1d)
    else:
        vmin = float(np.nanmin(data_1d))
        vmax = float(np.nanmax(data_1d))

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
        # Newer MNE fallback
        im, _ = mne.viz.plot_topomap(
            data_1d, pos=info, ch_type="eeg",
            axes=ax, show=False, cmap=cmap,
            outlines="head", sensors=True, contours=0,
            vmin=vmin, vmax=vmax
        )

    ax.set_title(title)
    # Attach colorbar to the same fig/ax; no tight_layout to avoid engine clash
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Value")

    fig.savefig(os.path.join(OUT_DIR, fname), dpi=DPI, bbox_inches="tight")
    plt.close(fig)

def channels_from_mask(mask):
    """mask shape (n_ch, 1) -> list of channel indices in cluster"""
    return np.where(mask[:, 0])[0]

# ------------- run per band --------------
summary_lines = []
for b_idx, band in enumerate(band_names):
    print(f"\n=== Band: {band} ===")
    X = diff_amp[:, b_idx, :]               # (n_subj, n_ch)
    mean_diff = np.nanmean(X, axis=0)       # (n_ch,)

    # Unthresholded mean difference topo
    save_topomap(
        mean_diff,
        title=f"ASMR - Control (mean diff) — {band}",
        fname=f"mean_diff_{band}.png",
        cmap="RdBu_r",
        symmetric=True
    )

    # Cluster test expects (n_obs, n_ch, n_times)
    X3 = X[:, :, np.newaxis]

    # Compatible call for your MNE version (no random_state / n_jobs)
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
    n_clusters = len(clusters)
    n_sig = len(sig_idx)
    print(f"Clusters found: {n_clusters} | significant: {n_sig} (alpha={ALPHA})")

    sig_mask_channels = np.zeros(n_ch, dtype=bool)
    lines = [f"[{band}] clusters={n_clusters}, significant={n_sig}"]
    for k in sig_idx:
        ch_inds = channels_from_mask(clusters[k])
        sig_mask_channels[ch_inds] = True
        ch_list = [ch_names[i] for i in ch_inds]
        lines.append(f"  - cluster #{k}  p={p_vals[k]:.4f}  n_ch={len(ch_inds)}  chans={ch_list}")

    # T-map of significant clusters only
    tmap_sig = np.zeros_like(T_ch)
    tmap_sig[sig_mask_channels] = T_ch[sig_mask_channels]
    save_topomap(
        tmap_sig,
        title=f"T-values (sig clusters) — {band}",
        fname=f"tmap_sigclusters_{band}.png",
        cmap="RdBu_r",
        symmetric=True
    )

    # Mean difference masked by significance
    diff_masked = np.zeros_like(mean_diff)
    diff_masked[sig_mask_channels] = mean_diff[sig_mask_channels]
    save_topomap(
        diff_masked,
        title=f"ASMR - Control (mean diff, sig-only) — {band}",
        fname=f"sigmask_diff_{band}.png",
        cmap="RdBu_r",
        symmetric=True
    )

    summary_lines.extend(lines)

# ------------- write summary -------------
summary_path = os.path.join(OUT_DIR, "cluster_summary.txt")
with open(summary_path, "w") as f:
    f.write("Cluster-based permutation summary (ASMR - Control FOOOF peak amplitudes)\n")
    f.write(f"subjects (n={len(subjects)}): {subjects}\n")
    f.write(f"alpha={ALPHA}, permutations={N_PERM}, tail={TAIL}, threshold={THRESHOLD}\n")
    f.write("\n".join(summary_lines))
print(f"\nSaved figures + cluster summary to: {OUT_DIR}")
