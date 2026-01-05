# EEG Morlet TFR: ASMR vs Control (grouped event IDs with single-code relabel)
# - Groups ASMR codes (1..13) and Control codes (14..18) even if some specific codes are missing.
# - Fixes: after selecting events, rewrites event codes to a single value (1),
#   so event_id={"cond": 1} is valid on older MNE builds.
# - Baseline-corrects each epoch (log-ratio to -2..0 s), averages per condition.
# - Saves per-subject TFR arrays + group summaries and per-band topo maps.
# - Optional cluster-based permutation over sensors on per-band topomaps.

Outputs:
  DATA_DIR/
    TFR_ASMRvCTRL_out/
      per_subject_tfr.pkl
      figs/
        grandmean_TFR_ASMR.png
        grandmean_TFR_CTRL.png
        grandmean_TFR_DIFF.png
        topo_DIFF_theta.png
        topo_DIFF_alpha.png
        topo_DIFF_beta.png
        topo_DIFF_gamma.png
        (optional) topo_DIFF_<band>_CLUSTER.png
"""

import os, sys, pickle, warnings
import numpy as np
import mne
import matplotlib.pyplot as plt
from mne.channels import find_ch_adjacency
from mne.stats import permutation_cluster_1samp_test

# ----------------------------
# 0) USER SETTINGS
# ----------------------------
DATA_DIR   = "/Users/jrf521/Documents/EEGdata/ASMRconverted"  # folder with P##_EEG.set
OUT_DIR    = os.path.join(DATA_DIR, "TFR_ASMRvCTRL_out")
FIG_DIR    = os.path.join(OUT_DIR, "figs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SUBJECTS   = [f"P{i}" for i in range(1, 65)]   # P1..P64

# ASMR and Control numeric codes in EEGLAB annotations
ASMR_CODES  = tuple(range(1, 14))      # 1..13
CTRL_CODES  = tuple(range(14, 19))     # 14..18

# Epoching
EPOCH_TMIN  = -2.0
EPOCH_TMAX  = 30.0
BASELINE    = (-2.0, 0.0)

# Preprocessing
RESAMPLE_HZ = 250
HPF, LPF    = 0.5, 100.0
NOTCHS      = [50, 100]

# TFR params
FREQS       = np.linspace(4, 48, 30)           # 30 freqs from 4..48 Hz
N_CYCLES    = np.linspace(3, 12, len(FREQS))   # more cycles at higher freq
DECIM       = 2                                # decimate time points

# Bands for topo summaries
BANDS = {
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta":  (15, 30),
    "gamma": (30, 48)
}

# Topomap time window
TOPO_TMIN, TOPO_TMAX = 2.0, 25.0

# Optional cluster stats on topo maps
DO_CLUSTER_STATS = True
N_PERM           = 1000
ALPHA            = 0.05

# ----------------------------
# Helpers
# ----------------------------
def load_preprocess_raw(eeg_set_path):
    """Load EEGLAB .set, drop EOG/Mastoids if present, set montage + ref, filter, resample."""
    raw = mne.io.read_raw_eeglab(eeg_set_path, preload=True)

    drop_try = ["HEOG", "VEOG", "M1", "M2"]
    raw.drop_channels([ch for ch in drop_try if ch in raw.ch_names])

    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore")
    raw.set_eeg_reference("average")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        raw.filter(HPF, LPF)
        if NOTCHS:
            raw.notch_filter(freqs=NOTCHS)
    if RESAMPLE_HZ:
        raw.resample(RESAMPLE_HZ)

    return raw

def get_events(raw):
    """Return MNE events array (n_events x 3)."""
    events, _ = mne.events_from_annotations(raw)
    return events

def count_codes(events):
    """Quick dict of counts per event code."""
    vals, cnts = np.unique(events[:, 2], return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, cnts)}

def compute_tfr_avg_grouped(raw, events, code_group):
    """
    Epoch, compute epoch-level TFR → baseline → average for a group of codes.
    This version rewrites the selected events' ids to a single code (1) to satisfy
    older MNE versions that require event_id[str] -> int (not list).
    Returns AverageTFR or None.
    """
    mask = np.isin(events[:, 2], code_group)
    ev_sel = events[mask]
    if len(ev_sel) == 0:
        return None

    ev_one = ev_sel.copy()
    ev_one[:, 2] = 1  # rewrite all selected to the same code
    event_id_sel = {"cond": 1}

    epochs = mne.Epochs(
        raw, ev_one, event_id=event_id_sel,
        tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
        baseline=BASELINE,
        reject=None, preload=True, picks="eeg", detrend=None
    )
    if len(epochs) == 0:
        return None

    power = mne.time_frequency.tfr_morlet(
        epochs, freqs=FREQS, n_cycles=N_CYCLES,
        use_fft=True, return_itc=False, decim=DECIM,
        average=False, picks="eeg", n_jobs=1
    )
    power.apply_baseline(mode="logratio", baseline=BASELINE)
    return power.average()  # AverageTFR

def reindex_to_template(data, from_names, to_names):
    """Reorder channel axis of (n_ch, n_f, n_t) to match template order."""
    if from_names == to_names:
        return data
    idx = [from_names.index(ch) for ch in to_names]
    return data[idx, :, :]

# ----------------------------
# 1) Main loop
# ----------------------------
per_subject = []
template_chs = None
template_info = None
kept_subjects = []
times = None
freqs = None

for subj in SUBJECTS:
    eeg_path = os.path.join(DATA_DIR, f"{subj}_EEG.set")
    if not os.path.exists(eeg_path):
        print(f"[WARN] Missing: {eeg_path} → skip {subj}.")
        continue

    try:
        print(f"\n--- {subj} ---")
        raw = load_preprocess_raw(eeg_path)
        events = get_events(raw)
        # Debug: print a compact code summary
        csum = count_codes(events)
        seen = sorted(csum.keys())
        print(f"  Event codes present: {seen}  (counts: e.g., 1→{csum.get(1,0)}, 14→{csum.get(14,0)})")

        tfr_asmr = compute_tfr_avg_grouped(raw, events, ASMR_CODES)
        tfr_ctrl = compute_tfr_avg_grouped(raw, events, CTRL_CODES)
        if (tfr_asmr is None) or (tfr_ctrl is None):
            print("  [WARN] Missing ASMR or Control TFR → skip subject.")
            continue

        # Init template on first subject
        if template_chs is None:
            template_chs  = tfr_asmr.ch_names
            template_info = tfr_asmr.info
            times = tfr_asmr.times
            freqs = tfr_asmr.freqs

        # Reindex to template if needed
        if tfr_asmr.ch_names != template_chs:
            data_a = reindex_to_template(tfr_asmr.data, tfr_asmr.ch_names, template_chs)
            data_c = reindex_to_template(tfr_ctrl.data, tfr_ctrl.ch_names, template_chs)
        else:
            data_a = tfr_asmr.data
            data_c = tfr_ctrl.data

        per_subject.append({"subject": subj, "asmr": data_a, "ctrl": data_c})
        kept_subjects.append(subj)
        print(f"  Kept {subj}: TFR OK (ASMR={data_a.shape}, CTRL={data_c.shape}).")

    except Exception as e:
        print(f"[ERROR] {subj}: {e}")
        continue

if len(per_subject) == 0:
    print("No valid subjects processed. Exiting.")
    sys.exit(1)

# Save per-subject TFRs
save_dict = {
    "subjects": kept_subjects,
    "ch_names": template_chs,
    "info":     template_info,
    "freqs":    freqs,
    "times":    times,
    "asmr":     np.stack([d["asmr"] for d in per_subject], axis=0),  # (n_subj, n_ch, n_f, n_t)
    "ctrl":     np.stack([d["ctrl"] for d in per_subject], axis=0),
}
with open(os.path.join(OUT_DIR, "per_subject_tfr.pkl"), "wb") as f:
    pickle.dump(save_dict, f)
print(f"\nSaved per-subject TFR → {os.path.join(OUT_DIR, 'per_subject_tfr.pkl')}")
print(f"Subjects kept (n={len(kept_subjects)}): {kept_subjects[:10]}{'...' if len(kept_subjects)>10 else ''}")

# ----------------------------
# 2) Group summaries (figures)
# ----------------------------
asmr = save_dict["asmr"]  # (n_subj, n_ch, n_f, n_t)
ctrl = save_dict["ctrl"]
freqs = save_dict["freqs"]
times = save_dict["times"]
info  = save_dict["info"]
ch_names = save_dict["ch_names"]

# Grand means (sensor-avg)
asmr_mean = np.nanmean(asmr, axis=(0, 1))  # (n_f, n_t)
ctrl_mean = np.nanmean(ctrl, axis=(0, 1))
diff_mean = asmr_mean - ctrl_mean

def _plot_tfr(mat, title, fname):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    im = ax.imshow(
        mat, aspect="auto", origin="lower",
        extent=[times[0], times[-1], freqs[0], freqs[-1]]
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, label="Baseline-corrected power (log-ratio)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=200)
    plt.close(fig)

_plot_tfr(asmr_mean, "Grand-mean TFR (ASMR) — sensor average", "grandmean_TFR_ASMR.png")
_plot_tfr(ctrl_mean, "Grand-mean TFR (Control) — sensor average", "grandmean_TFR_CTRL.png")
_plot_tfr(diff_mean, "Grand-mean TFR (ASMR − Control) — sensor average", "grandmean_TFR_DIFF.png")

# Per-band topomaps of ASMR−Control averaged over TOPO_TMIN..TOPO_TMAX
tmask = (times >= TOPO_TMIN) & (times <= TOPO_TMAX)
Xdiff = np.nanmean(asmr - ctrl, axis=0)  # (n_ch, n_f, n_t) grand mean across subj
for band, (fmin, fmax) in BANDS.items():
    fmask = (freqs >= fmin) & (freqs <= fmax)
    topo_vals = np.nanmean(Xdiff[:, fmask, :][:, :, tmask], axis=(1, 2))  # (n_ch,)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im, _ = mne.viz.plot_topomap(topo_vals, pos=info, ch_type="eeg", axes=ax, show=False, contours=0)
    fig.colorbar(im, ax=ax, label="ASMR−Control (log-ratio)")
    ax.set_title(f"Topomap: {band} ({fmin}-{fmax} Hz), {TOPO_TMIN}–{TOPO_TMAX}s")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"topo_DIFF_{band}.png"), dpi=200)
    plt.close(fig)

# ----------------------------
# 3) Optional cluster stats over sensors on per-band topo maps
# ----------------------------
if DO_CLUSTER_STATS:
    try:
        adjacency, _ = find_ch_adjacency(info, ch_type="eeg")
        for band, (fmin, fmax) in BANDS.items():
            fmask = (freqs >= fmin) & (freqs <= fmax)
            subj_maps = np.nanmean(asmr[:, :, fmask, :][:, :, :, tmask] - ctrl[:, :, fmask, :][:, :, :, tmask],
                                   axis=(2, 3))  # (n_subj, n_ch)

            X = subj_maps[:, :, np.newaxis]  # (n_subj, n_ch, 1)
            T_obs, clusters, p_vals, _ = permutation_cluster_1samp_test(
                X, adjacency=adjacency, n_permutations=N_PERM,
                tail=0, threshold=None, out_type="mask", verbose=False
            )
            T1 = T_obs[:, 0]

            sig_mask = np.zeros(len(ch_names), dtype=bool)
            n_sig = 0
            for ci, cmask in enumerate(clusters):
                if p_vals[ci] < ALPHA:
                    n_sig += 1
                    sig_mask |= cmask[:, 0].astype(bool)
            print(f"[cluster] {band}: {len(clusters)} clusters, {n_sig} significant (alpha={ALPHA}).")

            tmap = np.zeros_like(T1)
            tmap[sig_mask] = T1[sig_mask]
            fig, ax = plt.subplots(figsize=(5, 4.5))
            im, _ = mne.viz.plot_topomap(tmap, pos=info, ch_type="eeg", axes=ax, show=False, contours=0)
            fig.colorbar(im, ax=ax, label="t-value (sig clusters)")
            ax.set_title(f"Cluster-masked t-map: {band} ({fmin}-{fmax} Hz), {TOPO_TMIN}–{TOPO_TMAX}s")
            fig.tight_layout()
            fig.savefig(os.path.join(FIG_DIR, f"topo_DIFF_{band}_CLUSTER.png"), dpi=200)
            plt.close(fig)

    except Exception as e:
        print(f"[WARN] Cluster stats failed: {e}")

print(f"\nAll done. Figures in: {FIG_DIR}")
