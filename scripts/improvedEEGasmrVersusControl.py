improvedEEGasmrVersusControl script: 
Build per-subject FOOOF peak arrays for ALL ASMR vs Control trials (within-subject).
Key fixes vs. older version:
- PSD computed per condition using epochs selection (no event_id indexing).
- Consistent channel ordering across subjects.
- Subject IDs saved for downstream alignment.
"""

import os
import sys
import numpy as np
import pickle
import mne

# FOOOF (specparam)
from specparam import SpectralModel
from specparam.bands import Bands
from specparam.analysis import get_band_peak

# ----------------------------
# 0) USER SETTINGS
# ----------------------------
DATA_DIR   = "/Users/jrf521/Documents/EEGdata/ASMRconverted"         # folder with P##_EEG.set
OUTPUT_DIR = os.path.join(DATA_DIR, "AllASMRvControl_FOOOFout")       # output pickle folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Epoching window for stimulus-locked trials (your sounds were ~30 s)
EPOCH_TMIN = -2.0
EPOCH_TMAX = 30.0
BASELINE   = (-2.0, 0.0)   # PSD is absolute, but keep baseline for transparency

# Preprocessing
RESAMPLE_HZ = 250
HPF, LPF    = 0.5, 100.0
NOTCHS      = [50, 100]    # UK mains

# Bands & FOOOF configuration
FREQ_RANGE  = [1, 48]
FREQ_BANDS  = {"theta":[4,8], "alpha":[8,12], "beta":[15,30], "gamma":[30,48]}
FM = SpectralModel(peak_width_limits=[0.25, 6], max_n_peaks=6, verbose=False)
BANDS = Bands(FREQ_BANDS)
BAND_LIST = list(FREQ_BANDS.keys())

# Subjects (P1..P64)
SUBJECTS = [f"P{i}" for i in range(1, 65)]

# Event dictionary used in EEGLAB annotations → MNE events
EVENT_DICT = {
    "asmr/t1": 1, "asmr/t2": 2, "asmr/t3": 3, "asmr/t4": 4, "asmr/t5": 5, "asmr/t6": 6,
    "asmr/t7": 7, "asmr/t8": 8, "asmr/t9": 9, "asmr/t10": 10, "asmr/t11": 11, "asmr/t12": 12,
    "asmr/t13": 13,
    "control/c1": 14, "control/c2": 15, "control/c3": 16, "control/c4": 17, "control/c5": 18
}
ASMR_KEYS    = [k for k in EVENT_DICT if k.startswith("asmr/")]
CONTROL_KEYS = [k for k in EVENT_DICT if k.startswith("control/")]

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

    # ---- FIX: remove n_jobs="auto" (causing 'only allowed value is cuda' error on your build)
    raw.filter(HPF, LPF)
    if NOTCHS:
        raw.notch_filter(freqs=NOTCHS)
    if RESAMPLE_HZ:
        raw.resample(RESAMPLE_HZ)

    return raw

def epoch_all_conditions(raw, event_dict):
    """Create Epochs with all triggers, then we will select conditions from this Epochs object."""
    events, _ = mne.events_from_annotations(raw)
    epochs = mne.Epochs(
        raw, events, event_id=event_dict,
        tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
        baseline=BASELINE,
        reject=None, preload=True, picks="eeg",
    )
    return epochs

def avg_psd_for_keys(epochs, keys, fmin=1, fmax=48):
    """Compute PSD on the selection epochs[keys], average across epochs → (n_ch, n_freq), freqs."""
    if len(keys) == 0:
        return None, None
    sel = epochs[keys]  # selection by string list
    if len(sel) == 0:
        return None, None

    psd = sel.compute_psd(method="welch", fmin=fmin, fmax=fmax)
    spectra, freqs = psd.get_data(return_freqs=True)  # (n_ep, n_ch, n_freq)
    spectra = np.nan_to_num(spectra, nan=0.0, posinf=0.0, neginf=0.0)
    # Average across epochs → (n_ch, n_freq)
    return np.nanmean(spectra, axis=0), freqs

def reindex_channels(arr, from_names, to_names):
    """Reorder channel axis of arr (n_ch, ...) from 'from_names' to 'to_names'. """
    if from_names == to_names:
        return arr
    idx_map = [from_names.index(ch) for ch in to_names]
    return arr[idx_map, ...]

# ----------------------------
# 1) Main loop
# ----------------------------
peakFreqASMR    = None
peakAmpASMR     = None
peakFreqControl = None
peakAmpControl  = None
all_freqs       = None
template_chs    = None
template_info   = None
kept_subjects   = []

for si, subj in enumerate(SUBJECTS):
    eeg_path = os.path.join(DATA_DIR, f"{subj}_EEG.set")
    if not os.path.exists(eeg_path):
        print(f"[WARN] Missing: {eeg_path}. Skipping {subj}.")
        continue

    try:
        print(f"\n--- {subj} ---")
        raw = load_preprocess_raw(eeg_path)
        epochs = epoch_all_conditions(raw, EVENT_DICT)

        if len(epochs) == 0:
            print("No epochs formed, skipping.")
            continue

        # Condition-wise PSD
        asmr_avg, freqs = avg_psd_for_keys(epochs, ASMR_KEYS, fmin=FREQ_RANGE[0], fmax=FREQ_RANGE[1])
        ctrl_avg, _     = avg_psd_for_keys(epochs, CONTROL_KEYS, fmin=FREQ_RANGE[0], fmax=FREQ_RANGE[1])

        if asmr_avg is None or ctrl_avg is None:
            print("ASMR or Control had no valid epochs; skipping.")
            continue

        # Initialize template channel order from first valid subject
        ch_names = epochs.info["ch_names"]
        if template_chs is None:
            template_chs  = ch_names.copy()
            template_info = epochs.info

        # Reindex current subject’s channel order to template
        if ch_names != template_chs:
            asmr_avg = reindex_channels(asmr_avg, ch_names, template_chs)
            ctrl_avg = reindex_channels(ctrl_avg, ch_names, template_chs)

        # Init output arrays on first valid subject
        if all_freqs is None:
            all_freqs = freqs
            n_subj_est   = len(SUBJECTS)
            n_ch         = asmr_avg.shape[0]
            n_bands      = len(BAND_LIST)
            peakFreqASMR    = np.zeros((n_subj_est, n_bands, n_ch))
            peakAmpASMR     = np.zeros((n_subj_est, n_bands, n_ch))
            peakFreqControl = np.zeros((n_subj_est, n_bands, n_ch))
            peakAmpControl  = np.zeros((n_subj_est, n_bands, n_ch))

        # FOOOF per channel, ASMR
        for ch in range(asmr_avg.shape[0]):
            FM.fit(all_freqs, asmr_avg[ch, :], FREQ_RANGE)
            for b_idx, b_name in enumerate(BAND_LIST):
                pk_f, pk_a, _ = get_band_peak(FM, BANDS[b_name])
                if not np.isnan(pk_f): peakFreqASMR[si, b_idx, ch] = pk_f
                if not np.isnan(pk_a): peakAmpASMR[si,  b_idx, ch] = pk_a

        # FOOOF per channel, Control
        for ch in range(ctrl_avg.shape[0]):
            FM.fit(all_freqs, ctrl_avg[ch, :], FREQ_RANGE)
            for b_idx, b_name in enumerate(BAND_LIST):
                pk_f, pk_a, _ = get_band_peak(FM, BANDS[b_name])
                if not np.isnan(pk_f): peakFreqControl[si, b_idx, ch] = pk_f
                if not np.isnan(pk_a): peakAmpControl[si,  b_idx, ch] = pk_a

        kept_subjects.append(subj)
        print(f"Kept {subj}: ASMR/Control FOOOF done.")

    except Exception as e:
        print(f"[ERROR] {subj}: {e}")
        continue

# Truncate to kept subjects
n_keep = len(kept_subjects)
if n_keep == 0:
    print("No valid subjects processed. Exiting.")
    sys.exit(1)

peakFreqASMR     = peakFreqASMR[:n_keep, :, :]
peakAmpASMR      = peakAmpASMR[:n_keep, :, :]
peakFreqControl  = peakFreqControl[:n_keep, :, :]
peakAmpControl   = peakAmpControl[:n_keep, :, :]

# ----------------------------
# 2) Save pickle
# ----------------------------
out_pickle = os.path.join(OUTPUT_DIR, "per_subject_fooof_peaks_ASMR_allTrials.pkl")
save_dict = {
    "subjects":       kept_subjects,      # <- use this for alignment in stats
    "peakFreqASMR":   peakFreqASMR,
    "peakAmpASMR":    peakAmpASMR,
    "peakFreqControl":peakFreqControl,
    "peakAmpControl": peakAmpControl,
    "freqs":          all_freqs,
    "band_names":     BAND_LIST,
    "info":           template_info,      # adjacency reference
    "ch_names":       template_chs        # explicit channel order
}

with open(out_pickle, "wb") as f:
    pickle.dump(save_dict, f)

print(f"\nSaved per-subject FOOOF data to:\n  {out_pickle}")
print(f"Subjects kept (n={n_keep}): {kept_subjects}")
