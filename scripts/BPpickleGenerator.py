#Builds a BP (button-press) pickle for downstream BP-vs-Control stats.

# What it does per subject:
# 1) Load EEGLAB .set, standard preproc (filter, notch, resample, avg-ref).
# 2) Auto-detect ASMR and Control blocks from *numeric* annotation names:
 #  - ASMR labels: "1".."13"
 #   - CONTROL labels: "14".."18"
   (Ignores "__", "0, Impedance".)
# 3) Read BP CSV (Stimulus, Press Start Time, Press End Time) in ms.
# 4) Keep only presses that fall inside ASMR blocks; tile 2 s windows.
# 5) Draw the same number of 2 s windows from Control blocks (random, seeded).
# 6) PSD (Welch) → average → FOOOF per channel, per band (theta/alpha/beta/gamma).
# 7) Save pickle with subjects, ch_names, info, band_names, freqs,
   peakAmpASMR_BP, peakAmpCTRL_win, peakFreq..., and window counts.

Output:
  <DATA_DIR>/BPgroupASMRout_anyPress/per_subject_fooof_peaks_ASMR_BP_anyPress.pkl
"""

import os, sys, re
import numpy as np
import pandas as pd
import pickle
import mne
from specparam import SpectralModel
from specparam.bands import Bands
from specparam.analysis import get_band_peak

# ----------------------------
# PATHS: EDIT THESE TWO
# ----------------------------
DATA_DIR = "/Users/jrf521/Documents/EEGdata/ASMRconverted"  # P##_EEG.set lives here
CSV_DIR  = "/Users/jrf521/Documents/EEGdata/buttonpresstriggers/bpTrialsOutput"  # P##_press_events.csv

# ----------------------------
# OUTPUT (auto-created)
# ----------------------------
OUT_DIR = os.path.join(DATA_DIR, "BPgroupASMRout_anyPress")
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# SUBJECTS (start with P12 for a quick sanity check; then switch back)
# ----------------------------
# SUBJECTS = ["P12"]
SUBJECTS = [f"P{i}" for i in range(1, 65)]

# ----------------------------
# SIGNAL / ANALYSIS PARAMS
# ----------------------------
RESAMPLE_HZ = 250
HPF, LPF    = 0.5, 100.0
NOTCHS      = [50, 100]  # UK mains

BLOCK_TMAX  = 30.0       # seconds from block onset
WIN_LEN     = 2.0        # seconds
WIN_OVERLAP = 0.0        # seconds (0 = non-overlap)
MATCH_CTRL  = "random"   # "random" or "first"
RNG_SEED    = 7

FREQ_RANGE  = [1, 48]
FREQ_BANDS  = {"theta":[4,8], "alpha":[8,12], "beta":[15,30], "gamma":[30,48]}
BANDS       = Bands(FREQ_BANDS)
BAND_LIST   = list(FREQ_BANDS.keys())
FM          = SpectralModel(peak_width_limits=[0.25, 6], max_n_peaks=6, verbose=False)

# ----------------------------
# HELPERS
# ----------------------------
def load_preprocess_raw(eeg_set_path):
    raw = mne.io.read_raw_eeglab(eeg_set_path, preload=True)
    # drop common non-EEG channels if present
    for ch in ["HEOG", "VEOG", "M1", "M2"]:
        if ch in raw.ch_names:
            raw.drop_channels([ch])
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
    raw.set_eeg_reference("average")
    raw.filter(HPF, LPF)
    if NOTCHS:
        raw.notch_filter(freqs=NOTCHS)
    if RESAMPLE_HZ:
        raw.resample(RESAMPLE_HZ)
    return raw

def detect_block_codes(event_id):
    """
    Your files use numeric annotation *names* like "1".."18".
    Map names -> codes:
      ASMR: "1".."13"
      CTRL: "14".."18"
    Ignore "__" and "0, Impedance".
    """
    asmr, ctrl = set(), set()
    for name, code in event_id.items():
        s = str(name).strip()
        if s in {"__", "0, Impedance"}:
            continue
        if s.isdigit():
            n = int(s)
            if 1 <= n <= 13:
                asmr.add(int(code))
            elif 14 <= n <= 18:
                ctrl.add(int(code))
    return asmr, ctrl

def block_windows_from_codes(raw, codes, tmin_rel=0.0, tmax_rel=BLOCK_TMAX):
    """Return [(start_s, stop_s)] for each event whose code is in codes."""
    events, _ = mne.events_from_annotations(raw)
    sf = raw.info["sfreq"]
    out = []
    for sample, _, code in events:
        if int(code) in codes:
            t0 = sample / sf + tmin_rel
            t1 = sample / sf + tmax_rel
            out.append((t0, t1))
    return out

def intersect_windows(a, b):
    """Intersect two lists of (start, stop) windows (seconds)."""
    out = []
    for s0, e0 in a:
        for s1, e1 in b:
            s, e = max(s0, s1), min(e0, e1)
            if e > s:
                out.append((s, e))
    return out

def windows_to_events(raw, windows, win_len=WIN_LEN, overlap=WIN_OVERLAP):
    """Tile each (start, stop) with fixed-length events -> (N, 3) events array."""
    evs = []
    for (start, stop) in windows:
        if stop - start < win_len:
            continue
        ev = mne.make_fixed_length_events(raw, id=1, start=start, stop=stop,
                                          duration=win_len, overlap=overlap)
        if ev is not None and len(ev) > 0:
            evs.append(ev)
    if not evs:
        return np.zeros((0, 3), dtype=int)
    return np.vstack(evs)

def match_n_control_events(ev_ctrl, n_needed, rng, mode="random"):
    if len(ev_ctrl) <= n_needed:
        return ev_ctrl
    if mode == "first":
        return ev_ctrl[:n_needed, :]
    idx = rng.choice(len(ev_ctrl), size=n_needed, replace=False)
    return ev_ctrl[np.sort(idx), :]

def avg_psd_from_events(raw, events, tmin=0.0, tmax=WIN_LEN, fmin=FREQ_RANGE[0], fmax=FREQ_RANGE[1]):
    """Create fixed-length epochs at events, return mean PSD (n_ch, n_freq) + freqs."""
    if len(events) == 0:
        return None, None
    epochs = mne.Epochs(raw, events, event_id=dict(win=1), tmin=tmin, tmax=tmax,
                        baseline=None, reject=None, preload=True, picks="eeg")
    if len(epochs) == 0:
        return None, None
    psd = epochs.compute_psd(method="welch", fmin=fmin, fmax=fmax)
    spectra, freqs = psd.get_data(return_freqs=True)  # (n_ep, n_ch, n_freq)
    spectra = np.nan_to_num(spectra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.nanmean(spectra, axis=0), freqs

def read_bp_csv(csv_path):
    """Return [(start_s, end_s), ...] from per-subject CSV (ms -> s)."""
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path)
    need = {"Stimulus", "Press Start Time", "Press End Time"}
    if df.empty or not need.issubset(df.columns):
        return []
    starts = df["Press Start Time"].to_numpy(dtype=float) / 1000.0
    ends   = df["Press End Time"].to_numpy(dtype=float) / 1000.0
    out = [(float(s), float(e)) for s, e in zip(starts, ends)
           if np.isfinite(s) and np.isfinite(e) and e > s]
    return out

# ----------------------------
# MAIN
# ----------------------------
rng = np.random.default_rng(RNG_SEED)

# Collect per-subject results in lists (avoids indexing gaps if some subjects skip)
subjects_kept = []
bp_win_counts = []
ctrl_win_counts = []
peakAmp_ASMRBP_list   = []
peakAmp_CTRL_list     = []
peakFreq_ASMRBP_list  = []
peakFreq_CTRL_list    = []
template_info = None
template_chs  = None
all_freqs     = None

for subj in SUBJECTS:
    eeg_path = os.path.join(DATA_DIR, f"{subj}_EEG.set")
    csv_path = os.path.join(CSV_DIR,  f"{subj}_press_events.csv")

    if not os.path.exists(eeg_path):
        print(f"[WARN] Missing EEG: {eeg_path} -> skip {subj}")
        continue
    if not os.path.exists(csv_path):
        print(f"[WARN] Missing BP CSV: {csv_path} -> skip {subj}")
        continue

    try:
        print(f"\n--- {subj} ---")
        raw = load_preprocess_raw(eeg_path)

        # Detect ASMR/CTRL codes from numeric annotation names
        _, event_id = mne.events_from_annotations(raw)
        asmr_codes, ctrl_codes = detect_block_codes(event_id)
        print("  Detected ASMR label-codes:", sorted(asmr_codes))
        print("  Detected CTRL label-codes:", sorted(ctrl_codes))
        if not asmr_codes or not ctrl_codes:
            print("  [WARN] No ASMR/CTRL codes found; skipping.")
            continue

        # Build block windows
        asmr_blocks = block_windows_from_codes(raw, asmr_codes)
        ctrl_blocks = block_windows_from_codes(raw, ctrl_codes)
        if len(asmr_blocks) == 0 or len(ctrl_blocks) == 0:
            print("  [WARN] No ASMR/CTRL blocks in data; skipping.")
            continue

        # BP intervals from CSV, restricted to ASMR blocks
        bp_intervals = read_bp_csv(csv_path)
        print(f"  CSV BP intervals: {len(bp_intervals)}")
        if len(bp_intervals) == 0:
            print("  [WARN] No BP intervals in CSV; skipping.")
            continue
        bp_in_asmr = intersect_windows(bp_intervals, asmr_blocks)
        if len(bp_in_asmr) == 0:
            print("  [WARN] No BP intervals inside ASMR blocks; skipping.")
            continue

        # Tile windows
        ev_bp       = windows_to_events(raw, bp_in_asmr, win_len=WIN_LEN, overlap=WIN_OVERLAP)
        ev_ctrl_all = windows_to_events(raw, ctrl_blocks, win_len=WIN_LEN, overlap=WIN_OVERLAP)
        if len(ev_bp) == 0 or len(ev_ctrl_all) == 0:
            print(f"  [WARN] Too few windows (BP={len(ev_bp)}, CTRL={len(ev_ctrl_all)}); skipping.")
            continue

        # Match counts
        ev_ctrl = match_n_control_events(ev_ctrl_all, len(ev_bp), rng, mode=MATCH_CTRL)

        # PSD means
        bp_avg, freqs = avg_psd_from_events(raw, ev_bp,  tmin=0.0, tmax=WIN_LEN)
        ct_avg, _     = avg_psd_from_events(raw, ev_ctrl, tmin=0.0, tmax=WIN_LEN)
        if bp_avg is None or ct_avg is None:
            print("  [WARN] Invalid PSD averages; skipping.")
            continue

        # Template channel order
        eeg_only = raw.copy().pick("eeg")
        ch_names = eeg_only.ch_names
        if template_chs is None:
            template_chs  = ch_names.copy()
            template_info = eeg_only.info
        if ch_names != template_chs:
            idx_map = [ch_names.index(ch) for ch in template_chs]
            bp_avg = bp_avg[idx_map, :]
            ct_avg = ct_avg[idx_map, :]

        # Set freqs
        if all_freqs is None:
            all_freqs = freqs

        # FOOOF per channel for BP & CTRL
        n_ch    = bp_avg.shape[0]
        n_bands = len(BAND_LIST)
        amp_bp   = np.zeros((n_bands, n_ch))
        amp_ct   = np.zeros((n_bands, n_ch))
        freq_bp  = np.zeros((n_bands, n_ch))
        freq_ct  = np.zeros((n_bands, n_ch))

        for ch in range(n_ch):
            FM.fit(all_freqs, bp_avg[ch, :], FREQ_RANGE)
            for bi, bname in enumerate(BAND_LIST):
                pkf, pka, _ = get_band_peak(FM, BANDS[bname])
                if not np.isnan(pkf): freq_bp[bi, ch] = pkf
                if not np.isnan(pka): amp_bp[bi,  ch] = pka

            FM.fit(all_freqs, ct_avg[ch, :], FREQ_RANGE)
            for bi, bname in enumerate(BAND_LIST):
                pkf, pka, _ = get_band_peak(FM, BANDS[bname])
                if not np.isnan(pkf): freq_ct[bi, ch] = pkf
                if not np.isnan(pka): amp_ct[bi,  ch] = pka

        # Append per-subject results
        subjects_kept.append(subj)
        bp_win_counts.append(int(len(ev_bp)))
        ctrl_win_counts.append(int(len(ev_ctrl)))
        peakAmp_ASMRBP_list.append(amp_bp)
        peakAmp_CTRL_list.append(amp_ct)
        peakFreq_ASMRBP_list.append(freq_bp)
        peakFreq_CTRL_list.append(freq_ct)
        print(f"  OK: BP windows={len(ev_bp)}, CTRL windows={len(ev_ctrl)}")

    except Exception as e:
        print(f"[ERROR] {subj}: {e}")
        continue

# ----------------------------
# STACK & SAVE
# ----------------------------
if len(subjects_kept) == 0:
    print("No valid subjects processed. Exiting.")
    sys.exit(1)

peakAmpASMR_BP   = np.stack(peakAmp_ASMRBP_list,  axis=0)  # (n_subj, n_bands, n_ch)
peakAmpCTRL_win  = np.stack(peakAmp_CTRL_list,    axis=0)
peakFreqASMR_BP  = np.stack(peakFreq_ASMRBP_list, axis=0)
peakFreqCTRL_win = np.stack(peakFreq_CTRL_list,   axis=0)

save_dict = {
    "subjects":          subjects_kept,
    "ch_names":          template_chs,
    "info":              template_info,
    "band_names":        BAND_LIST,
    "freqs":             all_freqs,
    "win_len":           WIN_LEN,
    "bp_win_counts":     bp_win_counts,
    "ctrl_win_counts":   ctrl_win_counts,
    "peakAmpASMR_BP":    peakAmpASMR_BP,
    "peakFreqASMR_BP":   peakFreqASMR_BP,
    "peakAmpCTRL_win":   peakAmpCTRL_win,
    "peakFreqCTRL_win":  peakFreqCTRL_win,
}

out_pkl = os.path.join(OUT_DIR, "per_subject_fooof_peaks_ASMR_BP_anyPress.pkl")
with open(out_pkl, "wb") as f:
    pickle.dump(save_dict, f)

print(f"\nSaved BP pickle to:\n  {out_pkl}")
print(f"Subjects kept (n={len(subjects_kept)}): {subjects_kept[:10]}{'...' if len(subjects_kept)>10 else ''}")
print(f"Mean windows per subject: BP={np.mean(bp_win_counts):.1f}, CTRL={np.mean(ctrl_win_counts):.1f}")
