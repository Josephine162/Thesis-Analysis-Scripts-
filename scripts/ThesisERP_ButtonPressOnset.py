# Button-press ERP (onset-locked) analysis for ASMR trials.
# - Extracts ±1 s around each Press Start Time (from *_press_events.csv).
# - Baseline: −0.5 → 0 s.
# - Averages across presses per subject, then across subjects.
# - Produces grand-average ERP waveform and scalp topography.
"""

import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1) PATHS AND PARAMETERS
# --------------------------------------------------
DATA_DIR = "/Users/jrf521/Documents/EEGdata/ASMRconverted/"
CSV_DIR  = "/Users/jrf521/Documents/EEGdata/buttonpresstriggers/bpTrialsOutput/"
OUT_DIR  = os.path.join(DATA_DIR, "ERP_buttonpress_onset_out")
os.makedirs(OUT_DIR, exist_ok=True)

SUBJECTS = [f"P{i}" for i in range(1, 65)]
EPOCH_TMIN, EPOCH_TMAX = -1.0, 1.0
BASELINE = (-0.5, 0.0)

# Filtering for ERP clarity
HPF, LPF = 0.1, 30.0
NOTCHS = [50, 100]

# --------------------------------------------------
# 2) STORAGE
# --------------------------------------------------
all_evokeds = []
kept_subs = []

# --------------------------------------------------
# 3) LOOP OVER PARTICIPANTS
# --------------------------------------------------
for subj in SUBJECTS:
    eeg_path = os.path.join(DATA_DIR, f"{subj}_EEG.set")
    csv_path = os.path.join(CSV_DIR, f"{subj}_press_events.csv")
    if not os.path.exists(eeg_path) or not os.path.exists(csv_path):
        continue

    try:
        print(f"\n--- {subj} ---")
        raw = mne.io.read_raw_eeglab(eeg_path, preload=True)
        # drop artefactual channels if present
        for ch in ["HEOG", "VEOG", "M1", "M2"]:
            if ch in raw.ch_names:
                raw.drop_channels(ch)

        raw.set_montage("standard_1020", on_missing="ignore")
        raw.set_eeg_reference("average")
        raw.filter(HPF, LPF)
        raw.notch_filter(freqs=NOTCHS)

        # Load button-press CSV
        df = pd.read_csv(csv_path)
        df = df[df["Stimulus"].isin(range(1, 14))]   # ASMR trials only
        if df.empty:
            continue

        # Convert to seconds and make events for press onset
        sfreq = raw.info["sfreq"]
        events = np.array([[int(t * sfreq / 1000.0), 0, 999]
                           for t in df["Press Start Time"].values
                           if not np.isnan(t)], dtype=int)
        if len(events) == 0:
            continue

        # Epoch ±1 s around button-press onset
        epochs = mne.Epochs(
            raw, events, event_id={"press_onset": 999},
            tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
            baseline=BASELINE, preload=True, picks="eeg"
        )

        if len(epochs) == 0:
            continue

        evoked = epochs.average()
        all_evokeds.append(evoked)
        kept_subs.append(subj)
        print(f"Kept {subj}: {len(epochs)} press-onset epochs.")

    except Exception as e:
        print(f"[WARN] {subj}: {e}")
        continue

# --------------------------------------------------
# 4) GRAND-AVERAGE ERP
# --------------------------------------------------
if len(all_evokeds) == 0:
    raise RuntimeError("No valid ERPs computed — check data paths.")

grand_avg = mne.grand_average(all_evokeds)
grand_avg.comment = "Button-press onset ERP"

# Save grand average
grand_avg.save(os.path.join(OUT_DIR, "grand_average_buttonpress_ERP-ave.fif"), overwrite=True)

# --------------------------------------------------
# 5) VISUALISATION
# --------------------------------------------------
# ERP waveform at representative electrodes
grand_avg.plot_joint(title="Grand-average ERP (ASMR button-press onset)",
                     times=[-0.3, 0.0, 0.2, 0.5])

# Topography at key moments (fixed)
times_to_plot = np.linspace(-0.3, 0.3, 5)
fig = grand_avg.plot_topomap(times=times_to_plot, ch_type="eeg",
                             time_unit="s", show=True)
fig.suptitle("ERP topography around button-press onset", fontsize=14)

# Export static images
plt.savefig(os.path.join(OUT_DIR, "ERP_topography_buttonpress.png"), dpi=300, bbox_inches="tight")

print(f"\nSaved outputs in: {OUT_DIR}")
print(f"Subjects included: n={len(kept_subs)}")
