# Runs FOOOF on 32-s ASMR trials that contained any button-press event, 
# providing spectral characterisation of ASMR-responsive trials independent of press duration.
import os
import sys
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# FOOOF (specparam) imports
import specparam
from specparam import SpectralModel
from specparam.bands import Bands
from specparam.analysis import get_band_peak

from sklearn.preprocessing import StandardScaler
import pickle

# Custom utilities (assume you have readAndImportData to load MNE Raw)
from utils import readAndImportData

# --------------------------------------------------
# 1) SETUP
# --------------------------------------------------
DO_ZSCORING = True
Z_THRESHOLD = 8.0  # Looser threshold for artifact rejection
EPOCH_TMIN = -2.0
EPOCH_TMAX = 30.0   # Now capturing a full 32s window around press onset

data_dir = "/Users/jrf521/Documents/EEGdata/ASMRconverted/"
csv_dir  = "/Users/jrf521/Documents/EEGdata/buttonpresstriggers/bpTrialsOutput/"
output_directory = os.path.join(data_dir, "BPgroupASMRout_anyPress_FULLTRIAL")

os.makedirs(output_directory, exist_ok=True)

sublist = list(range(1, 65))

# Frequency range for FOOOF
freq_range = [1, 48]
freq_bands = {
    "theta": [4, 8],
    "alpha": [8, 12],
    "beta":  [15, 30],
    "gamma": [30, 48]
}
bands = Bands(freq_bands)

# Initialize a FOOOF model
fm = SpectralModel(
    peak_width_limits=[0.25, 6],
    max_n_peaks=6,
    verbose=False
)

max_subjects = len(sublist)
n_bands = len(freq_bands)

# Arrays to store results
peakFreqASMR_BP = None
peakAmpASMR_BP  = None

valid_subj_count = 0
allInfos = []
allFreqs = None

# --------------------------------------------------
# 2) LOOP OVER SUBJECTS
# --------------------------------------------------
for si, subj_num in enumerate(sublist):
    subject = f"P{subj_num}"
    print(f"\n--- Processing subject: {subject} ---")

    raw_fname = os.path.join(data_dir, f"{subject}_EEG.set")
    csv_file  = os.path.join(csv_dir, f"{subject}_press_events.csv")

    try:
        # 2.1) Load Raw EEG Data
        raw = readAndImportData(raw_fname)
        if raw is None:
            print(f"Raw file not found or unreadable for {subject}, skipping.")
            continue

        # 2.2) Load CSV
        if not os.path.exists(csv_file):
            print(f"No CSV file found for {subject} at {csv_file}, skipping.")
            continue
        press_data = pd.read_csv(csv_file)
        if press_data.empty:
            print(f"No data in CSV for {subject}, skipping.")
            continue

        # 2.3) Filter for ASMR Stimuli (1..13)
        asmr_df = press_data[press_data["Stimulus"].isin(range(1, 14))]
        if asmr_df.empty:
            print(f"No button-press ASMR events (1..13) for {subject}, skipping.")
            continue

        # Check columns for press onset/end in ms
        required_cols = ["Press Start Time", "Press End Time"]
        if not all(col in asmr_df.columns for col in required_cols):
            print(f"CSV missing Press Start Time / Press End Time for {subject}, skipping.")
            continue

        print(f"Subject {subject}: total ASMR button presses = {len(asmr_df)}")

        # 2.4) Convert ms -> seconds
        asmr_df["Onset_s"] = asmr_df["Press Start Time"] / 1000.0

        # 2.5) Create MNE events for each press onset
        sfreq = raw.info["sfreq"]
        events_list = []
        event_id = {"asmr_bp": 999}

        for _, row in asmr_df.iterrows():
            onset_sample = int(row["Onset_s"] * sfreq)
            events_list.append([onset_sample, 0, event_id["asmr_bp"]])

        if len(events_list) == 0:
            print(f"No valid event onsets for {subject}, skipping.")
            continue

        events_array = np.array(events_list, dtype=int)
        print(f"Subject {subject}: final # of events = {len(events_array)}")

        # 2.6) Create epochs from -2s to +30s around the press onset
        all_epochs = mne.Epochs(
            raw,
            events_array,
            event_id=event_id,
            tmin=EPOCH_TMIN,
            tmax=EPOCH_TMAX,
            baseline=None,
            reject=None,
            preload=True,
            picks="eeg"
        )

        if len(all_epochs) == 0:
            print(f"No epochs formed for {subject}, skipping.")
            continue

        print(f"Subject {subject}: formed {len(all_epochs)} epochs before artifact rejection.")

        # 2.7) Z-Score Artifact Rejection
        if DO_ZSCORING:
            data_epo = all_epochs.get_data()  # => (n_epochs, n_channels, n_times)
            epochStd = np.std(data_epo, axis=2)
            z_scores = (epochStd - np.nanmean(epochStd)) / (np.nanstd(epochStd) + 1e-8)
            z_flat = z_scores.flatten()
            bad_idxs = np.where(z_flat > Z_THRESHOLD)[0]
            if len(bad_idxs) > 0:
                all_epochs.drop(bad_idxs, reason="zscore")

        if len(all_epochs) == 0:
            print(f"All epochs dropped for {subject} after artifact threshold, skipping.")
            continue

        print(f"Subject {subject}: {len(all_epochs)} epochs remain after zscore artifact removal.")

        # 2.8) Compute PSD
        psd = all_epochs.compute_psd(method="welch", fmin=1, fmax=48)
        spectra, freqs = psd.get_data(return_freqs=True)  # shape => (n_epochs, n_channels, n_freqs)

        spectra[spectra == 0] = 1e-10
        spectra = np.nan_to_num(spectra, nan=0.0, posinf=0.0, neginf=0.0)

        avg_spectra = np.nanmean(spectra, axis=0)  # => (n_channels, n_freqs)
        n_channels, n_freqs = avg_spectra.shape

        # Init arrays on first valid subject
        if valid_subj_count == 0:
            peakFreqASMR_BP = np.zeros((max_subjects, n_bands, n_channels))
            peakAmpASMR_BP  = np.zeros((max_subjects, n_bands, n_channels))
            allFreqs = freqs

        # 2.9) Fit FOOOF for each channel
        band_list = list(freq_bands.keys())
        for ch_idx in range(n_channels):
            spec_ch = avg_spectra[ch_idx, :]
            if np.sum(spec_ch) > 0:
                fm.fit(freqs, spec_ch, freq_range)
                for b_idx, b_name in enumerate(band_list):
                    pk_freq, pk_amp, _ = get_band_peak(fm, bands[b_name])
                    if not np.isnan(pk_freq):
                        peakFreqASMR_BP[valid_subj_count, b_idx, ch_idx] = pk_freq
                    if not np.isnan(pk_amp):
                        peakAmpASMR_BP[valid_subj_count, b_idx, ch_idx]  = pk_amp

        valid_subj_count += 1
        allInfos.append(raw.info)
        print(f"Subject {subject} => final PSD from {len(all_epochs)} epochs. [valid_subj_count={valid_subj_count}]")

    except Exception as e:
        print(f"Error processing {subject}: {e}")
        continue

# End subject loop
if valid_subj_count == 0:
    print("No valid button-press ASMR data found for any subject.")
    sys.exit(0)

# Truncate arrays to number of valid subjects
peakFreqASMR_BP = peakFreqASMR_BP[:valid_subj_count, :, :]
peakAmpASMR_BP  = peakAmpASMR_BP[:valid_subj_count, :, :]

info = allInfos[-1] if len(allInfos) > 0 else None

# --------------------------------------------------
# 3) SAVE RESULTS
# --------------------------------------------------
out_pickle = os.path.join(output_directory, "per_subject_fooof_peaks_ASMR_BP_FULLTRIAL.pkl")
save_dict = {
    "peakFreqASMR_BP": peakFreqASMR_BP,
    "peakAmpASMR_BP":  peakAmpASMR_BP,
    "freqs": allFreqs,
    "band_names": list(freq_bands.keys()),
    "info": info
}

with open(out_pickle, "wb") as f:
    pickle.dump(save_dict, f)

print(f"\nSaved ASMR button-press (full 32s) data to: {out_pickle}")
print("Done.")

