#raw MEG analysis pipeline comparing all ASMR experimental stimuli trials to Control trials 

#!/usr/bin/env python
#source /scratch/groups/Projects/P1476/mne_env/bin/activate
#python3 /groups/Projects/P1476/MEGanalysisASMRvControl.py
import os
import re
import sys
import numpy as np
import mne
import pickle
import matplotlib.pyplot as plt

from specparam import SpectralModel
from specparam.bands import Bands
from specparam.analysis import get_band_peak

# Maxwell filter for realignment
from mne.preprocessing import maxwell_filter

data_dir = "/groups/Projects/P1476/fifdata"
output_dir = "/scratch/groups/Projects/P1476/MEGanalysis"
os.makedirs(output_dir, exist_ok=True)

# parse participant IDs
participant_id_set = set()
for fname in os.listdir(data_dir):
	match = re.match(r"(R\d{4})_block[123]_meg\.fif", fname)
	if match:
    	pid = match.group(1)  
    	participant_id_set.add(pid)

participant_ids = sorted(list(participant_id_set))
print(f"Found {len(participant_ids)} participants: {participant_ids}")

# event codes for ASMR (101-113) and Control (114-118)
event_id = {}
for i in range(1, 14):
	event_id[f"asmr/t{i}"] = 100 + i
for j in range(14, 19):
	label_j = j - 13
	event_id[f"control/c{label_j}"] = 100 + j

freq_bands = {
	"theta": [4, 8],
	"alpha": [8, 12],
	"beta":  [15, 30],
	"gamma": [30, 48]
}
bands = Bands(freq_bands)

fm = SpectralModel(
	peak_width_limits=[0.25, 6],
	max_n_peaks=6,
	verbose=False
)

n_bands = len(freq_bands)

peakFreqASMR = None
peakAmpASMR  = None
peakFreqCtrl = None
peakAmpCtrl  = None

valid_subj_count = 0
allInfos = []
allFreqs = None

destination_dev_head_t = None
got_reference = False

# loop over the participantts
for pid in participant_ids:
	print(f"\n----- Processing participant {pid} -----")

	block_files = [
    	os.path.join(data_dir, f"{pid}_block1_meg.fif"),
    	os.path.join(data_dir, f"{pid}_block2_meg.fif"),
    	os.path.join(data_dir, f"{pid}_block3_meg.fif")
	]

	asmr_epochs_list = []
	ctrl_epochs_list = []
	any_block_found = False

	for bf in block_files:
    	if not os.path.exists(bf):
        	print(f"File not found: {bf}, skipping.")
        	continue

    	any_block_found = True

    	# load in raw data
    	raw = mne.io.read_raw_fif(bf, preload=True)
    	raw.set_channel_types({'P_PORT_A': 'stim'})

    	if not got_reference:
        	destination_dev_head_t = raw.info['dev_head_t']
        	got_reference = True

    	# Maxwell filter (no calibration/crosstalk files available at ynic)
    	raw_sss = maxwell_filter(
        	raw,
        	destination=destination_dev_head_t,
        	coord_frame='head'
    	)

    	# Notch + band-pass
    	raw_sss.notch_filter(freqs=[50, 100], fir_design='firwin')
    	# keep it consistent with your PSD fmax ~48, so let's band-pass 1..48 or 1..50
    	raw_sss.filter(l_freq=1.0, h_freq=48.0, fir_design='firwin')

    	# find events
    	events = mne.find_events(raw_sss, stim_channel="P_PORT_A")
    	if len(events) == 0:
        	print(f"No events found in {bf}.")
        	continue

    	# pick magnetometers
    	raw_sss.pick_types(meg="mag", stim=False)

    	# create epochs
    	tmin, tmax = -2.0, 30.0
    	reject_criteria = None  # Keep it off for now
    	epochs = mne.Epochs(
        	raw_sss, events, event_id=event_id,
        	tmin=tmin, tmax=tmax,
        	baseline=(-2, 0),
        	reject=reject_criteria,
        	preload=True
    	)
    	if len(epochs) == 0:
        	print(f"No valid epochs in {bf}")
        	continue

    	# separate asmr and control 
    	asmr_event_names = [k for k in event_id if "asmr" in k]
    	ctrl_event_names = [k for k in event_id if "control" in k]

    	asmr_cond_epochs = epochs[asmr_event_names]
    	ctrl_cond_epochs = epochs[ctrl_event_names]

    	if len(asmr_cond_epochs) > 0:
        	asmr_epochs_list.append(asmr_cond_epochs)
    	if len(ctrl_cond_epochs) > 0:
        	ctrl_epochs_list.append(ctrl_cond_epochs)

	# end block loop

	if not any_block_found:
    	print(f"No block files found for {pid}. Skipping participant.")
    	continue

	if (len(asmr_epochs_list) == 0) or (len(ctrl_epochs_list) == 0):
    	print(f"No ASMR or Control epochs for {pid}. Skipping.")
    	continue

	try:
    	asmr_all = mne.concatenate_epochs(asmr_epochs_list)
    	ctrl_all = mne.concatenate_epochs(ctrl_epochs_list)
	except ValueError as e:
    	print(f"Cannot concatenate for {pid}: {e}")
    	continue

	if (len(asmr_all) == 0) or (len(ctrl_all) == 0):
    	print(f"Insufficient data for {pid}. Skipping.")
    	continue

	# PSD (Welch method)
	psd_asmr = asmr_all.compute_psd(method="welch", fmin=1, fmax=48)
	psd_ctrl = ctrl_all.compute_psd(method="welch", fmin=1, fmax=48)

	spectra_asmr, freqs_asmr = psd_asmr.get_data(return_freqs=True)
	spectra_ctrl, freqs_ctrl = psd_ctrl.get_data(return_freqs=True)

	avg_spec_asmr = np.nanmean(spectra_asmr, axis=0)
	avg_spec_ctrl = np.nanmean(spectra_ctrl, axis=0)

	n_channels = avg_spec_asmr.shape[0]

	# init arrays on first valid subject
	if valid_subj_count == 0:
    	from math import ceil
    	max_subj = len(participant_ids)
    	peakFreqASMR = np.zeros((max_subj, n_bands, n_channels))
    	peakAmpASMR  = np.zeros((max_subj, n_bands, n_channels))
    	peakFreqCtrl = np.zeros((max_subj, n_bands, n_channels))
    	peakAmpCtrl  = np.zeros((max_subj, n_bands, n_channels))
    	allFreqs = freqs_asmr

	info = asmr_all.info

	# it’s FOOOFing time
	from specparam.analysis import get_band_peak
	for ch_idx in range(n_channels):
    	asmr_ch = avg_spec_asmr[ch_idx, :]
    	ctrl_ch = avg_spec_ctrl[ch_idx, :]

    	# asmr
    	fm.fit(freqs_asmr, asmr_ch, [1, 48])
    	for b_idx, b_name in enumerate(freq_bands.keys()):
        	pkf, pka, _ = get_band_peak(fm, bands[b_name])
        	if not np.isnan(pkf):
            	peakFreqASMR[valid_subj_count, b_idx, ch_idx] = pkf
        	if not np.isnan(pka):
            	peakAmpASMR[valid_subj_count, b_idx, ch_idx] = pka

    	# control
    	fm.fit(freqs_asmr, ctrl_ch, [1, 48])
    	for b_idx, b_name in enumerate(freq_bands.keys()):
        	pkf, pka, _ = get_band_peak(fm, bands[b_name])
        	if not np.isnan(pkf):
            	peakFreqCtrl[valid_subj_count, b_idx, ch_idx] = pkf
        	if not np.isnan(pka):
            	peakAmpCtrl[valid_subj_count, b_idx, ch_idx] = pka

	valid_subj_count += 1

	print(f"Participant {pid} => ASMR: {len(asmr_all)} epochs, Control: {len(ctrl_all)} epochs.")
	print(f"Done participant {pid}. [valid_subj_count={valid_subj_count}]")

	# store info for topomaps
	import copy
	allInfos.append(copy.deepcopy(info))

# end participant loop

if valid_subj_count == 0:
	print("No valid data found for any subject.")
	sys.exit(0)

# trim arrays
peakFreqASMR = peakFreqASMR[:valid_subj_count, :, :]
peakAmpASMR  = peakAmpASMR[:valid_subj_count, :, :]
peakFreqCtrl = peakFreqCtrl[:valid_subj_count, :, :]
peakAmpCtrl  = peakAmpCtrl[:valid_subj_count, :, :]

info_final = allInfos[-1] if len(allInfos) > 0 else None

# SAVE it
out_data = {
	"peakFreqASMR": peakFreqASMR,
	"peakAmpASMR":  peakAmpASMR,
	"peakFreqCtrl": peakFreqCtrl,
	"peakAmpCtrl":  peakAmpCtrl,
	"freqs":    	allFreqs,
	"band_names":   list(freq_bands.keys()),
	"info":     	info_final
}
out_file = os.path.join(output_dir, "per_subject_fooof_peaks_meg.pkl")
with open(out_file, "wb") as f:
	pickle.dump(out_data, f)

print(f"\nSaved MEG FOOOF data to {out_file}")

# then do group level topomaps, same as before
print("Done.")

peakAmpASMR  = out_data["peakAmpASMR"]
peakAmpCtrl  = out_data["peakAmpCtrl"]
band_names   = out_data["band_names"]
info_meg 	= out_data["info"]

if info_meg is None or peakAmpASMR.shape[0] < 1:
	print("No valid data to plot group topomaps. Exiting.")
	sys.exit(0)

mean_peakAmp_ASMR = np.nanmean(peakAmpASMR, axis=0)  # shape (n_bands, n_channels)
mean_peakAmp_CTRL = np.nanmean(peakAmpCtrl, axis=0)
diff_amp = mean_peakAmp_ASMR - mean_peakAmp_CTRL

import mne
for b_idx, b_name in enumerate(band_names):
	data_asmr = mean_peakAmp_ASMR[b_idx, :]
	data_ctrl = mean_peakAmp_CTRL[b_idx, :]
	data_diff = diff_amp[b_idx, :]

	picks_meg = mne.pick_types(info_meg, meg="mag")
	data_asmr_meg = data_asmr[picks_meg]
	data_ctrl_meg = data_ctrl[picks_meg]
	data_diff_meg = data_diff[picks_meg]

	# ASMR
	fig_asmr, ax_asmr = plt.subplots()
	im_asmr, _ = mne.viz.plot_topomap(
    	data_asmr_meg,
    	pos=info_meg,
    	#picks=picks_meg,
    	cmap="Reds",
    	axes=ax_asmr,
    	show=False
	)
	plt.title(f"ASMR: {b_name}")
	fig_asmr.colorbar(im_asmr, ax=ax_asmr, label="Peak Amp (AU)")
	fig_asmr.savefig(os.path.join(output_dir, f"MEG_ASMR_{b_name}.png"), bbox_inches="tight")
	plt.close(fig_asmr)

	# Control
	fig_ctrl, ax_ctrl = plt.subplots()
	im_ctrl, _ = mne.viz.plot_topomap(
    	data_ctrl_meg,
    	pos=info_meg,
    	#picks=picks_meg,
    	cmap="Blues",
    	axes=ax_ctrl,
    	show=False
	)
	plt.title(f"Control: {b_name}")
	fig_ctrl.colorbar(im_ctrl, ax=ax_ctrl, label="Peak Amp (AU)")
	fig_ctrl.savefig(os.path.join(output_dir, f"MEG_Control_{b_name}.png"), bbox_inches="tight")
	plt.close(fig_ctrl)

	# Difference
	fig_diff, ax_diff = plt.subplots()
	im_diff, _ = mne.viz.plot_topomap(
    	data_diff_meg,
    	pos=info_meg,
    	#picks=picks_meg,
    	cmap="RdBu_r",
    	axes=ax_diff,
    	show=False
	)
	plt.title(f"ASMR-Control: {b_name}")
	fig_diff.colorbar(im_diff, ax=ax_diff, label="Diff Amp (AU)")
	fig_diff.savefig(os.path.join(output_dir, f"MEG_Diff_{b_name}.png"), bbox_inches="tight")
	plt.close(fig_diff)

print("Done: group-level MEG topomaps saved.")
