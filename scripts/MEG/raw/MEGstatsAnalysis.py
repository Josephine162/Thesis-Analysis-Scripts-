#!/usr/bin/env python
#source /scratch/groups/Projects/P1476/mne_env/bin/activate
#python3 /groups/Projects/P1476/MEGstatsAnalysis.py

# a script to run cluster based stats analysis with bonferonni correction on the MEG pickle data
import os
import sys
import numpy as np
import mne
import pickle
import matplotlib.pyplot as plt

from mne.stats import spatio_temporal_cluster_1samp_test, bonferroni_correction
from mne.channels import find_ch_adjacency

# load the per subject FOOOF data
# pickle path
pkl_file = "/scratch/groups/Projects/P1476/MEGanalysis/per_subject_fooof_peaks_meg.pkl"

with open(pkl_file, "rb") as f:
	out_data = pickle.load(f)

peakAmpASMR  = out_data["peakAmpASMR"]   # shape [n_subj, n_bands, n_channels]
peakAmpCtrl  = out_data["peakAmpCtrl"]
band_names   = out_data["band_names"]	
info_meg 	= out_data["info"]      	# for the sensor layout

if info_meg is None or peakAmpASMR.shape[0] < 2:
	print("No valid data or too few subjects for stats.")
	sys.exit(0)

n_subj  = peakAmpASMR.shape[0]
n_bands = peakAmpASMR.shape[1]
n_ch	= peakAmpASMR.shape[2]

print(f"Data loaded: {n_subj} subjects, {n_bands} bands, {n_ch} channels.")

# build adjacency for the MEG sensors 
# the ynic cryo MEG only used magnetometers 
ch_adjacency, ch_names = find_ch_adjacency(info_meg, ch_type='mag')
print(f"Adjacency matrix shape: {ch_adjacency.shape}")

stats_output_dir = "/scratch/groups/Projects/P1476/MEGstats"
os.makedirs(stats_output_dir, exist_ok=True)

# loop over each band, run cluster tests
# store the min cluster p value per band to do a bonferroni correction
band_min_pvals = np.ones(n_bands)  # initialize all to 1.0

T_obs_all = []  	# to store T_obs arrays for each band
mask_all = []   	# to store the significance masks for each band

for b_idx, b_name in enumerate(band_names):
	print(f"\n=== Running cluster-based test for band: {b_name} ===")

	# construct difference => shape [n_subj, n_channels]
	diff_data = peakAmpASMR[:, b_idx, :] - peakAmpCtrl[:, b_idx, :]

	# reshape for cluster function => [n_samples, n_ch, n_times=1]
	X = diff_data[:, :, np.newaxis]  # => (n_subj, n_ch, 1)

	# run spatio_temporal_cluster_1samp_test
	threshold = None
	n_permutations = 2000
	T_obs, clusters, cluster_p_values, H0 = spatio_temporal_cluster_1samp_test(
    	X,
    	adjacency=ch_adjacency,
    	n_permutations=n_permutations,
    	threshold=threshold,
    	tail=0,
    	n_jobs=1
	)
	# T_obs => shape (n_channels, 1)
	# clusters => list of cluster definitions
	# cluster_p_values => p value per cluster
	# H0 => distribution of cluster level stats

	# identify sensors in significant clusters (p<0.05 uncorrected)
	alpha = 0.05
	signif_mask = np.zeros(T_obs.shape, dtype=bool)
	n_clusters = len(clusters)

	if n_clusters > 0:
    	# track the min cluster p value for the band
    	min_cluster_p = np.min(cluster_p_values)
    	band_min_pvals[b_idx] = min_cluster_p

    	for c_idx, c_pval in enumerate(cluster_p_values):
        	if c_pval < alpha:
            	# mark channels in this cluster
            	cluster_inds = clusters[c_idx][0]  # this means channel indices btw
            	for ch_ in cluster_inds:
                	signif_mask[ch_, 0] = True
	else:
    	print("No clusters found at all for this band (maybe no variance?).")

	# Flatten T_obs => shape (n_channels,)
	t_map = T_obs[:, 0]
	# Flatten mask => shape (n_channels,)
	mask = signif_mask[:, 0]

	T_obs_all.append(t_map)
	mask_all.append(mask)

	# plot a topo (unadjusted alpha)
	fig, ax = plt.subplots()
	im, cn = mne.viz.plot_topomap(
    	t_map,
    	pos=info_meg,
    	cmap='RdBu_r',
    	sensors=True,
    	show=False,
    	mask=mask,
    	mask_params=dict(marker='o', markerfacecolor='none',
                     	markeredgecolor='black', linewidth=2),
    	axes=ax
	)
	plt.title(f"{b_name} band (ASMR - Control)\nUncorr p<0.05 cluster test")
	fig.colorbar(im, ax=ax, label="T-value")
	out_fig = os.path.join(stats_output_dir, f"Cluster_{b_name}.png")
	fig.savefig(out_fig, bbox_inches="tight")
	plt.close(fig)
    

# Bonferroni correction across all of the bands 
# we have band_min_pvals => shape (n_bands,)
# if a band had no cluster, it's min p=1.0 by default
print("\n=== Bonferroni Correction for multiple bands ===")
print("Raw min cluster p-values per band:")
for b_idx, b_name in enumerate(band_names):
	print(f"  {b_name}: {band_min_pvals[b_idx]:.5f}")

# correct them
reject_bonf, pvals_bonf = bonferroni_correction(band_min_pvals, alpha=0.05)
# Now pvals_bonf => corrected p values for each band
# reject_bonf => boolean array telling us which bands remain significant

print("\nAfter Bonferroni Correction (familywise alpha=0.05), results:")
for b_idx, b_name in enumerate(band_names):
	print(f"  {b_name} => raw p={band_min_pvals[b_idx]:.5f}, corrected p={pvals_bonf[b_idx]:.5f}, significant={reject_bonf[b_idx]}")

print("\nDone. See cluster maps in:", stats_output_dir)
