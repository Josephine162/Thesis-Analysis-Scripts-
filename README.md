# Thesis-Analysis-Scripts

Analysis scripts for EEG and MEG sensor-space analyses, including:
- FOOOF spectral parametrisation
- TFR (time-frequency representations)
- ERP (event-related potentials)
- Behavioural and LMM analyses in R

This repository contains only analysis code (no data).  
Detailed instructions for reproducing the EEG analysis pipeline are provided in the main README once scripts are added.
All EEG analysis scripts used in the thesis are provided below, grouped by analytic step.

### EEG Spectral Parameterisation (FOOOF)
- `improvedEEGasmrVersusControl.py` – per-participant ASMR vs Control analysis  
- `ASMRvControlTopoAndClusterStats.py` – group-level permutation stats and topomaps  

### EEG Button-Press Analyses
- `BPpickleGenerator.py` – extract and preprocess button-press epochs  
- `improvedBPvControlTopoAndClusterStats.py` – button-press vs Control cluster analysis  
- `BP_Onset_FullTrial_FOOOF.py` – optional full-trial FOOOF for ASMR trials with presses  

### Time-Frequency and ERP Analyses
- `EEGtfASMRvControl.py` – time0frequency decomposition (ASMR vs Control)  
- `ThesisERP_ButtonPressOnset.py` – ERP analysis around button-press onset  

All scripts were implemented in Python 3.11 using MNE-Python v1.7, Matplotlib, NumPy, and SpecParam (FOOOF). Figures in Chapter 3 correspond directly to the outputs from these scripts.

### MEG Spectral Parameterisation (FOOOF)
- `scripts/MEG/raw/MEGanalysisASMRvControl.py` – per-participant MEG ASMR vs Control PSD + FOOOF peak extraction (magnetometers only; saves per-subject pickle + basic topomaps)
- `scripts/MEG/raw/MEGstatsAnalysis.py` – group-level cluster-based permutation stats across MEG magnetometers (Bonferroni across frequency bands)

### R Scripts (Behavioural / Survey / Intervention)
- `scripts/r/raw/EEGSurvey_IllustrativeModels_original.R` - EEG study survey models & figures (pleasantness/arousal predicting ASMR reports + tingle intensity LMMs)
- `scripts/r/raw/PainSurveyPrepnAnalysis_original.R` – chronic pain intervention data prep + mixed-effects analyses and visualisations

*Note: files in `raw/` are the original thesis-analysis scripts and may contain exploratory/duplicated plotting code; cleaned/annotated versions can later be added*

