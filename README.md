# Master of Data Science - Universitat de Girona
## EEG Data Analysis

Electroencephalography (EEG) is one of the best and most recommended non-invasive methods used in neural science, it is responsible for capturing the electrical activity that occurs in the brain of living beings.

The present repository contains a set of Python Notebooks developed for the Master's Final Project in the Data Science Master Degree at the University of Girona. This is a project implemented in collaboration with the eXiT Research Group.

The main objective is to interpret the EEG data structure to determine the possible predictive scope by using only frontotemporal EEG Channels such as FP, FP1, and FP2. After developing the predictive model for only such channels, a comparative analysis will be run with the Prediction Results by using all the possible channels available and identifying the impact and importance of the Frontotemporal Channels previously analyzed.

Tools used:
* Python Notebooks (Libraries: MNE, Pandas, Matplotlib, and Scipy)
* Google Colab
* VSCode

## Dataset Description

This dataset contains the EEG resting state-closed eyes recordings from 88 subjects in total.

Participants:

- 36 of them were diagnosed with Alzheimer's disease (AD group)
- 23 were diagnosed with Frontotemporal Dementia (FTD group)
- 29 were healthy subjects (CN group).

Cognitive and neuropsychological state was evaluated by the international Mini-Mental State Examination (MMSE). MMSE score ranges from 0 to 30, with lower MMSE indicating more severe cognitive decline.

The duration of the disease was measured in months and the median value was 25 with IQR range (Q1-Q3) being 24 - 28.5 months. Concerning the AD groups, no dementia-related comorbidities have been reported.

The average MMSE was:

- For the AD group was 17.75 (sd=4.5)
- For the FTD group was 22.17 (sd=8.22)
- For the CN group was 30.

The mean age:

- AD group was 66.4 (sd=7.9)
- FTD group was 63.6 (sd=8.2)
- CN group was 67.9 (sd=5.4).

Source Dataset:\
[Open Neuro: Alzheimer's disease, Frontotemporal dementia and Healthy subjects](https://openneuro.org/datasets/ds004504/versions/1.0.5)

## Libraries

**OPEN NEURO**\
https://pypi.org/project/openneuro-py/

Installing Open Neuro package:\
Using a local environment with conda requires several packages before install Openneuro. Please make sure you already installed the following packages:
- `conda install -c anaconda jupyter`
- `conda update ipywidgets`
- `conda install -c conda-forge tqdm`

Then, to install Open Neuro can be throught the following commands:
- Pip: `pip install openneuro-py`
- Conda: `conda install -c conda-forge openneuro-py`

**MNE**\
Open-source Python package for exploring, visualizing, and analyzing human neurophysiological data: MEG, EEG, sEEG, ECoG, NIRS, and more.\
https://mne.tools/stable/index.html

- Pip: `pip install mne`
- Conda: `conda install -c conda-forge mne-base`

_Suggestion_: Create a new conda environment to install MNE
