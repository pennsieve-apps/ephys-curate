# Sample input

ephys-curate needs, per session, a preprocessed recording and a sorting of that same recording.

## In a workflow

`ephys-prep → spike-sort → ephys-curate`: link ephys-prep's `recording` output to the `recording`
input and spike-sort's `sorting` output to the `sorting` input. Optionally link a `reference`.

## Demo session (IBL, DANDI:000409, CC-BY-4.0)

Session `sub-NR-0031_ses-caa5dddc`, Neuropixels 1.0 probe `Probe01`, 10 minutes (899.67–1499.67 s
on the session clock):

1. Raw: `sub-NR-0031_ses-caa5dddc-9290-4e27-9f5e-575ba3598614_desc-raw_ecephys.nwb` (52.5 GB on
   DANDI; the demo stages a 10-minute cut, `scripts/stage-demo-data.py ibl` in the
   workflow-systems-neuroscience repository) → run `ephys-prep` (keep the NWB timestamps) →
   `spike-sort`.
2. Reference: the session's processed file
   `sub-NR-0031_ses-caa5dddc-9290-4e27-9f5e-575ba3598614_desc-processed_behavior+ecephys.nwb`
   (0.51 GB; IBL's own Kilosort 2.5 sorting, both probes; ephys-curate keeps Probe01's units and
   the spikes inside the window).

```sh
dandi download "https://dandiarchive.org/dandiset/000409/0.260309.1324/files?location=sub-NR-0031"
```

## Ground truth (DANDI:001250, CC-BY-4.0)

`sub-demo-npx-bio-evoked/sub-demo-npx-bio-evoked_ecephys.nwb` (0.62 GB, 60 s, 120 channels,
20 kHz): a biophysically detailed simulation with 1,836 known neurons (Laquitaine et al.). Run it
through `ephys-prep` and `spike-sort`, and link the same NWB file as `reference`: its Units table
is the ground truth (REFERENCE_KIND=auto recognises it from the file's description).

## Your own Kilosort / Phy output

Put the Kilosort output folder (`params.py`, `spike_times.npy`, `spike_clusters.npy`,
`cluster_KSLabel.tsv`) in `sorting`, and the recording it was sorted from, preprocessed and saved
with SpikeInterface (`recording.save(folder=...)`, with its times), in `recording`. Name both with
the same `sub-<subject>_ses-<session>` or add a `sessions.csv`; with one of each they are paired
anyway. The sorting must have been made on exactly this recording (same samples).

## Synthetic

`python -m tests.make_fixture OUT` writes a 60 s, 32-channel session with 8 good, 1 contaminated
and 3 noise units and its ground truth (needs only numpy).
