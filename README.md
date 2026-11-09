# ephys-curate

This app curates spike-sorted extracellular recordings with three open-source tools built by
others. **[SpikeInterface](https://github.com/SpikeInterface/spikeinterface)** (Buccino et al.
2020, [eLife](https://doi.org/10.7554/eLife.61834)), developed by Alessio Buccino, Samuel Garcia,
Pierre Yger, Cole Hurwitz, Jeremy Magland, Matthias Hennig and the SpikeInterface team, computes
the waveforms, unit locations and quality metrics and compares sortings.
**[Bombcell](https://github.com/Julie-Fabre/bombcell)** ([Zenodo](https://doi.org/10.5281/zenodo.8172821)),
developed by Julie Fabre, Enny van Beest, Andrew Peters, Matteo Carandini and Kenneth Harris at
University College London, labels units with interpretable thresholds.
**[UnitRefine](https://github.com/anoushkajain/UnitRefine)** (Jain et al. 2025,
[bioRxiv](https://doi.org/10.1101/2025.03.30.645770)), developed by Anoushka Jain, Chris Halcrow,
Simon Musall and colleagues together with the SpikeInterface team, supplies the pretrained
classifiers. Over-split units are merged first with SpikeInterface's implementation of
**[SLAy](https://github.com/saikoukunt/SLAy)** (Koukuntla et al. 2025,
[bioRxiv](https://doi.org/10.1101/2025.06.20.660590)), developed by Sai Koukuntla, Adam Charles
and colleagues (Johns Hopkins University and HHMI Janelia). The demo data are the [International Brain Laboratory](https://www.internationalbrainlab.com)'s
Brain Wide Map ([DANDI:000409](https://doi.org/10.48324/dandi.000409/0.260309.1324)) and the
ground-truth simulation of Laquitaine, Imbeni, Tharayil, Isbister and Reimann (Blue Brain Project,
EPFL; [DANDI:001250](https://doi.org/10.48324/dandi.001250/0.261108.1908)). **Pennsieve packages
these methods as a workflow step; it did not develop them.** Please cite them (see Citations).

**What it answers:** which of the units a spike sorter found are well-isolated single neurons,
which are multi-unit activity, and which are noise; where in the brain each one sits; and, when a
reference exists, how well the sorting agrees with it.

It is the third step of the **Neuropixels Spike Sorting** workflow,
`ephys-prep → spike-sort → ephys-curate → neuro-report`, and its NWB output feeds
`neuro-align` and the decoding workflow.

## Why, for a neuroscientist

A spike sorter returns clusters, not neurons. On a typical Neuropixels probe a third to a half of
them are not usable single units: some are noise (artefacts, out-of-brain channels), many are
multi-unit activity (several neurons mixed, or a neuron whose small spikes are partly lost under
the noise). Labs used to sort these by eye in Phy, which takes hours per probe and differs between
people. This app gives every unit two independent automated opinions and a combined label:

- **Bombcell** applies thresholds you can read and argue with: waveform shape (number of peaks,
  trough-to-peak duration, baseline flatness) for noise; amplitude, SNR, refractory-period
  contamination, fraction of missed spikes, presence through the recording and drift for
  multi-unit; and waveform shape again for non-somatic (axonal) spikes.
- **UnitRefine** is a pair of random-forest classifiers trained on units that several experts
  labelled by hand (Neuropixels 1.0, mouse visual cortex, superior colliculus and ALM): first
  noise vs neural, then single (SUA) vs multi-unit (MUA).

A unit is **good** only when both call it a single unit. The output is an NWB file you can
upload to DANDI, with spike times on the same clock as the session's behaviour and trials.

## For a computational analyst

Per session, streaming (one session in memory at a time; SpikeInterface reads the recording
lazily in 1 s chunks):

1. **Inputs.** The recording (`si.load` of ephys-prep's binary_folder; probe, channel locations,
   channel property `brain_area`, times) and the sorting (`si.load` of spike-sort's numpy_folder,
   or `read_kilosort` of a Kilosort/Phy folder), matched on `session_id`. The sorting must index
   samples of this recording (same rate, no spikes beyond its end); spikes in the last samples
   are trimmed (`remove_excess_spikes`) and units with fewer than 2 spikes dropped.
2. **Auto-merge** (`AUTO_MERGE`, default on): sorters, Kilosort4 among them, split some neurons
   into two or more units (for instance when the spike amplitude drifts). SpikeInterface's
   `compute_merge_unit_groups(preset="slay")`, its implementation of SLAy (template similarity plus a
   cross-correlogram significance score; Koukuntla et al. report that SLAy recovers most of the merges
   human curators make), runs on a light analyzer (sampled spikes, noise levels, templates); the
   groups are applied to the sorting (`apply_merges_to_sorting`, spikes closer than 0.3 ms within a
   merged unit removed as double counts, the merged unit keeps the first unit's id) and recorded in
   `merges.csv` and the `merged_from` column. Everything below runs on the merged sorting.
3. **SortingAnalyzer**, sparse (`radius_um=SPARSITY_RADIUS_UM`, 100 µm), with the extensions of
   the AIND ephys pipeline (Buccino et al. 2026): `random_spikes` (`MAX_SPIKES_PER_UNIT`, 500),
   `noise_levels`, `waveforms` (−2/+3 ms), `templates` (mean, std), `spike_amplitudes`,
   `amplitude_scalings` (without collision handling: with it, it took 9× longer and ran out of 8 GB
   on 449 units of the IBL probe), `unit_locations` (monopolar triangulation), `spike_locations` (grid
   convolution), `correlograms` (±25 ms, 1 ms bins), `template_similarity`, `template_metrics`
   (with multi-channel metrics), `principal_components` (5 PCs per channel, `PCA_METRICS`) and
   `quality_metrics`: num_spikes, firing_rate, presence_ratio, snr, isi_violation, rp_violation,
   sliding_rp_violation, amplitude_cutoff, amplitude_median, amplitude_cv, firing_range, drift,
   sd_ratio, synchrony, noise_cutoff, and with `PCA_METRICS` isolation distance, L-ratio, d′,
   nearest-neighbour hit/miss rates and silhouette. The PCA metrics use SpikeInterface's functions
   and neighbourhoods but are computed one unit at a time over forked workers: SpikeInterface's own
   `quality_metrics` prepares every unit's PCA neighbourhood at once, which ran out of 4–8 GB of
   memory on 238 units (the values are identical; `tests/` checks it). They are in `units.csv` and the
   NWB file, not in the analyzer's `quality_metrics` extension. Presence-ratio bins and drift intervals are
   Bombcell's 60 s, shortened for windows under 10 min so there are at least 10 bins / 5
   intervals.
4. **Bombcell** (`bombcell.run_bombcell_qc`, which labels with SpikeInterface's
   `bombcell_label_units`) with Bombcell's default thresholds (`bombcell_thresholds.json` in the
   output), two changes: the distance metrics (isolation distance, L-ratio) are left out, as
   Bombcell itself does by default for acute recordings; and in windows shorter than 10 minutes the
   minimum spike count (300) becomes 0.5 Hz × duration, because 300 spikes in a 60 s window would
   reject every neuron firing below 5 Hz. `BOMBCELL_THRESHOLDS` overrides any threshold.
5. **UnitRefine**: the two pretrained classifiers (`SpikeInterface/UnitRefine_noise_neural_classifier`
   and `..._sua_mua_classifier`, baked into the image at pinned commits) applied as a cascade, as
   SpikeInterface's `unitrefine_label_units` does, on the analyzer's metrics. Features a model
   needs that were not computed (the PCA metrics with `PCA_METRICS=false`) are passed as missing
   and imputed by the model's own imputer, with a warning.
6. **Final label** (`LABEL_RULE`, below).
7. **Times.** Spike times in seconds are the recording's own sample times (`recording.get_times()`:
   ephys-prep's saved timestamps, e.g. IBL's sync-corrected NWB timestamps, or `t_start + n/fs`),
   never re-zeroed. If the saved recording has lost its times (it starts at 0 s while
   session.json says the window starts later), the times are rebuilt linearly from session.json's
   `t_start`/`t_stop`, with a warning.
8. **Reference / ground truth** (optional): the reference's spikes (NWB Units table, or a CSV),
   restricted to the recording's probe (a `probe_name`/electrode-group column matching session.json)
   and window, are put on the recording's samples through the same times (nearest sample), units
   with fewer than `REFERENCE_MIN_SPIKES` spikes in the window dropped, and compared with
   `compare_two_sorters` (reference sorting) or `compare_sorter_to_ground_truth` (ground truth),
   spikes matching within `MATCH_DELTA_MS`. A unit is matched at agreement ≥ 0.5; a ground-truth
   unit is well detected at accuracy ≥ 0.8.
9. **Outputs** written, the raw waveform snippets and PCA models deleted from the analyzer
   (they are gigabytes, and the PCA models are pickled objects zarr refuses to load), the analyzer
   saved as zarr.

### Labels

| `label` | When (`LABEL_RULE=consensus`, default) |
|---|---|
| `good` | Bombcell `good` **and** UnitRefine `sua`: both call it a well-isolated single unit. |
| `noise` | At least one method calls it `noise`, and neither calls it a single unit (Bombcell `good` / UnitRefine `sua`). |
| `mua` | Everything else: multi-unit activity, Bombcell's `non_soma` (axonal) units, and units where one method says noise but the other calls it a single unit (kept rather than thrown away, so a reviewer can look). |

`LABEL_RULE=bombcell` uses Bombcell alone (`non_soma` → `mua`), `LABEL_RULE=unitrefine` UnitRefine
alone (`sua` → `good`). When UnitRefine is off or fails, Bombcell's label is used. The individual
labels stay in the table, so any other rule can be applied downstream.

## Ports

| Port | Kind | Content |
|---|---|---|
| `recording` | input (subfolder) | ephys-prep: `ephys-prep/<session_id>/recording/` (SpikeInterface binary_folder) + `session.json`. Any SpikeInterface-saved recording folder also works. |
| `sorting` | input (subfolder) | spike-sort: `spike-sort/<session_id>/sorting/` (numpy_folder) + `sort.json` (+ `sorter_output/`, read for Kilosort's `KSLabel`); or a Kilosort / Phy folder. |
| `reference` | input (subfolder, optional) | NWB with a Units table (IBL processed NWB, ground-truth simulation, a lab's earlier curation), or CSV `unit_id, spike_time` (session seconds). |
| `units` | output | Everything below, under `ephys-curate/` (the whole of `OUTPUT_DIR`). |

Without links everything may sit side by side at the root of `INPUT_DIR`; files are found by name
and content, recursively, following symlinks. Sessions: `session.json` / `sort.json` `session_id`,
else `sessions.csv`, else `sub-<a>_ses-<b>` in a name, else the folder name; one recording and one
sorting are paired even when their names differ (with a warning).

## Parameters

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `AUTO_MERGE` | boolean | `true` | Merge over-split units first (SLAy preset); merges in `merges.csv`. |
| `LABEL_RULE` | string | `consensus` | `consensus`, `bombcell` or `unitrefine` (see Labels). |
| `PCA_METRICS` | boolean | `true` | PCA-based metrics (UnitRefine's models were trained with them). The slowest step on large sortings; `false` imputes them for UnitRefine. |
| `UNITREFINE` | boolean | `true` | Run UnitRefine. |
| `BOMBCELL_THRESHOLDS` | string (JSON) | empty | Overrides, e.g. `{"mua": {"snr": {"greater": 3}, "drift_ptp": null}}`. |
| `SPARSITY_RADIUS_UM` | number | `100` | Channels within this radius of the peak channel are used per unit. |
| `MAX_SPIKES_PER_UNIT` | integer | `500` | Spikes sampled per unit for waveforms/templates/PCA. |
| `EXPORT_PHY` | boolean | `false` | Write `<session_id>_phy/` (no copy of the raw data; point Phy's `dat_path` at your recording). |
| `GALLERY_UNITS` | integer | `12` | Units in the gallery figure; `0` = none. |
| `REFERENCE_KIND` | string | `auto` | `auto`, `sorting` or `ground_truth`. |
| `REFERENCE_MIN_SPIKES` | integer | `10` | Reference units with fewer spikes in the window are not scored. |
| `MATCH_DELTA_MS` | number | `0.4` | Spike-matching tolerance. |
| `SEED` | integer | `42` | Seed for spike sampling, noise levels and spike locations. |
| `THREADS` | integer | `0` | Parallel jobs; `0` = from the cgroup CPU quota. |

Bad parameters exit 2; no usable input or every session failed exit 1; a failing session is
logged and skipped.

## Outputs (`OUTPUT_DIR/ephys-curate/`)

| File | Contents |
|---|---|
| `<session_id>_units.nwb` | NWB file (session identity, subject and `session_start_time` copied from session.json) with an electrodes table (all channels, `location` = brain area, `rel_x`, `rel_y`) and the **Units** table: `spike_times` (session seconds), `obs_intervals` (the recording's window), `electrodes` (peak channel), `electrode_group`, `waveform_mean` (peak channel, 81 samples centred on the trough, µV), and the columns of `units.csv`. |
| `units.csv` | One row per unit and session: `session_id, unit_name, peak_channel, brain_area, x, y, z, depth` (µm, monopolar triangulation; depth = position along the probe), `peak_channel_x/y`, every quality metric, every template metric, `bombcell_label, bombcell_reasons` (failed thresholds), `unitrefine_label, unitrefine_probability, unitrefine_noise_probability, label`, `merged_from` (`7;12` when auto-merge joined units 7 and 12), `kslabel` (Kilosort's own label),
`reference_match, reference_agreement` and the matched reference unit's own labels as
`reference_<column>` (IBL: `reference_ibl_quality_score`, `reference_kilosort2_label`) (with a reference), `subject_id, condition, group` (from sessions.csv). |
| `<session_id>_analyzer.zarr` | The SpikeInterface SortingAnalyzer (templates, metrics, locations, amplitudes, correlograms, ...; without raw waveforms and PCA models). `si.load_sorting_analyzer(path)`; reopen it with the recording to recompute anything. |
| `merges.csv` | `session_id, unit_name, merged_unit_names, n_units_merged, n_spikes`: one row per merged unit (`AUTO_MERGE=true`). |
| `<session_id>_bombcell/` | Bombcell's `labeling_results_wide.csv`, `labeling_results_narrow.csv` (pass/fail per metric), `thresholds.json`. |
| `<session_id>_phy/` | With `EXPORT_PHY=true`. |
| `<session_id>_reference_units.csv`, `<session_id>_groundtruth_performance.csv` | With a reference: per reference unit the best-matching unit and agreement; for ground truth, accuracy / recall / precision per true unit. |
| `<session_id>_unit_yield.png`, `_metrics.png`, `_unit_gallery.png`, `_depth_map.png`, `_agreement.png` | Report figures; captions in `summary.json`. |
| `sessions.csv`, `summary.json`, `provenance.json`, `bombcell_thresholds.json` | Per conventions.md. `summary.json` per session: unit counts per label and method, agreement between the methods, units per region, median SNR / rate of good units, time source, reference agreement. |

## Results on the demo data

RESULTS_PLACEHOLDER

## Runtime

RUNTIME_PLACEHOLDER

## Running it locally

The image is Lambda-safe (read-only root, any unprivileged user, writable caches under `$TMPDIR`),
but the step is sized for standard compute: curating a 10-minute Neuropixels session takes longer
than Lambda's 15 minutes. Tested with `docker run --read-only --user 65534:65534 --tmpfs /tmp`.


```sh
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
make test PYTHON=.venv/bin/python    # downloads the models to models/, runs the tests (no Docker)
make run PYTHON=.venv/bin/python     # Docker over the synthetic fixture -> data/output/
```

By hand:

```sh
docker build --platform linux/amd64 -t ephys-curate .
docker run --rm --platform linux/amd64 --network none --memory 16g \
  -e INPUT_DIR=/data/input -e OUTPUT_DIR=/data/output \
  -v "$PWD/in/recording:/data/input/recording:ro" -v "$PWD/in/sorting:/data/input/sorting:ro" \
  -v "$PWD/out:/data/output" ephys-curate
```

## Limitations

- **Automated labels are opinions, not truth.** Bombcell's thresholds and UnitRefine's training
  labels encode particular labs' judgement on particular probes (UnitRefine: Neuropixels 1.0 in
  mouse V1, SC and ALM). On other probes, species, brain regions or simulated data they can be
  systematically off (on the ground-truth simulation UnitRefine calls most units noise; see
  Results). Look at the gallery and metric figures, and check a sample of units by eye before you
  trust the counts.
- **Short windows bias the metrics.** Presence ratio, drift and refractory-period contamination
  need many spikes; in a few minutes of recording low-rate neurons cannot pass them (Bombcell's
  sliding refractory test returns "not conclusive", which counts as a failure). Curate the longest
  stretch you have.
- `good` means *isolated*, not *responsive* or *healthy*, and `mua` units still carry information
  for population decoding; drop only `noise` for those analyses.
- The **brain area** is the area of the unit's peak channel as ephys-prep received it (e.g. IBL's
  histology); errors in probe localisation pass through.
- Agreement with a **reference sorting** measures consistency, not correctness: two sorters can
  agree on the same mistake. Only a ground truth measures accuracy, and only in simulation. No
  sorting is perfect even there: in the detailed cortical simulation used here, Laquitaine et al.
  found that sorters isolate only a small fraction of the neurons near the probe (about 10 %),
  biased towards large, active neurons; "good" means the unit is likely one neuron, not that
  every neuron was found.
- Auto-merge can join two different neurons with similar waveforms on the same channels, and leaves
  splits it is not sure about; `merges.csv` lists what it did, and `AUTO_MERGE=false` turns it off.
- The sorting must come from exactly the recording given (same samples). A Kilosort run on the full
  file cannot be curated against a window of it.
- The analyzer is saved without raw waveform snippets and PCA models; reopen it with the recording
  (`analyzer.set_temporary_recording`) to recompute those.

## Licences

**GPL-3.0-or-later** (see `LICENSE`): the image imports **Bombcell** (GPL-3.0) in-process
(`bombcell.run_bombcell_qc`), so this app is licensed compatibly. SpikeInterface is MIT; the
UnitRefine models are MIT. Every bundled component, its authors and licence: `THIRD_PARTY.md`.

## Citations

Please cite these works if you use this app. Verified on Crossref/DataCite and PubMed; full author
lists in `CITATION.cff`.

Bombcell (as its authors ask):

> Fabre JMJ, Van Beest EH, Peters AJ, Carandini M, Harris KD. **Bombcell: automated curation and
> cell classification of spike-sorted electrophysiology data.** Zenodo (2023).
> https://doi.org/10.5281/zenodo.8172821

SLAy (auto-merge):

> Koukuntla S, DeWeese T, Cheng A, Mildren R, Lawrence A, Graves AR, Cullen KE, Colonell J,
> Harris TD, Charles AS. **SLAy-ing oversplitting errors in high-density electrophysiology spike
> sorting.** bioRxiv (2025). https://doi.org/10.1101/2025.06.20.660590, PMID 40666932.

UnitRefine (as its authors ask), and its pretrained models:

> Jain A, Greene R, Halcrow C, Swann JA, Kleinjohann A, Spurio F, Graff S, Pan-Vazquez A, Kampa B,
> Gall J, Grün S, Winter O, Buccino A, Hennig MH, Musall S. **UnitRefine: A Community Toolbox for
> Automated Spike Sorting Curation.** bioRxiv (2025). https://doi.org/10.1101/2025.03.30.645770

> Musall S, Jain A. **UnitRefine.** figshare (2025). https://doi.org/10.6084/m9.figshare.28282841.v2

SpikeInterface:

> Buccino AP, Hurwitz CL, Garcia S, Magland J, Siegle JH, Hurwitz R, Hennig MH.
> **SpikeInterface, a unified framework for spike sorting.** eLife 9:e61834 (2020).
> https://doi.org/10.7554/eLife.61834, PMID 33170122.

The AIND pipeline whose postprocessing and curation defaults this app follows:

> Buccino AP, Sridhar A, Feng D, Svoboda K, Siegle JH. **Efficient and reproducible pipelines for
> spike sorting large-scale electrophysiology data.** eLife 15:RP110170 (2026).
> https://doi.org/10.7554/eLife.110170, PMID 42554034.

Quality metrics (refractory-period violations, missed spikes):

> Hill DN, Mehta SB, Kleinfeld D. **Quality Metrics to Accompany Spike Sorting of Extracellular
> Signals.** J Neurosci 31(24):8699-8705 (2011). https://doi.org/10.1523/JNEUROSCI.0971-11.2011,
> PMID 21677152.

Kilosort (when the sorting is Kilosort's):

> Pachitariu M, Sridhar S, Pennington J, Stringer C. **Spike sorting with Kilosort4.** Nat Methods
> 21(5):914-921 (2024). https://doi.org/10.1038/s41592-024-02232-7, PMID 38589517.

NWB:

> Rübel O, Tritt A, Ly R, Dichter BK, et al. **The Neurodata Without Borders ecosystem for
> neurophysiological data science.** eLife 11:e78362 (2022). https://doi.org/10.7554/eLife.78362,
> PMID 36193886.

Demo data and reference sortings:

> International Brain Laboratory. **IBL - Brain Wide Map.** DANDI Archive (2026).
> https://doi.org/10.48324/dandi.000409/0.260309.1324 (CC-BY-4.0)

> International Brain Laboratory, Meshulam L, Angelaki D, et al. **A brain-wide map of neural
> activity during complex behaviour.** Nature 645:177-191 (2025).
> https://doi.org/10.1038/s41586-025-09235-0, PMID 40903598.

> International Brain Laboratory. **Spike sorting pipeline for the International Brain
> Laboratory.** figshare (2024). https://doi.org/10.6084/m9.figshare.19705522

> Laquitaine S, Imbeni M, Tharayil J, Isbister J, Reimann M. **Spike sorting biases and information
> loss in a detailed cortical model.** DANDI Archive (2026).
> https://doi.org/10.48324/dandi.001250/0.261108.1908 (CC-BY-4.0); and bioRxiv (2024),
> https://doi.org/10.1101/2024.12.04.626805

Phy (with `EXPORT_PHY=true`): https://github.com/cortex-lab/phy (Cyrille Rossant and the
cortex-lab, UCL).

## Acknowledgements

Bombcell is developed by Julie Fabre and colleagues in the Carandini–Harris lab at UCL;
UnitRefine by Anoushka Jain, Simon Musall and colleagues with the SpikeInterface team;
SpikeInterface by its community of contributors. The demo recordings are from the **International Brain Laboratory**
(supported by the Simons Foundation and Wellcome) and the ground-truth simulation from Laquitaine
et al. (Blue Brain Project, EPFL), both shared through the **DANDI Archive** (NIH BRAIN
Initiative) in **Neurodata Without Borders** format. We thank all of them for making their
methods, models and data openly available.
