# Third-party software and models

This app is **GPL-3.0-or-later** (see `LICENSE`) because its image imports **Bombcell** (GPL-3.0)
in the same process: `bombcell.run_bombcell_qc` decides which metrics Bombcell needs, computes
them on the SortingAnalyzer and runs the labelling. One of Bombcell's dependencies, `cachecache`,
is GPL-3.0-only; combined with this app the image as a whole is distributed under GPL-3.0. The
source of every component is public (PyPI sdists and the repositories below), and this repository
is the source of the app.

The methods are other people's work. Pennsieve packages them and did not develop them.

| Component | What it does here | Authors / copyright | Licence | Source |
|---|---|---|---|---|
| SpikeInterface 0.105.4 | Loading recordings and sortings, the SortingAnalyzer and its extensions, quality and template metrics, `bombcell_label_units`, UnitRefine model loading and prediction, `compare_two_sorters` / `compare_sorter_to_ground_truth`, Phy export | Alessio P. Buccino, Samuel Garcia, Pierre Yger, Cole L. Hurwitz, Jeremy Magland, Matthias H. Hennig and the SpikeInterface contributors | MIT | https://github.com/SpikeInterface/spikeinterface |
| Bombcell 0.94 | `run_bombcell_qc` (metric choice and labelling pipeline) | Julie M. J. Fabre, Enny H. van Beest, Andrew J. Peters, Matteo Carandini, Kenneth D. Harris (UCL) | **GPL-3.0** | https://github.com/Julie-Fabre/bombcell |
| UnitRefine pretrained models: `SpikeInterface/UnitRefine_noise_neural_classifier` @280cd07f, `SpikeInterface/UnitRefine_sua_mua_classifier` @9ee216e7 (scikit-learn pipelines, skops files, sha256 in `scripts/fetch_models.py`) | Noise/neural and SUA/MUA classification | Anoushka Jain, Chris Halcrow, Simon Musall and the UnitRefine authors; trained on Neuropixels 1.0 recordings of mouse V1, SC and ALM | MIT | https://huggingface.co/SpikeInterface/UnitRefine_noise_neural_classifier, https://huggingface.co/SpikeInterface/UnitRefine_sua_mua_classifier, https://github.com/anoushkajain/UnitRefine |
| scikit-learn 1.7.2, skops 0.16.0 | Running the UnitRefine pipelines; loading the skops files | scikit-learn developers; skops developers | BSD-3-Clause; MIT | https://scikit-learn.org, https://github.com/skops-dev/skops |
| pynwb 4.2.0, hdmf 6.2.0 | Writing the NWB Units file | NWB / HDMF developers (LBNL, NWB team) | BSD-3-Clause | https://github.com/NeurodataWithoutBorders/pynwb |
| probeinterface 0.4.2, neo 0.14.6, quantities 0.16.4 | Probe geometry; file readers used by SpikeInterface | SpikeInterface / NeuralEnsemble developers | MIT; BSD-3-Clause; BSD-3-Clause | GitHub / PyPI |
| NumPy 2.5.4, SciPy 1.18.1, pandas 3.1.1, numba 0.68.1 (+ llvmlite 0.50.0), networkx 3.7, h5py 3.16.0, zarr 2.18.7, numcodecs 0.15.2 | Arrays, statistics, tables, JIT kernels, matching, HDF5 and zarr storage | their developers | BSD-3-Clause (NumPy also 0BSD/MIT/Zlib/CC0 parts); numba BSD-2-Clause, llvmlite BSD-2-Clause AND Apache-2.0 WITH LLVM-exception; zarr, numcodecs MIT | PyPI |
| Matplotlib 3.11.2 | Figures | Matplotlib development team | Matplotlib licence (PSF-based) | https://matplotlib.org |
| Dependencies of Bombcell: upsetplot-bombcell 0.11.0, mtscomp 1.0.2, pyarrow 26.0.1, ipywidgets 8.1.9 (+ ipython, traitlets, comm, ...), joblib, tqdm | Installed with Bombcell, not called by this app | their developers | BSD-3-Clause; BSD; Apache-2.0; BSD-3-Clause; BSD-3-Clause; MPL-2.0 AND MIT | PyPI |
| cachecache 1.0.1 | Dependency of Bombcell | its author | GPL-3.0-only | PyPI |
| huggingface_hub 2.4.0 (+ hf-xet) | Downloading the models at build time only | Hugging Face | Apache-2.0 | https://github.com/huggingface/huggingface_hub |
| Other transitive dependencies (pydantic, attrs, requests, urllib3, certifi, packaging, PyYAML, pillow, ...) | Versions in `requirements.txt` | their developers | MIT / BSD / Apache-2.0 / ISC / PSF-2.0 / MPL-2.0 (certifi, tqdm) / MIT-CMU (Pillow) | PyPI |
| python:3.12.11-slim-bookworm | Base image (Debian 12 + CPython 3.12) | PSF; Debian | PSF; Debian packages under their own free licences | https://hub.docker.com/_/python |

Licences were read from each installed package's metadata (`importlib.metadata`) in a clean
environment resolved from `requirements.txt`, and from Bombcell's and UnitRefine's repositories.

## Not bundled

- **Kilosort4** (GPL-3.0; Pachitariu, Sridhar, Pennington and Stringer, HHMI Janelia) is not in
  the image. This app reads Kilosort's output files (`spike_times.npy`, `spike_clusters.npy`,
  `cluster_KSLabel.tsv`) through SpikeInterface.
- **Phy** (BSD-3-Clause; Cyrille Rossant and the cortex-lab, UCL) is not in the image; with
  `EXPORT_PHY=true` the app writes a folder Phy opens.
- **Demo data** (not in the image or the repository): IBL Brain Wide Map, DANDI:000409 (CC-BY-4.0,
  International Brain Laboratory); Laquitaine et al. ground-truth simulation, DANDI:001250
  (CC-BY-4.0).
