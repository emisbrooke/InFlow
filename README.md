<p align="center">
  <img src="assets/inFlow_logo.JPG" alt="InFlow Logo" width="350"/>
</p>

<h1 align="center">InFlow</h1>
<p align="center"><b>An information-theoretic framework for quantifying transcriptional information flow in aging</b></p>

---

InFlow models transcriptional regulation inside cells using scRNA-seq data. It learns
transcription-factor (TF)–TF and TF–target-gene (TG) couplings through a physics-inspired
conditional maximum-entropy model, computes the mutual information (MI) between TFs and each
target to quantify how communication fidelity decays with age, and performs *rejuvenation swaps*
of TF distributions and gene-regulatory networks (GRNs) to identify the source of information loss
in cellular aging. The model and analysis code live in the `inflow/` folder; runnable examples
live in `examples/`.

---

### Key Features
- **GRN inference:** learns asymmetric TF–TF and TF–TG couplings and captures higher-order interactions.
- **Mutual-information quantification:** measures information flow from TF inputs to a TG output.
- **Rejuvenation swaps:** tests how restoring TF distributions or GRN wiring affects MI recovery.
- **Network structure analysis:** out-degree centralization and adapting-motif (IFFL/IFL) counts across ages.
- **Knock-in / perturbation analysis:** in-silico activation or knock-out of TFs to test restoration and network fragility.
- **Tissue-resolved analysis:** compare information decay and restoration across mouse tissues.

---

### Repository layout

```
inflow/
  model.py                     GeneModel class: builds and trains a conditional MaxEnt model
  inference_funcs_tf.py         TF-model log-likelihood, gradients (Lasso), Adam/Yogi updates
  inference_funcs_tg.py         TG-model equivalents
  train_sweep.py                CLI: train models over tissue/age/gene-type/lambda sweeps
  compare_model_stats.py        CLI: moment-matching (model vs. data means & correlations)
  plot_model_stats_summary.py   CLI: summary plots of model-vs-data statistics
  calc_MI_funcs.py              MI computation and MCMC sampling of the TF stationary distribution
  run_mi_swaps.py               CLI: MI and heterochronic rejuvenation swaps across ages
  get_st_funcs_generalized.py   Network-structure utilities (out-degree, null model, IFFL/IFL)
  analyze_structure_script_generalized.py  CLI: structure analysis driver
  knock_in_funcs.py             MCMC knock-in / knock-out engine (KI, KO, combined)
  knock_in_final.py             Batch driver for per-tissue knock-in experiments
data/
  homologs_csv.csv              Human->mouse ortholog map for the TF list
examples/
  write_filtered_data.ipynb     Data prep: build binarized TF/TG matrices from the atlas
  run_train_sweep_example.sh     Local training example
  run_mi_swaps_example.sh        Local MI-swap example
  run_compare_model_stats.sh     Local moment-matching example
  run_plot_model_stats_summary.sh
  slurm_*.sh                     Slurm array-job versions of the above
  *_params.txt                   Per-job parameter tables for the array jobs
  figs_notebook.ipynb            Main-text figure generation
  structure_analysis_notebook.ipynb  Figure 5 (structure) generation
```

---

### 🚀 Quick Start

```bash
git clone https://github.com/emisbrooke/InFlow.git
cd InFlow
```

Scripts import sibling modules by name, so **run every command from the repository root**
(e.g. `python inflow/train_sweep.py ...`), which puts `inflow/` on the import path.

#### Requirements

- Python ≥ 3.10
- A CUDA-capable GPU is strongly recommended (the MI, swap, and knock-in steps are MCMC-heavy;
  the knock-in functions in `knock_in_funcs.py` currently require CUDA). Training and structure
  analysis also run on CPU.

Install the dependencies:

```bash
pip install torch numpy pandas scipy h5py networkx matplotlib seaborn tqdm statsmodels adjustText scanpy
```

(`scanpy` is only needed for the data-preparation notebook; `statsmodels` and `adjustText`
only for the figure notebooks.)

---

### Data preparation

InFlow trains on the [Tabula Muris Senis](https://figshare.com/projects/Tabula_Muris_Senis/64982)
single-cell atlas (Smart-seq2 in the main text; droplet data for the platform-robustness check).
Use `examples/write_filtered_data.ipynb` to download/load the atlas, filter genes and tissues,
binarize each cell (nonzero count → 1), and split genes into TFs and TGs using
`data/homologs_csv.csv`. The notebook writes one matrix per tissue/age/gene-type into `data/`:

```
data/data_bin_filt_<tissue>_<age>m_<TF|TG>.npy      # shape: n_genes x n_cells
```

All downstream scripts expect exactly this naming convention.

---

### Workflow

The pipeline reproduces the analyses in the paper in five stages. Copy the example scripts in
`examples/` and edit the tissue/age/lambda lists for your experiment.

**1. Train the models** — one TF model and one TG model per tissue/age, over a λ (Lasso) sweep.
Each configuration is trained with several random restarts and the best-likelihood run is kept.

```bash
python inflow/train_sweep.py \
  --data-dir data --out-dir outputs/models \
  --tissues brain liver --ages 3 24 --gene-types TF TG \
  --lambdas 0 1 2 3 4 5 \
  --steps 5000 --optimizer adam --n-restarts 10
# writes outputs/models/model_<tissue>_<age>m_<TF|TG>_lam<tag>.pt
```

Select λ per tissue by moment matching (below), then optionally rerun a finer sweep around the
best value (see the commented block in `examples/run_train_sweep_example.sh`).

**2. Validate by moment matching** — generate in-silico cells from the fitted models and compare
their means and pairwise correlations against the data (the r-values reported in the paper).

```bash
python inflow/compare_model_stats.py \
  --models-dir outputs/models --data-dir data \
  --out-csv outputs/model_stats.csv --plot-dir outputs/model_plots
python inflow/plot_model_stats_summary.py \
  --csv outputs/model_stats.csv --out-dir outputs/model_plots/summary
```

**3. Mutual information and rejuvenation swaps** — compute the MI between all TFs and each TG,
and the heterochronic swap matrix (young/aged TF distribution × young/aged GRN) that separates
input mismatch from channel corruption.

```bash
python inflow/run_mi_swaps.py \
  --tissue brain --ages 3 24 --lambda 1.0 \
  --iters 10 --n-tf-samples 60000 \
  --int-burn 200000 --int-save 1000 --device cuda \
  --out-dir outputs/mi_swaps
```

For every ordered pair of ages this writes per-gene MI, per-gene entropy, and a JSON summary;
the diagonal entries are the true young/aged MI and the off-diagonal entries are the swaps.

**4. Network structure analysis** — out-degree distributions (with a rarefied, density-matched
null model) and counts of adapting motifs (incoherent feedforward loops, integral feedback loops)
for each age.

```bash
python inflow/analyze_structure_script_generalized.py \
  --tissue brain --ages 3 24 \
  --models-dir outputs/models --data-dir data \
  --lambda 1.0 --theta-threshold 0.01 \
  --out-dir outputs/structure
```

**5. Knock-in and perturbation** — `knock_in_funcs.py` provides the MCMC engine for setting TFs
always-on (`knock_in`, `knock_in_many`), always-off (`knock_out`, `knock_out_many`), or both
(`knock_both_many`). These support the master-regulator knock-in analysis (activating top TFs to
restore MI) and the perturbation-stability test (random knock-in/knock-out of a subset of TFs).
`knock_in_final.py` is a per-tissue batch driver over these functions.

The batch driver reads two small CSVs that are **not shipped with the repo** — create them for
your own tissue/age/λ selection. Both are keyed by `tissue`:

- `examples/ages_chosen.csv` — which ages count as young and old per tissue:

  ```csv
  tissue,age_young,age_old
  brain,3,24
  liver,3,24
  ```

- `best_lams_droplet_all.csv` (repo root) — the λ selected by moment matching per tissue:

  ```csv
  tissue,lambda
  brain,1.0
  liver,1.0
  ```

---

### Outputs

By default everything is written under `outputs/`:

```
outputs/models/      trained model_*.pt files (theta, m, metadata)
outputs/model_plots/ moment-matching scatter plots and summaries
outputs/mi_swaps/    per-gene MI / entropy arrays and swap summaries
outputs/structure/   out-degree and motif-count results
outputs/knock_in/    knock-in TF means and L1 restoration scores
```

Figures in the paper are assembled from these outputs in `examples/figs_notebook.ipynb`
and `examples/structure_analysis_notebook.ipynb`.

---
This code can also be found at: [![DOI](https://zenodo.org/badge/1301918291.svg)](https://doi.org/10.5281/zenodo.21383284)

### Citation

If you use InFlow, please cite:

> Emison, B., Lynn, C.W., Mugler, A., Ambrosio, F., and Dixit, P. *An information-theoretic
> framework for transcriptional regulation reveals declining communication fidelity in aging.*

---

### Contact

Questions and issues: open a GitHub issue, or contact the lead contact,
Purushottam Dixit (purushottam.dixit@yale.edu).
