
# Sequence Modeling Benchmarks and Temporal Convolutional Networks (TCN) + Linear-Phase Front-End

This repository extends the original TCN benchmarks by Bai, Kolter, and Koltun with a **learnable, linear-phase (Type-I) Conv1D front-end** that acts as a lightweight **anti-aliasing / shift-stability** layer before the dilated TCN stack.

- Original TCN paper/repo: Bai et al., *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling* (2018).  
- This fork adds a **symmetric Conv1D front-end** (odd length, even symmetry) that guarantees **constant group delay** (linear phase), optionally **residual bypass**, **causal/non-causal** modes, and **unity-DC** constraint.  
- We provide evaluation **metrics and scripts** for **shift-stability** (output sensitivity under ±k-sample shifts) and **HF-energy above the alias band** (≥ π/d for the first dilation), reported alongside task metrics under **1-epoch** and **10-epoch** budgets with **multi-seed mean±std**.

> **Why this helps:** Dilations behave like effective sub-sampling; low-pass filtering before them reduces aliasing and improves shift stability while linear phase preserves waveform shape.

---

## Table of Contents
- [Domains and Datasets](#domains-and-datasets)
- [What’s New in This Fork](#whats-new-in-this-fork)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
  - [Run baseline TCN](#run-baseline-tcn)
  - [Run TCN + Linear-Phase Front-End](#run-tcn--linearphase-front-end)
- [Evaluation & Metrics](#evaluation--metrics)
- [Reproducing Paper-style Tables](#reproducing-paperstyle-tables)
- [Results Folder Layout](#results-folder-layout)
- [Tips & Gotchas](#tips--gotchas)
- [Citations](#citations)
- [License & Acknowledgements](#license--acknowledgements)

---

## Domains and Datasets

All original TCN tasks are supported (see per-task subfolders for details):

- **Adding Problem** (T ∈ {200, 400, 600})
- **Copying Memory** (T ∈ {500, 1000, 2000})
- **Sequential MNIST** and **Permuted Sequential MNIST**
- **JSB Chorales**, **Nottingham** (polyphonic)
- **Penn Treebank** (word-level SMALL; char-level MEDIUM)
- **WikiText-103**, **LAMBADA**, **text8** (LARGE)

Large corpora are fetched with the `observations` package on first run.

**Additional (optional) benches in this fork**
- **pMNIST (sequential, non-permuted)** — shift/stability checks
- **Text classification**: AG News, IMDb, TREC, Yelp
- **Simple 1-D sensor/time-series** (e.g., ECG-like) for HF/shift diagnostics

> See `benchmarks/` for dataset notes, configs, and download helpers.

---

## What’s New in This Fork

**Front-end layer (LPFE = Linear-Phase Front-End)**  
A learnable **symmetric Conv1D** with odd kernel length `K=2k+1`:
- **Even symmetry**: learn half the taps and mirror (parameters cut ≈ 50%).
- **Linear phase**: constant group delay `(K−1)/2` samples (shape-preserving).
- **Modes**: `non-causal` (symmetric padding) or `causal` (left padding; fixed latency).
- **Residual bypass**: `y = x + λ·LPFE(x)` with fixed or learnable `λ`.
- **Unity-DC**: optional ∑ taps = 1 (preserve average level).
- **Depthwise** variant: per-channel smoothing (optional).

**Evaluation & logging**
- **Shift-stability sweep**: run outputs under shifts ±1…±3 samples; log L2/CE deltas.
- **HF-energy**: spectral energy above **π/d** (alias band for first dilation).
- **Budgets**: report **after 1 epoch** and **best within 10 epochs** for fair speed/quality.
- **Seeds**: 3–5 seeds; log **mean±std** for all metrics.

---

## Installation

```bash
# Python 3.9+ recommended
pip install -r requirements.txt
# If you train large LMs, install a recent PyTorch build with CUDA
# (PyTorch ≥1.3 recommended; ≥1.0 works per the original TCN benchmarks).


---

## Repository Structure

```
[TASK_NAME]/
  data/                     # auto-populated as needed
  [TASK_NAME]_test.py       # entry point for the task
  models.py
  utils.py

tcn/                        # TCN backbone (unchanged API)
frontends/                  # NEW: linear-phase front-end (LPFE) modules
  symmetric_conv1d.py
  init.py

benchmarks/                 # NEW: configs + runners for add’l datasets
  text/
  sensors/
eval/                       # NEW: shift-stability + HF-energy metrics
  shift_eval.py
  spectral_metrics.py
scripts/                    # convenience launchers
results/                    # logs, CSVs, figures (created at runtime)
```

---

## Quick Start

### Run baseline TCN

Each task still runs as in the original repo:

```bash
python adding/add_test.py           # Adding problem
python copying/copying_test.py      # Copying Memory
python mnist/mnist_test.py          # (Sequential) MNIST
```

Use `-h` on any script to see task-specific flags.

### Run TCN + Linear-Phase Front-End

Enable the front-end with consistent flags across tasks:

```bash
# Example: Adding (T=400), non-causal LPFE, residual bypass, K=21
python adding/add_test.py \
  --model tcn \
  --lpfe on \
  --lpfe-k 21 \
  --lpfe-mode noncausal \
  --lpfe-residual on \
  --lpfe-lambda 1.0 \
  --lpfe-unity-dc on \
  --eval-shift "1,2,3" \
  --eval-hf auto \
  --epochs 10 --seed 0
```

```bash
# Example: PTB char-level (strictly CAUSAL; no look-ahead leakage)
python ptb_char/ptb_char_test.py \
  --model tcn \
  --lpfe on \
  --lpfe-k 21 \
  --lpfe-mode causal \
  --lpfe-residual off \
  --lpfe-unity-dc on \
  --epochs 10 --seed 0
```

```bash
# Example: text classification (AG News)
python benchmarks/text/agnews_test.py \
  --model tcn \
  --lpfe on --lpfe-k 21 --lpfe-mode noncausal --lpfe-residual on \
  --epochs 10 --seeds 5
```

**Flag glossary**

* `--lpfe {on,off}`: enable/disable linear-phase front-end (default: off)
* `--lpfe-k INT`: odd kernel length (e.g., 11, 21, 31)
* `--lpfe-mode {causal,noncausal}`
* `--lpfe-residual {on,off}` and `--lpfe-lambda FLOAT`
* `--lpfe-unity-dc {on,off}`
* `--lpfe-depthwise {on,off}` (optional)
* `--eval-shift "1,2,3"`: evaluate shift-stability at ±1, ±2, ±3
* `--eval-hf {auto,FLOAT}`: HF band cutoff (auto = π/d from the first dilation)

> **Note:** If your TCN is strided anywhere, the front-end can also precede the first stride.

---

## Evaluation & Metrics

**Task metrics**

* Accuracy/F1 (classification), bpc (language modeling), MSE (synthetic)

**Stability metrics**

* **Shift-stability**: average output change over a shift sweep (±1…±3 samples)
  `eval/shift_eval.py` runs forward passes for each shift and logs L2/CE deltas.
* **HF-energy above alias band**: power above **π/d** for the first TCN dilation
  `eval/spectral_metrics.py` computes FFT energy; `--eval-hf auto` sets the band.

**Reporting protocol**

* **1-epoch** snapshot (optimization speed)
* **10-epoch** budget: final epoch value and **best-within-10** value (+ epoch index)
* **3–5 seeds** with **mean±std**

All metrics are exported as CSV to `results/` and pretty-printed to console.

---

## Reproducing Paper-style Tables

We provide minimal repro scripts under `scripts/`. Examples:

```bash
# 1) Baselines vs LPFE on Adding (T=400)
bash scripts/repro_adding.sh

# 2) Shift-stability & HF-energy on pMNIST
bash scripts/repro_pmnist_stability.sh

# 3) Text classification sweep (AG/IMDb/TREC/Yelp)
bash scripts/repro_text.sh
```

Each script runs:

* **No front-end**
* **Fixed symmetric filters**: MovingAvg(K), Hamming(K), Savitzky–Golay(K,3)
* **Unconstrained Conv1D(K)**
* **LPFE (ours)** with/without residual, causal/non-causal

---

## Results Folder Layout

```
results/
  adding_T400/
    summary_seed*.csv
    shift_sweep.csv
    spectra.csv
    best_within_10.json
  pmnist/
    ...
  text/
    agnews/
      ...
```

CSV fields include: `task`, `variant`, `seed`, `epoch`, task metric(s), `shift_err@±k`, `hf_energy_ratio`, params/FLOPs for the front-end, etc.

---

## Tips & Gotchas

* **Causal tasks (e.g., PTB char LM):** use `--lpfe-mode causal`. Non-causal SAME padding leaks future tokens.
* **Kernel length:** start with `K ∈ {11,21,31}`; longer `K` increases latency in causal mode.
* **Residual path:** enables detail retention when high-frequency content is task-relevant.
* **Unity-DC:** stabilizes training by preserving mean level through the front-end.
* **Depthwise LPFE:** consider for multi-channel inputs with distinct channel statistics.

---

## Citations

If you find this repository useful, please cite:

```bibtex
@article{BaiTCN2018,
  author  = {Shaojie Bai and J. Zico Kolter and Vladlen Koltun},
  title   = {An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling},
  journal = {arXiv:1803.01271},
  year    = {2018}
}

@inproceedings{Zhang2019Antialiased,
  author    = {Richard Zhang},
  title     = {Making Convolutional Networks Shift-Invariant Again},
  booktitle = {ICML},
  year      = {2019},
  note      = {anti-aliased CNNs / shift-stability}
}

@book{OppenheimSchaferDTSP,
  author    = {Alan V. Oppenheim and Ronald W. Schafer},
  title     = {Discrete-Time Signal Processing},
  edition   = {2nd},
  publisher = {Prentice Hall},
  year      = {1999}
}
```

---

## License & Acknowledgements

This repository is distributed under the MIT License.
We acknowledge the original TCN codebase and benchmarks by Bai, Kolter, and Koltun.

```

