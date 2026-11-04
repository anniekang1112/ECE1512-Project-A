# ECE1512 Project A — Whole-Slide Image Multiple Instance Learning

## Code structure

- `Project_A_Starter_Code/`: original instructor notebook, completed notebooks, ABMIL implementation, unchanged data-loading and WSI visualization utilities, configurations and dataset metadata. `ECE1512_ProjectA_Final.ipynb` is the executed notebook; `ECE1512_ProjectA_Student.ipynb` is the original incomplete starter. Completed model and training functions are shared through `mil_pipeline.py`.
- `experiments/`: baseline and extension training, checkpoint evaluation, controlled prevalence and patient-grouped protocols, validation-based attention-fraction selection, instance subsampling, data download utilities and experiment specifications.
- `analysis/`: numerical aggregation (`summarize_results.py`) followed by graph generation (`make_figures.py`). Generated summaries go to `analysis/generated/`; regenerated plots go to `figures/generated/`.
- `results/`: recorded metrics, predictions, training histories, configurations, split IDs and selection records. The C16 baseline checkpoint is included for notebook replay.
- `figures/`: saved loss curves, comparisons, diagnostic graphs and WSI attention panels.
- `report/`: reserved for the final report supplied separately.

## Environment

Tested with Python 3.14 on macOS arm64, PyTorch 2.14.1 and Apple MPS; CPU inference is supported. Install the recorded training and notebook/visualization environments from the repository root:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/pip install -r requirements-visualization-lock.txt
```

Select this virtual environment when opening the notebooks. Run code from the repository root; notebooks also resolve paths from `Project_A_Starter_Code/`. Pre-extracted features belong under `Project_A_Starter_Code/feature/{C16,C17,BRACS}/patch_feats_pretrain_medical_ssl.h5`. Download URLs, sizes and checksums are defined in `experiments/feature_manifest.json`; optional original slide assets are defined in `experiments/slide_manifest.json`. Large raw inputs are not included.
