# Table Semantic Relationship Extraction (PyTorch)

This version uses PyTorch and HuggingFace Transformers with **DeBERTa-v3-large** as the default encoder.

## Project Structure

- `baseline/`
  - `train.py`: PyTorch training script (competition-weighted loss by default)
  - `optuna_search.py`: Optuna hyperparameter search (objective: validation `Score_final`)
  - `infer.py`: PyTorch inference script
  - `metrics.py`: `Score_final` and class weight helpers
- `dataset/`
  - `Train_Set/`: training CSV files (extract from `Train_Set.zip`)
  - `test.csv`: test dataset
  - `labels.txt`: label set
- `environment/`
  - `requirements.txt`: minimal PyTorch runtime dependencies
- `../../deberta-v3-large/` (sibling under `bd/`, not in git): local pretrained weights

## Environment

```bash
conda create -n bd_2 python=3.10 -y
conda activate bd_2
pip install -r environment/requirements.txt \
  --index-url https://download.pytorch.org/whl/cu121 \
  --extra-index-url https://pypi.org/simple
```

## Download DeBERTa-v3-large (first time)

From any shell with network/proxy configured:

```bash
source ~/.config/proxy.env   # if Hugging Face needs proxy
conda activate bd_2
huggingface-cli download microsoft/deberta-v3-large \
  --local-dir /home/gmn/danger/codes/bd/deberta-v3-large
```

When running from `baseline/`, the default model path is `../../deberta-v3-large`.

Weights are loaded from `pytorch_model.bin` via `torch.load` (compatible with `torch 2.5.x` + `transformers 4.57`). The checkpoint uses a `deberta.` prefix; the training code strips it automatically.

## Train

Run from the `baseline/` directory. Defaults:

- `batch_size=24`, `gradient_accumulation_steps=2` (effective batch **48**), AMP on
- `--loss_type competition`: class-weighted CE aligned with leaderboard few-shot weights
- `--selection_metric score_final`: early stopping and `best_model.pt` by validation `Score_final`

```bash
cd baseline
unzip -o ../dataset/Train_Set.zip -d ../dataset/   # if Train_Set/ not extracted

python train.py \
  --train_dir ../dataset/Train_Set \
  --shortcut_name ../../deberta-v3-large \
  --use_amp \
  --device gpu
```

If GPU OOM, try `--batch_size 16` or `--batch_size 8 --gradient_accumulation_steps 3`.

Revert to the old unweighted loss and accuracy-based early stopping:

```bash
python train.py --loss_type ce --selection_metric acc ...
```

The best checkpoint is saved as:

```text
cpa_output/cpa_YYYYMMDD_HHMMSS/best_model.pt
```

The same output directory also contains `label_classes.txt`, `class_weights.json`, tokenizer files, and `train_args.json`.

**Note:** local `Val Score_final` uses the same formula as the competition leaderboard (weighted per-class accuracy; rare relations in training get higher weight). It is computed on a held-out 10% validation split and may differ from the AI Studio test score.

Optional training flags: `--dropout`, `--weight_decay`, `--run_name` (fixed output dir `cpa_{run_name}` for Optuna trials).

## Optuna hyperparameter search

Run from `baseline/`. Objective: maximize validation **Score_final** (same as `--selection_metric score_final`).

### Search space (per trial)

| Parameter | Suggested range |
|-----------|-----------------|
| `lr` | log-uniform `2e-5` … `1e-4` |
| `batch_size` | 16, 24, 32 |
| `gradient_accumulation_steps` | 1, 2, 3 (effective batch `batch_size × grad_accum` must be 32–64) |
| `warmup_ratio` | uniform `0.05` … `0.12` |
| `max_length` | 128, 160 |
| `dropout` | uniform `0.05` … `0.3` |
| `weight_decay` | log-uniform `1e-3` … `0.1` |

Fixed during search: `loss_type=competition`, `selection_metric=score_final`, `val_ratio=0.1`, `random_seed=42`, `patience=3`, `epoch=20` (early stop applies).

### Run search

```bash
cd baseline
pip install optuna>=3.0   # or reinstall from ../environment/requirements.txt

CUDA_VISIBLE_DEVICES=0 python optuna_search.py \
  --train_dir ../dataset/Train_Set \
  --shortcut_name ../../deberta-v3-large \
  --device gpu \
  --n_trials 20 \
  --pruning \
  --storage sqlite:///./optuna_output/extraction_optuna.db \
  --artefact_root ./optuna_output
```

Each trial writes to `cpa_output/cpa_optuna_trial_XXXX/`. Study summary: `optuna_output/study_summary.json`. Enable `--pruning` for epoch-level `MedianPruner` (requires `trial.report` in `train.py`).

### Retrain best trial and export submission

```bash
CUDA_VISIBLE_DEVICES=0 python optuna_search.py \
  --train-best \
  --infer-after-best \
  --storage sqlite:///./optuna_output/extraction_optuna.db \
  --shortcut_name ../../deberta-v3-large \
  --device gpu \
  --infer-output ../../contest/submission_deberta_optuna.csv
```

Checkpoint: `cpa_output/cpa_optuna_best/best_model.pt`.

## Inference

Use the same `shortcut_name` as training. Old BERT checkpoints are not compatible with DeBERTa.

```bash
python infer.py \
  --input_csv ../dataset/test.csv \
  --labels_path ../dataset/labels.txt \
  --model_path ./cpa_output/cpa_YYYYMMDD_HHMMSS/best_model.pt \
  --shortcut_name ../../deberta-v3-large \
  --max_length 128 \
  --output_file ./submission.csv \
  --use_amp \
  --device gpu
```
