# Extraction PyTorch Reference: Doduo-Inspired Pair

This directory is a reference experiment based on the Doduo paper in `../nlp/reference`.

The competition data only provides `Subject`, `Object`, and the relation `Label`; it does not provide original table IDs, full table columns, or auxiliary column type labels. Because of that, this implementation is not the full table-wise, multi-task Doduo model. It adapts the most useful part for the available data: learned pseudo-column representations from multiple `[CLS]` tokens.

## Architecture

Each `(Subject, Object)` pair is serialized as two pseudo columns:

```text
[CLS] Subject tokens [SEP] [CLS] Object tokens [SEP]
```

The model takes the contextualized embeddings of both `[CLS]` tokens as the Subject/Object column representations, then predicts the 563-way relation label with one of two pair heads:

- `concat`: `[subject_cls, object_cls]`
- `interaction`: `[subject_cls, object_cls, abs(subject_cls - object_cls), subject_cls * object_cls]`

`tokens_per_column` caps each pseudo-column before packing, matching Doduo's finding that short column token budgets can work well.

## Fixed Trial 9-Style Run

Use the best `Extraction_pytorch` trial 9 hyperparameters as a first baseline:

```bash
cd /home/gmn/danger/codes/bd/Extraction_pytorch_reference/baseline
CUDA_VISIBLE_DEVICES=2 ./run_doduo_trial9.sh
```

The checkpoint will be written to:

```text
baseline/cpa_output/cpa_doduo_pair_trial9_seed42/best_model.pt
```

## Optuna Search

```bash
cd /home/gmn/danger/codes/bd/Extraction_pytorch_reference/baseline
CUDA_VISIBLE_DEVICES=2 N_TRIALS=30 ./run_doduo_optuna.sh
```

The Doduo pair study is isolated from the original Optuna run:

```text
baseline/optuna_output/doduo_pair_optuna.db
```

The search includes:

- `batch_size`: 16, 24, 32
- `gradient_accumulation_steps`: 1, 2, 3
- `max_length`: 128, 160, 192
- `tokens_per_column`: 8, 16, 32
- `pair_head`: `concat`, `interaction`
- `lr`, `warmup_ratio`, `dropout`, `weight_decay`

## Inference

`infer.py` reads `max_length`, `tokens_per_column`, `pair_head`, and dropout from the checkpoint's `train_args.json`.

```bash
cd /home/gmn/danger/codes/bd/Extraction_pytorch_reference/baseline
CUDA_VISIBLE_DEVICES=2 python infer.py \
  --input_csv ../dataset/test.csv \
  --labels_path ./cpa_output/cpa_doduo_pair_trial9_seed42/label_classes.txt \
  --model_path ./cpa_output/cpa_doduo_pair_trial9_seed42/best_model.pt \
  --output_file ../../nlp/submission_doduo_reference.csv \
  --device gpu
```
