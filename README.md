# Table Semantic Relationship Extraction (PyTorch)

This version uses PyTorch and HuggingFace Transformers with **DeBERTa-v3-large** as the default encoder.

## Project Structure

- `baseline/`
  - `train.py`: PyTorch training script
  - `infer.py`: PyTorch inference script
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

Run from the `baseline/` directory. Defaults: `batch_size=24`, `gradient_accumulation_steps=2` (effective batch **48**), AMP on.

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

The best checkpoint is saved as:

```text
cpa_output/cpa_YYYYMMDD_HHMMSS/best_model.pt
```

The same output directory also contains `label_classes.txt`, tokenizer files, and `train_args.json`.

**Note:** validation accuracy is not the same as the competition `Score_final` on AI Studio.

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
