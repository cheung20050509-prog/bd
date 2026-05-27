# Table Semantic Relationship Extraction (PyTorch)

This version uses PyTorch and HuggingFace Transformers.

## Project Structure

- `baseline/`
  - `train.py`: PyTorch training script
  - `infer.py`: PyTorch inference script
- `dataset/`
  - `Train_Set/`: training CSV files, one label per file
  - `test.csv`: test dataset
  - `labels.txt`: label set
- `environment/`
  - `requirements.txt`: minimal PyTorch runtime dependencies

## Environment

```bash
conda create -n bd_2 python=3.10 -y
conda activate bd_2
pip install -r environment/requirements.txt \
  --index-url https://download.pytorch.org/whl/cu121 \
  --extra-index-url https://pypi.org/simple
```

## Train

Run from the `baseline/` directory:

```bash
cd baseline
python train.py --train_dir ../dataset/Train_Set --device gpu
```

The best checkpoint is saved as:

```text
cpa_output/cpa_YYYYMMDD_HHMMSS/best_model.pt
```

The same output directory also contains `label_classes.txt`, tokenizer files, and `train_args.json`.

## Inference

```bash
python infer.py \
  --input_csv ../dataset/test.csv \
  --labels_path ../dataset/labels.txt \
  --model_path ./cpa_output/cpa_YYYYMMDD_HHMMSS/best_model.pt \
  --output_file ./submission.csv \
  --device gpu
```
