import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer


def load_pretrained_encoder(model_name):
    path = resolve_model_path(model_name)
    config = AutoConfig.from_pretrained(path if os.path.isdir(path) else model_name)
    weights_path = os.path.join(path, 'pytorch_model.bin') if os.path.isdir(path) else None
    if weights_path and os.path.isfile(weights_path):
        encoder = AutoModel.from_config(config)
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        prefix = 'deberta.'
        encoder_sd = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        encoder_sd.pop('embeddings.position_embeddings.weight', None)
        encoder.load_state_dict(encoder_sd, strict=False)
        print(f'Loaded encoder weights from {weights_path}')
        return encoder
    try:
        return AutoModel.from_pretrained(path if os.path.isdir(path) else model_name)
    except OSError:
        print(f'Pretrained weights for {model_name} not found; initialize model from config.')
        return AutoModel.from_config(config)


# ==========================================
# 1. Model definition (must match training)
# ==========================================
class CPAModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder = load_pretrained_encoder(model_name)
        self.dropout = nn.Dropout(0.1)

        hidden_size = getattr(self.encoder.config, 'hidden_size', None)
        if hidden_size is None:
            raise ValueError('Unable to infer hidden_size automatically. Please check the pretrained model.')

        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
        cls_embedding = sequence_output[:, 0, :]
        logits = self.classifier(self.dropout(cls_embedding))
        return logits


# ==========================================
# 2. Tokenization helper
# ==========================================
def encode_pair(tokenizer, text_a, text_b, max_length):
    encoding = tokenizer(
        text_a,
        text_b,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
    )
    return (
        torch.tensor(encoding['input_ids'], dtype=torch.long),
        torch.tensor(encoding['attention_mask'], dtype=torch.long),
    )


# ==========================================
# 3. Single-table inference dataset
# ==========================================
class SingleTableInferenceDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        # Read the CSV file.
        df = pd.read_csv(csv_path, low_memory=False, encoding='utf-8-sig')

        # Normalize column names by trimming whitespace.
        df.columns = [str(col).strip() for col in df.columns]

        # Locate Subject and Object columns in a case-insensitive way.
        subject_col = None
        object_col = None
        for col in df.columns:
            if col.lower() == 'subject':
                subject_col = col
            elif col.lower() == 'object':
                object_col = col

        if subject_col is None or object_col is None:
            raise ValueError("The CSV file must contain 'Subject' and 'Object' columns (case-insensitive).")

        # Only rows with valid Subject/Object values receive predictions.
        valid_df = df[[subject_col, object_col]].dropna()
        for original_idx, row in valid_df.iterrows():
            self.samples.append((original_idx, str(row[subject_col]), str(row[object_col])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        original_idx, subject_text, object_text = self.samples[idx]
        input_ids, attention_mask = encode_pair(
            self.tokenizer,
            subject_text,
            object_text,
            self.max_length,
        )
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'orig_idx': torch.tensor(original_idx, dtype=torch.long),
        }


def collate_fn(samples):
    return {
        'input_ids': torch.stack([s['input_ids'] for s in samples]),
        'attention_mask': torch.stack([s['attention_mask'] for s in samples]),
        'orig_idx': torch.stack([s['orig_idx'] for s in samples]),
    }


# ==========================================
# 4. Device helper
# ==========================================
def resolve_model_path(model_name):
    if os.path.isdir(model_name):
        return os.path.abspath(model_name)
    path = os.path.abspath(os.path.expanduser(model_name))
    if os.path.isdir(path):
        if not os.path.isfile(os.path.join(path, 'config.json')):
            raise FileNotFoundError(
                f'Model directory exists but config.json is missing: {path}'
            )
        return path
    return model_name


def resolve_device(device_arg):
    requested = (device_arg or '').lower()
    if requested.startswith('gpu'):
        requested = requested.replace('gpu', 'cuda', 1)

    if requested.startswith('cuda'):
        if torch.cuda.is_available():
            device = torch.device(requested)
            print(f'Using requested device: {device}')
            return device
        print(f'Failed to set requested device {device_arg}: CUDA is not available.')

    if requested == 'cpu':
        device = torch.device('cpu')
        print(f'Using requested device: {device}')
        return device

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using default device: {device}')
        return device

    device = torch.device('cpu')
    print('Falling back to CPU.')
    return device


# ==========================================
# 5. Inference pipeline
# ==========================================
def run_inference(args):
    device = resolve_device(args.device)

    # Load label mapping.
    with open(args.labels_path, 'r', encoding='utf-8-sig') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    id2label = {idx: label for idx, label in enumerate(classes)}

    model_name = resolve_model_path(args.shortcut_name)

    # Initialize tokenizer and model.
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = CPAModel(model_name, len(classes)).to(device)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f'Model file not found: {args.model_path}')

    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Load the dataset.
    dataset = SingleTableInferenceDataset(args.input_csv, tokenizer, args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == 'cuda',
    )

    print(f'Starting inference. Total valid rows: {len(dataset)}')
    predictions = {}
    use_amp = args.use_amp and device.type == 'cuda'

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Running inference'):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            orig_indices = batch['orig_idx'].cpu().tolist()

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(ids, mask)

            preds = torch.argmax(logits, dim=1).cpu().tolist()
            for original_row_idx, pred_idx in zip(orig_indices, preds):
                predictions[original_row_idx] = id2label[pred_idx]

    # Reload the original CSV and attach predictions.
    original_df = pd.read_csv(args.input_csv, low_memory=False, encoding='utf-8-sig')
    original_df.columns = [str(col).strip() for col in original_df.columns]

    subject_col = None
    object_col = None
    for col in original_df.columns:
        if col.lower() == 'subject':
            subject_col = col
        elif col.lower() == 'object':
            object_col = col

    if subject_col is None or object_col is None:
        raise ValueError("The CSV file must contain 'Subject' and 'Object' columns (case-insensitive).")

    original_df['Label'] = None
    for row_idx, pred_label in predictions.items():
        original_df.loc[row_idx, 'Label'] = pred_label

    # Save the result without modifying the source file.
    original_df.to_csv(args.output_file, index=False, encoding='utf-8-sig')
    print(f'Inference completed. Results saved to: {args.output_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, default="../dataset/test.csv")
    parser.add_argument('--labels_path', type=str, default="../dataset/labels.txt")
    parser.add_argument('--model_path', type=str, default="./cpa_output/cpa_YYYYMMDD_HHMMSS/best_model.pt")
    parser.add_argument('--output_file', type=str, default='./submission.csv')
    parser.add_argument('--shortcut_name', type=str, default='../../deberta-v3-large')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--no_amp', action='store_false', dest='use_amp')
    args = parser.parse_args()
    run_inference(args)
