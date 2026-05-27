import argparse
import json
import logging
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


# data load
def load_data_from_directory(dir_path):
    all_data = []
    if not os.path.exists(dir_path):
        raise ValueError(f"can't find: {dir_path}")

    csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
    logging.info(f"load data from {dir_path} ...")

    for filename in tqdm(csv_files, desc=f"loading {os.path.basename(dir_path)}"):
        file_path = os.path.join(dir_path, filename)
        label_name = filename[:-4]
        try:
            df = pd.read_csv(file_path, low_memory=False, encoding='utf-8-sig')
            if df.empty:
                continue
            df.columns = [str(col).strip() for col in df.columns]
            if 'Subject' in df.columns and 'Object' in df.columns:
                df = df[['Subject', 'Object']].dropna()
                df['label'] = label_name
                all_data.append(df)
        except Exception as e:
            logging.warning(f"{filename} load error: {e}")

    if not all_data:
        raise ValueError(f"{dir_path} not valid data")

    full_df = pd.concat(all_data, ignore_index=True)
    full_df['Subject'] = full_df['Subject'].astype(str)
    full_df['Object'] = full_df['Object'].astype(str)
    return full_df


# encode data
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


# dataset
class RelationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, label_encoder, max_length=128):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.le = label_encoder
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        input_ids, attention_mask = encode_pair(
            self.tokenizer,
            str(row['Subject']),
            str(row['Object']),
            self.max_length,
        )
        label_id = self.le.transform([row['label']])[0]
        return {
            'valid': True,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label_id, dtype=torch.long),
        }


def dynamic_collate_fn(samples):
    valid_samples = [s for s in samples if s.get('valid', False)]
    if not valid_samples:
        return None

    return {
        'input_ids': torch.stack([s['input_ids'] for s in valid_samples]),
        'attention_mask': torch.stack([s['attention_mask'] for s in valid_samples]),
        'label': torch.stack([s['label'] for s in valid_samples]),
    }


# model
class CPAModel(nn.Module):
    def __init__(self, model_name, num_labels, use_flash_attn=False):
        super().__init__()
        try:
            self.encoder = AutoModel.from_pretrained(model_name)
        except OSError:
            logging.warning(f'pretrained weights for {model_name} not found; initialize model from config')
            config = AutoConfig.from_pretrained(model_name)
            self.encoder = AutoModel.from_config(config)
        self.dropout = nn.Dropout(0.1)

        hidden_size = getattr(self.encoder.config, 'hidden_size', None)
        if hidden_size is None:
            raise ValueError('hidden_size is None')

        self.classifier = nn.Linear(hidden_size, num_labels)

        if use_flash_attn:
            logging.warning('flash attention not activated')

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
        cls_embedding = sequence_output[:, 0, :]
        logits = self.classifier(self.dropout(cls_embedding))
        return logits


# seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# log
def setup_logging(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(save_dir, 'train.log'), mode='w', encoding='utf-8'),
            logging.StreamHandler(),
        ],
    )


# device
def resolve_device(device_arg):
    requested = (device_arg or '').lower()
    if requested.startswith('gpu'):
        requested = requested.replace('gpu', 'cuda', 1)

    if requested.startswith('cuda'):
        if torch.cuda.is_available():
            device = torch.device(requested)
            logging.info(f'use: {device}')
            return device
        logging.warning(f'{device_arg} requested but CUDA is not available')

    if requested == 'cpu':
        device = torch.device('cpu')
        logging.info(f'use: {device}')
        return device

    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f'use: {device}')
        return device

    device = torch.device('cpu')
    logging.warning('set device to CPU')
    return device


# labels
def save_label_classes(label_encoder, save_dir):
    path = os.path.join(save_dir, 'label_classes.txt')
    with open(path, 'w', encoding='utf-8') as f:
        for label in label_encoder.classes_:
            f.write(f'{label}\n')


def save_train_args(args, save_dir, num_classes):
    config = vars(args).copy()
    config['num_classes'] = int(num_classes)
    with open(os.path.join(save_dir, 'train_args.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


# train
def run_training(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.output_dir, f'cpa_{timestamp}')
    setup_logging(save_dir)
    set_seed(args.random_seed)
    device = resolve_device(args.device)

    logging.info(f'device: {device}')

    # 1. load train data
    raw_train_df = load_data_from_directory(args.train_dir)

    # 2. build label
    label_encoder = LabelEncoder()
    label_encoder.fit(raw_train_df['label'].unique())
    num_classes = len(label_encoder.classes_)
    logging.info(f'label_num: {num_classes}')
    save_label_classes(label_encoder, save_dir)
    save_train_args(args, save_dir, num_classes)

    # 3. split dataset
    counts = raw_train_df['label'].value_counts()
    rare_labels = counts[counts < 2].index
    df_rare = raw_train_df[raw_train_df['label'].isin(rare_labels)]
    df_common = raw_train_df[~raw_train_df['label'].isin(rare_labels)]

    if len(df_common) == 0:
        raise ValueError("data num < 2, can't split dataset")

    train_c, val_c = train_test_split(
        df_common,
        test_size=args.val_ratio,
        stratify=df_common['label'],
        random_state=args.random_seed,
    )
    train_df = pd.concat([train_c, df_rare]).sample(frac=1, random_state=args.random_seed).reset_index(drop=True)
    val_df = val_c.reset_index(drop=True)
    logging.info(f'split success: train={len(train_df)}, val={len(val_df)}')

    # 4. Tokenizer & DataLoader
    tokenizer = AutoTokenizer.from_pretrained(args.shortcut_name)
    train_loader = DataLoader(
        RelationDataset(train_df, tokenizer, label_encoder, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dynamic_collate_fn,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )
    val_loader = DataLoader(
        RelationDataset(val_df, tokenizer, label_encoder, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dynamic_collate_fn,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )

    # 5. model init
    model = CPAModel(args.shortcut_name, num_classes, args.use_flash_attention).to(device)
    total_steps = max(1, len(train_loader) * args.epoch)
    warmup_steps = int(total_steps * args.warmup_ratio)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    loss_fn = nn.CrossEntropyLoss()

    use_amp = args.use_amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # 6. training
    best_acc = 0.0
    patience_counter = 0
    patience_limit = args.patience

    logging.info('start training...')
    for epoch in range(args.epoch):
        model.train()
        tr_loss = 0.0
        train_steps = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epoch}')
        for batch in pbar:
            if batch is None:
                continue

            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            label_ids = batch['label'].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(input_ids, mask)
                loss = loss_fn(logits, label_ids)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            lr_scheduler.step()
            loss_value = float(loss.item())
            tr_loss += loss_value
            train_steps += 1
            pbar.set_postfix({'loss': f'{loss_value:.4f}'})

        # val stage
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue

                input_ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                label_ids = batch['label'].to(device)

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(input_ids, mask)

                preds = torch.argmax(logits, dim=1)
                val_correct += int((preds == label_ids).sum().item())
                val_total += int(label_ids.shape[0])

        avg_train_loss = tr_loss / max(1, train_steps)
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        logging.info(f'Epoch {epoch + 1} | Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
            try:
                tokenizer.save_pretrained(save_dir)
            except Exception:
                pass
            logging.info(f'best model! (Acc: {best_acc:.4f})')
        else:
            patience_counter += 1
            logging.info(f'early stop count: {patience_counter}/{patience_limit}')
            if patience_counter == patience_limit:
                logging.info(f'{patience_limit} epoch not up, early stop!!!')
                break

    logging.info(f'train finish, best acc: {best_acc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default="../dataset/Train_Set")
    parser.add_argument('--output_dir', type=str, default='./cpa_output')
    parser.add_argument('--shortcut_name', type=str, default='bert-base-uncased')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--use_flash_attention', action='store_true')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='gpu')
    args = parser.parse_args()
    run_training(args)
