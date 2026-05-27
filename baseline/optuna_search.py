"""
Optuna hyperparameter search for Extraction_pytorch (objective: validation Score_final).
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from train import build_train_parser, run_training


def suggest_train_args(trial, base_args):
    """Map Optuna trial to training Namespace fields."""
    batch_size = trial.suggest_categorical('batch_size', [16, 24, 32])
    grad_accum = trial.suggest_categorical('gradient_accumulation_steps', [1, 2, 3])
    effective_batch = batch_size * grad_accum
    if effective_batch < 32 or effective_batch > 64:
        raise optuna.TrialPruned(
            f'effective_batch={effective_batch} outside [32, 64]'
        )

    lr = trial.suggest_float('lr', 2e-5, 1e-4, log=True)
    warmup_ratio = trial.suggest_float('warmup_ratio', 0.05, 0.12)
    max_length = trial.suggest_categorical('max_length', [128, 160])
    dropout = trial.suggest_float('dropout', 0.05, 0.3)
    weight_decay = trial.suggest_float('weight_decay', 1e-3, 0.1, log=True)

    args = argparse.Namespace(**vars(base_args))
    args.batch_size = batch_size
    args.gradient_accumulation_steps = grad_accum
    args.lr = lr
    args.warmup_ratio = warmup_ratio
    args.max_length = max_length
    args.dropout = dropout
    args.weight_decay = weight_decay
    args.run_name = f'optuna_trial_{trial.number:04d}'
    args.loss_type = 'competition'
    args.selection_metric = 'score_final'
    args.normalize_class_weights = True
    return args


def create_objective(base_args, enable_pruning):
    def objective(trial):
        train_args = suggest_train_args(trial, base_args)
        try:
            result = run_training(
                train_args,
                optuna_trial=trial if enable_pruning else None,
            )
        except optuna.TrialPruned:
            raise
        trial.set_user_attr('save_dir', result['save_dir'])
        trial.set_user_attr('best_acc', result['best_acc'])
        return result['best_score_final']

    return objective


def train_args_from_trial_params(params, base_args, run_name='optuna_best'):
    args = argparse.Namespace(**vars(base_args))
    for key, value in params.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.run_name = run_name
    args.loss_type = 'competition'
    args.selection_metric = 'score_final'
    args.normalize_class_weights = True
    return args


def run_infer(checkpoint_dir, base_args, output_file):
    infer_script = os.path.join(os.path.dirname(__file__), 'infer.py')
    labels_path = os.path.join(checkpoint_dir, 'label_classes.txt')
    model_path = os.path.join(checkpoint_dir, 'best_model.pt')
    cmd = [
        sys.executable,
        infer_script,
        '--input_csv',
        base_args.infer_input_csv,
        '--labels_path',
        labels_path,
        '--model_path',
        model_path,
        '--shortcut_name',
        base_args.shortcut_name,
        '--max_length',
        str(getattr(base_args, 'max_length', 128)),
        '--output_file',
        output_file,
        '--device',
        base_args.device,
        '--use_amp',
    ]
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def build_optuna_parser():
    train_parser = build_train_parser()
    parser = argparse.ArgumentParser(
        description='Optuna search for CPA training',
        parents=[train_parser],
        conflict_handler='resolve',
    )
    parser.add_argument('--n_trials', type=int, default=20)
    parser.add_argument('--study_name', type=str, default='extraction_deberta_score_final')
    parser.add_argument(
        '--storage',
        type=str,
        default='sqlite:///./optuna_output/extraction_optuna.db',
    )
    parser.add_argument('--artefact_root', type=str, default='./optuna_output')
    parser.add_argument('--pruning', action='store_true', help='Enable MedianPruner + epoch reports')
    parser.add_argument('--n_startup_trials', type=int, default=4)
    parser.add_argument('--sampler_seed', type=int, default=42)
    parser.add_argument(
        '--train-best',
        action='store_true',
        help='Retrain with best trial hyperparameters (run_name=optuna_best)',
    )
    parser.add_argument(
        '--infer-after-best',
        action='store_true',
        help='After --train-best, run infer.py on best checkpoint',
    )
    parser.add_argument(
        '--infer-input-csv',
        type=str,
        default='../dataset/test.csv',
        dest='infer_input_csv',
    )
    parser.add_argument(
        '--infer-output',
        type=str,
        default='../../contest/submission_deberta_optuna.csv',
    )
    return parser


def main():
    parser = build_optuna_parser()
    args = parser.parse_args()
    base_args = args

    os.makedirs(args.artefact_root, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction='maximize',
        load_if_exists=True,
        sampler=TPESampler(seed=args.sampler_seed, n_startup_trials=args.n_startup_trials),
        pruner=MedianPruner(n_startup_trials=args.n_startup_trials, n_warmup_steps=2)
        if args.pruning
        else None,
    )

    if args.train_best:
        if study.best_trial is None:
            raise RuntimeError('No completed trials in study; run search first.')
        best_params = study.best_trial.params
        print('Best trial:', study.best_trial.number, 'value=', study.best_trial.value)
        print('Params:', json.dumps(best_params, indent=2))
        train_args = train_args_from_trial_params(best_params, base_args)
        result = run_training(train_args)
        summary_path = os.path.join(args.artefact_root, 'best_retrain_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'best_trial_number': study.best_trial.number,
                    'optuna_best_value': study.best_trial.value,
                    'retrain': result,
                    'params': best_params,
                },
                f,
                indent=2,
            )
        print('Retrain finished:', result)
        if args.infer_after_best:
            run_infer(result['save_dir'], train_args, args.infer_output)
            print('Submission saved to:', args.infer_output)
        return

    objective = create_objective(base_args, args.pruning)
    print(f'Starting study={args.study_name} n_trials={args.n_trials} pruning={args.pruning}')
    study.optimize(objective, n_trials=args.n_trials)

    summary = {
        'study_name': args.study_name,
        'storage': args.storage,
        'n_trials': len(study.trials),
        'best_value': study.best_value,
        'best_params': study.best_params,
        'best_trial': study.best_trial.number if study.best_trial else None,
        'finished_at': datetime.now().isoformat(),
    }
    summary_path = os.path.join(args.artefact_root, 'study_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print('Study complete. Best Score_final:', study.best_value)
    print('Best params:', json.dumps(study.best_params, indent=2))
    print('Summary:', summary_path)


if __name__ == '__main__':
    main()
