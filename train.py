"""
 train.py - Training & Hyperparameter Tuning

 Phase 5: Training Loop & Optuna Hyperparameter Tuning
 - ASHRAELoss (MSE + RMSLE)
 - Training loop dengan early stopping
 - Optuna hyperparameter tuning
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import copy
import optuna
from optuna.trial import TrialState
import joblib


class ASHRAELoss(torch.nn.Module):
    """
    Custom loss: Kombinasi MSE + RMSLE
    RMSLE lebih toleran ke outlier dan fokus ke relative error
    """
    def __init__(self, alpha=0.5):
        super(ASHRAELoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        # MSE
        mse = F.mse_loss(pred, target)

        # RMSLE
        log_pred = torch.log1p(pred)
        log_target = torch.log1p(target)
        rmsle = torch.sqrt(F.mse_loss(log_pred, log_target))

        return self.alpha * mse + (1 - self.alpha) * rmsle


class EarlyStopping:
    """Early stopping untuk training"""
    def __init__(self, patience=10, min_delta=1e-4, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model)


def train_epoch(model, loader, criterion, optimizer, device, clip_grad=0.5):
    """Train satu epoch"""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        sequence = batch['sequence'].to(device)
        building_id = batch['building_id'].to(device)
        target = batch['target'].to(device)

        optimizer.zero_grad()
        pred, _ = model(sequence, building_id)
        loss = criterion(pred, target)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def validate(model, loader, criterion, device):
    """Validation"""
    model.eval()
    total_loss = 0
    n_batches = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            sequence = batch['sequence'].to(device)
            building_id = batch['building_id'].to(device)
            target = batch['target'].to(device)

            pred, _ = model(sequence, building_id)
            loss = criterion(pred, target)

            total_loss += loss.item()
            n_batches += 1

            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())

    avg_loss = total_loss / n_batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Metrics
    rmse = torch.sqrt(F.mse_loss(all_preds, all_targets)).item()
    mae = F.l1_loss(all_preds, all_targets).item()
    rmsle = torch.sqrt(F.mse_loss(
        torch.log1p(all_preds), torch.log1p(all_targets)
    )).item()

    return avg_loss, rmse, mae, rmsle


def train_model(model, train_loader, val_loader, config, device='cuda'):
    """
    Train model dengan early stopping
    """
    print("=" * 60)
    print("🚀 TRAINING")
    print("=" * 60)

    criterion = ASHRAELoss(alpha=config.get('loss_alpha', 0.5))
    optimizer = AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-5
    )

    early_stopping = EarlyStopping(patience=config.get('patience', 10))
    best_val_rmsle = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_rmse': [],
               'val_mae': [], 'val_rmsle': []}

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_rmse, val_mae, val_rmsle = validate(
            model, val_loader, criterion, device
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        history['val_mae'].append(val_mae)
        history['val_rmsle'].append(val_rmsle)

        print(f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"RMSE: {val_rmse:.4f} | RMSLE: {val_rmsle:.4f} | LR: {current_lr:.6f}")

        if val_rmsle < best_val_rmsle:
            best_val_rmsle = val_rmsle
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'val_rmsle': val_rmsle
            }, 'best_model.pth')
            print(f"✅ Saved best model (RMSLE: {val_rmsle:.4f})")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("🛑 Early stopping triggered!")
            break

    print("\n✅ Training complete!")
    return history


def create_optuna_study(train_loader, val_loader, base_config, device='cuda'):
    """
    Hyperparameter tuning dengan Optuna
    """
    print("=" * 60)
    print("🔧 HYPERPARAMETER TUNING WITH OPTUNA")
    print("=" * 60)

    def objective(trial):
        # Hyperparameter search space
        config = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
            'num_layers': trial.suggest_int('num_layers', 1, 3),
            'num_attention_heads': trial.suggest_categorical('num_attention_heads', [2, 4, 8]),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'loss_alpha': trial.suggest_float('loss_alpha', 0.3, 0.7),
            'n_buildings': base_config['n_buildings'],
            'embedding_dim': base_config['embedding_dim'],
            'input_dim': base_config['input_dim'],
            'output_horizon': base_config['output_horizon'],
            'epochs': 5,  # Short for tuning
            'patience': 3
        }

        # Validate d_model divisible by num_heads
        d_model = config['hidden_dim'] * 2
        if d_model % config['num_attention_heads'] != 0:
            raise optuna.TrialPruned()

        print(f"\n--- Trial {trial.number} ---")
        print(f"Config: hidden={config['hidden_dim']}, heads={config['num_attention_heads']}, "
              f"lr={config['lr']:.6f}, dropout={config['dropout']:.3f}")

        # Create model
        from model import create_model
        model = create_model(config, device)

        # Train (short)
        criterion = ASHRAELoss(alpha=config['loss_alpha'])
        optimizer = AdamW(model.parameters(), lr=config['lr'],
                         weight_decay=config['weight_decay'])

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(config['epochs']):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

            # Quick validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    seq = batch['sequence'].to(device)
                    bid = batch['building_id'].to(device)
                    tgt = batch['target'].to(device)
                    pred, _ = model(seq, bid)
                    val_loss += criterion(pred, tgt).item()
            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 3:
                    break

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_val_loss

    # Create study
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(objective, n_trials=20, timeout=3600)

    print("\n" + "=" * 60)
    print("🏆 BEST HYPERPARAMETERS")
    print("=" * 60)
    print(f"Best Val Loss: {study.best_value:.4f}")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    joblib.dump(study, 'optuna_study.pkl')
    print("\n✅ Study saved to optuna_study.pkl")

    return study.best_params


if __name__ == "__main__":
    print("Train module - import and use train_model() or create_optuna_study()")
