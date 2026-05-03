import os
import json
import time
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from baseline_model import CustomStandardScaler, CustomOneHotEncoder
from utils import EarlyStopping


# =============================================================================
# Feature configuration (same cyclical set used for both FNN and LSTM)
# =============================================================================
NUMERIC_FEATURES  = ['temp', 'rain_1h', 'snow_1h', 'clouds_all']
CYCLICAL_FEATURES = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                     'month_sin', 'month_cos']
CATEGORICAL_BASE  = ['holiday', 'weather_main']


def preprocess(train_df, val_df, test_df):
    """
    Fit scaler and OHE on train only, then apply to val and test.
    Cyclical features are already in [-1, 1] — no scaling needed.
    """
    scaler = CustomStandardScaler()
    X_num_tr = scaler.fit_transform(train_df[NUMERIC_FEATURES])
    X_num_va = scaler.transform(val_df[NUMERIC_FEATURES])
    X_num_te = scaler.transform(test_df[NUMERIC_FEATURES])

    ohe = CustomOneHotEncoder()
    X_cat_tr = ohe.fit_transform(train_df[CATEGORICAL_BASE])
    X_cat_va = ohe.transform(val_df[CATEGORICAL_BASE])
    X_cat_te = ohe.transform(test_df[CATEGORICAL_BASE])

    X_cyc_tr = train_df[CYCLICAL_FEATURES].values
    X_cyc_va = val_df[CYCLICAL_FEATURES].values
    X_cyc_te = test_df[CYCLICAL_FEATURES].values

    X_train = np.hstack([X_num_tr, X_cyc_tr, X_cat_tr]).astype(np.float32)
    X_val   = np.hstack([X_num_va, X_cyc_va, X_cat_va]).astype(np.float32)
    X_test  = np.hstack([X_num_te, X_cyc_te, X_cat_te]).astype(np.float32)

    return X_train, X_val, X_test


# =============================================================================
# Model definition
# =============================================================================

class FNN(nn.Module):
    """
    Multi-layer perceptron with ReLU activations and dropout.

    Architecture for hidden_layers=[256, 128, 64]:
      Input → Linear(256) → ReLU → Dropout
            → Linear(128) → ReLU → Dropout
            → Linear(64)  → ReLU          (no dropout before output)
            → Linear(1)
    """
    def __init__(self, input_dim: int, hidden_layers: list, dropout: float):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for i, h in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if i < len(hidden_layers) - 1:   # dropout on all but last hidden
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# =============================================================================
# Training
# =============================================================================

def train_one_config(X_train, y_train, X_val, y_val, params, device,
                     n_epochs=200, batch_size=256):
    """Train a single FNN config and return (best_val_rmse, train_losses, val_losses)."""
    model = FNN(X_train.shape[1],
                params['hidden_layers'],
                params['dropout']).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=params['lr'],
                                 weight_decay=params['weight_decay'])
    criterion = nn.MSELoss()
    stopper   = EarlyStopping(patience=10)

    X_tr = torch.tensor(X_train, device=device)
    y_tr = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_va = torch.tensor(X_val,   device=device)
    y_va = torch.tensor(y_val,   dtype=torch.float32, device=device)

    loader = DataLoader(TensorDataset(X_tr, y_tr),
                        batch_size=batch_size, shuffle=False)

    train_losses, val_losses = [], []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        train_losses.append(epoch_loss / len(X_train))

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_va), y_va).item()
        val_losses.append(val_loss)

        if stopper.step(val_loss, model):
            break

    stopper.restore(model)
    best_val_rmse = float(np.sqrt(stopper.best_loss))
    return model, best_val_rmse, train_losses, val_losses


SEED = 42


def set_seed(seed: int) -> None:
    """Fix all random sources for full reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    set_seed(SEED)
    os.makedirs('../results', exist_ok=True)
    os.makedirs('../images', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  Seed: {SEED}")

    print("Loading splits...")
    train_df = pd.read_csv('../data/train.csv')
    val_df   = pd.read_csv('../data/val.csv')
    test_df  = pd.read_csv('../data/test.csv')

    y_train = train_df['traffic_volume'].values.astype(np.float32)
    y_val   = val_df['traffic_volume'].values.astype(np.float32)
    y_test  = test_df['traffic_volume'].values.astype(np.float32)

    X_train, X_val, X_test = preprocess(train_df, val_df, test_df)
    print(f"Feature dim: {X_train.shape[1]}")

    # -------------------------------------------------------------------------
    # Hyperparameter grid search (val set only — test set never touched here)
    # -------------------------------------------------------------------------
    param_grid = {
        'lr':            [1e-3, 1e-4],
        'dropout':       [0.2, 0.4],
        'weight_decay':  [1e-4, 1e-3],
        'hidden_layers': [[256, 128, 64], [128, 64]],
    }

    keys   = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    print(f"\nGrid search: {len(combos)} combinations")

    grid_results = []
    best_val_rmse, best_params, best_losses = float('inf'), None, None

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        _, val_r, tr_l, va_l = train_one_config(
            X_train, y_train, X_val, y_val, params, device)
        grid_results.append({**params, 'hidden_layers': str(params['hidden_layers']),
                              'val_rmse': round(val_r, 4)})
        print(f"  [{i+1:2d}/{len(combos)}] {params}  val_rmse={val_r:.2f}")
        if val_r < best_val_rmse:
            best_val_rmse = val_r
            best_params = params
            best_losses = (tr_l, va_l)

    pd.DataFrame(grid_results).to_csv('../results/fnn_grid_results.csv', index=False)
    print(f"\nBest config: {best_params}  val_rmse={best_val_rmse:.2f}")

    # -------------------------------------------------------------------------
    # Retrain best config on train+val, then evaluate on test once
    # -------------------------------------------------------------------------
    print("Retraining best config on train+val...")
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    t0 = time.time()
    final_model, _, tr_l_final, va_l_final = train_one_config(
        X_trainval, y_trainval, X_val, y_val, best_params, device)
    train_time = time.time() - t0

    final_model.eval()
    with torch.no_grad():
        X_te_t = torch.tensor(X_test, device=device)
        y_pred = final_model(X_te_t).cpu().numpy()

    y_test_np = y_test
    test_mse  = float(np.mean((y_test_np - y_pred) ** 2))
    test_rmse = float(np.sqrt(test_mse))
    test_mae  = float(np.mean(np.abs(y_test_np - y_pred)))

    metrics = {
        'hidden_layers':  str(best_params['hidden_layers']),
        'lr':             best_params['lr'],
        'dropout':        best_params['dropout'],
        'weight_decay':   best_params['weight_decay'],
        'val_rmse':       round(best_val_rmse, 4),
        'test_rmse':      round(test_rmse, 4),
        'test_mae':       round(test_mae, 4),
        'test_mse':       round(test_mse, 4),
        'train_time_sec': round(train_time, 2),
    }
    print("\n--- FNN Final Results (test set) ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    np.save('../results/fnn_predictions_test.npy', y_pred)
    np.save('../results/fnn_train_losses.npy', np.array(best_losses[0]))
    np.save('../results/fnn_val_losses.npy',   np.array(best_losses[1]))
    with open('../results/fnn_best_params.json', 'w') as f:
        json.dump({**best_params,
                   'hidden_layers': best_params['hidden_layers'],
                   'val_rmse': best_val_rmse}, f, indent=2)
    with open('../results/fnn_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Learning curve for best config
    tr_l, va_l = best_losses
    stop_epoch  = len(tr_l)
    epochs      = range(1, stop_epoch + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, [np.sqrt(x) for x in tr_l], label='Train RMSE', color='coral')
    plt.plot(epochs, [np.sqrt(x) for x in va_l], label='Val RMSE',   color='steelblue')
    plt.axvline(stop_epoch, color='green', linestyle=':', linewidth=1.5,
                label=f'Early stop (epoch {stop_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('FNN Learning Curve (best hyperparameter config)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../images/fnn_loss_curve.png', dpi=150)
    plt.close()
    print("Saved ../images/fnn_loss_curve.png")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()
