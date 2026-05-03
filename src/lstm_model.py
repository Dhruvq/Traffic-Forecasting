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

# Feature configuration (same as FNN)
NUMERIC_FEATURES  = ['temp', 'rain_1h', 'snow_1h', 'clouds_all']
CYCLICAL_FEATURES = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                     'month_sin', 'month_cos']
CATEGORICAL_BASE  = ['holiday', 'weather_main']


def preprocess(train_df, val_df, test_df):
    """Fit scaler/OHE on train only, transform all three splits."""
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


def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """
    Converts a (T, F) feature matrix into sliding-window sequences.

    For each position i the input window is X[i : i+seq_len] and the
    label is y[i + seq_len], giving shape (T-seq_len, seq_len, F).

    The 'look-back prepend' trick is handled by the caller so that val
    and test sequences can use the tail of the preceding split as context.
    """
    T = len(X)
    Xs = np.stack([X[i: i + seq_len] for i in range(T - seq_len)])
    ys = y[seq_len:]
    return Xs.astype(np.float32), ys.astype(np.float32)


# Model definition

class LSTMModel(nn.Module):
    """
    LSTM followed by a two-layer MLP head.

    The LSTM processes the full sequence and we take only the last hidden
    state (out[:, -1, :]) as a summary of the window, then predict traffic
    volume through a small MLP.
    """
    def __init__(self, input_dim: int, hidden_size: int,
                 num_layers: int, dropout: float):
        super().__init__()
        # Dropout only applies between LSTM layers; set to 0 for single layer
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers,
                            batch_first=True, dropout=lstm_dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# Training

def train_one_config(X_seq_tr, y_seq_tr, X_seq_va, y_seq_va,
                     params, device, n_epochs=200, batch_size=256):
    """Train a single LSTM config. Returns (model, best_val_rmse, tr_losses, va_losses)."""
    input_dim = X_seq_tr.shape[2]
    model = LSTMModel(input_dim,
                      params['hidden_size'],
                      params['num_layers'],
                      params['dropout']).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-3,
                                 weight_decay=1e-4)
    criterion = nn.MSELoss()
    stopper   = EarlyStopping(patience=10)

    X_tr = torch.tensor(X_seq_tr, device=device)
    y_tr = torch.tensor(y_seq_tr, device=device)
    X_va = torch.tensor(X_seq_va, device=device)
    y_va = torch.tensor(y_seq_va, device=device)

    loader = DataLoader(TensorDataset(X_tr, y_tr),
                        batch_size=batch_size, shuffle=False)

    train_losses, val_losses = [], []

    for _ in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        train_losses.append(epoch_loss / len(X_seq_tr))

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_va), y_va).item()
        val_losses.append(val_loss)

        if stopper.step(val_loss, model):
            break

    stopper.restore(model)
    return model, float(np.sqrt(stopper.best_loss)), train_losses, val_losses


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

    # Hyperparameter grid search
    param_grid = {
        'seq_len':    [12, 24],
        'hidden_size':[64, 128],
        'num_layers': [1, 2],
        'dropout':    [0.1, 0.2],
    }

    keys   = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    print(f"\nGrid search: {len(combos)} combinations")

    grid_results = []
    best_val_rmse = float('inf')
    best_params, best_losses, best_seq_len = None, None, None
    best_seq_data = None

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        seq_len = params['seq_len']

        # Build sequences with look-back prepend so every val/test sample has
        # a full seq_len context window from the correct temporal split.
        # Train needs no prepend — sequences are cut from the train block only.
        # Val/test prepend the real tail of the preceding split (not zeros) so
        # the first window still captures genuine historical context.
        X_va_prep = np.vstack([X_train[-seq_len:], X_val])
        X_te_prep = np.vstack([X_val[-seq_len:], X_test])

        y_va_prep = np.concatenate([y_train[-seq_len:], y_val])
        y_te_prep = np.concatenate([y_val[-seq_len:], y_test])

        X_seq_tr, y_seq_tr = create_sequences(X_train, y_train, seq_len)
        X_seq_va, y_seq_va = create_sequences(X_va_prep, y_va_prep, seq_len)
        X_seq_te, y_seq_te = create_sequences(X_te_prep, y_te_prep, seq_len)

        lstm_params = {k: params[k] for k in ['hidden_size', 'num_layers', 'dropout']}
        _, val_r, tr_l, va_l = train_one_config(
            X_seq_tr, y_seq_tr, X_seq_va, y_seq_va, lstm_params, device)

        grid_results.append({**params, 'val_rmse': round(val_r, 4)})
        print(f"  [{i+1:2d}/{len(combos)}] {params}  val_rmse={val_r:.2f}")

        if val_r < best_val_rmse:
            best_val_rmse = val_r
            best_params   = params
            best_losses   = (tr_l, va_l)
            best_seq_data = (X_seq_tr, y_seq_tr, X_seq_va, y_seq_va,
                             X_seq_te, y_seq_te)

    pd.DataFrame(grid_results).to_csv('../results/lstm_grid_results.csv', index=False)
    print(f"\nBest config: {best_params}  val_rmse={best_val_rmse:.2f}")

    # Retrain best config on train+val, evaluate on test once
    print("Retraining best config on train+val...")
    X_seq_tr, y_seq_tr, X_seq_va, y_seq_va, X_seq_te, y_seq_te = best_seq_data

    X_seq_trainval = np.vstack([X_seq_tr, X_seq_va])
    y_seq_trainval = np.concatenate([y_seq_tr, y_seq_va])

    lstm_params = {k: best_params[k]
                   for k in ['hidden_size', 'num_layers', 'dropout']}
    t0 = time.time()
    final_model, _, tr_l_f, va_l_f = train_one_config(
        X_seq_trainval, y_seq_trainval, X_seq_va, y_seq_va, lstm_params, device)
    train_time = time.time() - t0

    final_model.eval()
    with torch.no_grad():
        X_te_t = torch.tensor(X_seq_te, device=device)
        y_pred = final_model(X_te_t).cpu().numpy()

    test_mse  = float(np.mean((y_seq_te - y_pred) ** 2))
    test_rmse = float(np.sqrt(test_mse))
    test_mae  = float(np.mean(np.abs(y_seq_te - y_pred)))

    metrics = {
        'seq_len':        best_params['seq_len'],
        'hidden_size':    best_params['hidden_size'],
        'num_layers':     best_params['num_layers'],
        'dropout':        best_params['dropout'],
        'val_rmse':       round(best_val_rmse, 4),
        'test_rmse':      round(test_rmse, 4),
        'test_mae':       round(test_mae, 4),
        'test_mse':       round(test_mse, 4),
        'train_time_sec': round(train_time, 2),
        'n_test_preds':   int(len(y_pred)),
    }
    print("\n--- LSTM Final Results (test set) ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    np.save('../results/lstm_predictions_test.npy', y_pred)
    np.save('../results/lstm_y_test_aligned.npy',   y_seq_te)
    np.save('../results/lstm_train_losses.npy', np.array(best_losses[0]))
    np.save('../results/lstm_val_losses.npy',   np.array(best_losses[1]))
    with open('../results/lstm_best_params.json', 'w') as f:
        json.dump({**best_params, 'val_rmse': best_val_rmse}, f, indent=2)
    with open('../results/lstm_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

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
    plt.title('LSTM Learning Curve (best hyperparameter config)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../images/lstm_loss_curve.png', dpi=150)
    plt.close()
    print("Saved ../images/lstm_loss_curve.png")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()
