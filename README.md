# Traffic Volume Forecasting — I-94 Westbound
### EE 559 — University of Southern California, Spring 2026

Predicts hourly interstate traffic volume using Ridge Regression, a Feedforward Neural Network (FNN), and an LSTM. Dataset: [UCI Metro Interstate Traffic Volume](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) (48,204 hourly observations, 2012–2018).

## Setup

```bash
# Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

## Run

All scripts are run from the `src/` directory:

```bash
cd src

# 1. Download dataset, engineer features, create 80/10/10 splits, generate EDA plots
python data_preparation.py

# 2. Ridge Regression — grid search over alpha, 5-fold TimeSeriesSplit CV
python baseline_model.py

# 3. FNN — 16-config hyperparameter grid search (5–10 min on CPU)
python fnn_model.py

# 4. LSTM — 16-config hyperparameter grid search (15–25 min on CPU)
python lstm_model.py

# 5. Unified comparison: 6 plots + model comparison table
python evaluate_all.py
```

Each script is self-contained and saves all outputs before exiting. Scripts 3–5 automatically use a GPU if one is available (`torch.cuda.is_available()`).

## Output Structure

```
data/
  traffic_data.csv                raw download
  preprocessed_traffic_data.csv   all features including cyclical encoding
  train.csv / val.csv / test.csv  80/10/10 chronological splits
  split_indices.json              split boundaries for LSTM sequence alignment

results/
  ridge_alpha_grid.csv            alpha vs val/CV RMSE
  ridge_metrics.json              final Ridge test metrics
  fnn_grid_results.csv            all 16 FNN configs + val RMSE
  fnn_metrics.json              final FNN test metrics
  lstm_grid_results.csv           all 16 LSTM configs + val RMSE
  lstm_metrics.json               final LSTM test metrics
  model_comparison_table.csv      side-by-side MSE / RMSE / MAE / time

images/
  traffic_volume_dist.png       target distribution (bimodal)
  avg_hourly_traffic.png        rush-hour peaks (7am, 4-5pm)
  avg_by_dow.png                weekday vs weekend pattern
  correlation_heatmap.png       feature–target correlations
  monthly_traffic_boxplot.png   seasonal spread
  cyclical_encoding_demo.png    motivation for sin/cos encoding
  ridge_alpha_vs_rmse.png       bias–variance trade-off for Ridge
  fnn_loss_curve.png            FNN train/val RMSE per epoch
  lstm_loss_curve.png           LSTM train/val RMSE per epoch
  scatter_all_models.png        predicted vs actual (all 3 models)
  residual_distributions.png    residual KDE plots (all 3 models)
  learning_curves.png           FNN + LSTM learning curves side-by-side
  ridge_bias_variance.png       annotated Ridge alpha plot
  ridge_feature_importance.png  top 20 Ridge coefficients
  rmse_comparison.png           bar chart of test RMSE across models
```

## Estimated CPU Training Times

| Script | Time |
|---|---|
| `data_preparation.py` | ~30 sec (includes download) |
| `baseline_model.py` | < 5 sec |
| `fnn_model.py` | ~5–10 min |
| `lstm_model.py` | ~15–25 min |
| `evaluate_all.py` | ~10 sec |

## Key Design Decisions

- **80/10/10 time-ordered split** — no random shuffling; prevents temporal data leakage
- **Cyclical encoding** (`sin`/`cos`) for hour, day-of-week, month in FNN/LSTM — encodes circular adjacency (hour 23 ≈ hour 0)
- **One-hot encoding** for temporal features in Ridge — each category gets an independent linear coefficient
- **Custom ML classes** — `CustomRidgeRegression`, `CustomStandardScaler`, `CustomOneHotEncoder` in `baseline_model.py` (no sklearn for the core ML)
- **`sklearn.TimeSeriesSplit`** used as a fold-index generator only (not as a model)
- **EarlyStopping** with `patience=10` shared across FNN and LSTM via `utils.py`
