import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit


# Custom preprocessing and model classes
# (kept as custom implementations — no sklearn for core ML)
class CustomStandardScaler:
    """Z-score normalizer: subtracts mean and divides by std (fit on train only)."""
    def fit(self, X):
        X = np.asarray(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (np.asarray(X) - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class CustomOneHotEncoder:
    """One-hot encoder. Unknown categories at transform time become all-zeros."""
    def fit(self, X):
        X = np.asarray(X).astype(str)
        self.categories_ = [np.unique(X[:, i]) for i in range(X.shape[1])]
        return self

    def transform(self, X):
        X = np.asarray(X).astype(str)
        out = []
        for i, cats in enumerate(self.categories_):
            col = np.zeros((X.shape[0], len(cats)))
            for j, cat in enumerate(cats):
                col[:, j] = (X[:, i] == cat).astype(float)
            out.append(col)
        return np.hstack(out)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def get_feature_names(self, input_features):
        """Returns column names matching the one-hot output order."""
        names = []
        for feat, cats in zip(input_features, self.categories_):
            for cat in cats:
                names.append(f"{feat}_{cat}")
        return names


class CustomRidgeRegression:
    """
    Closed-form Ridge Regression: θ = (X^T X + αI)^{-1} X^T y
    L2 regularization via alpha prevents overfitting on high-dimensional
    one-hot feature spaces.
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None

    def get_params(self, deep=True):  # noqa: ARG002 — sklearn-compatible signature
        return {"alpha": self.alpha}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        n, p = X.shape
        X_aug = np.column_stack([np.ones(n), X])
        I = np.eye(p + 1)
        I[0, 0] = 0.0          # don't regularize the intercept
        theta = np.linalg.solve(X_aug.T @ X_aug + self.alpha * I, X_aug.T @ y)
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        return self

    def predict(self, X):
        return np.asarray(X) @ self.coef_ + self.intercept_


def mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))

def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


# Feature preprocessing helpers
NUMERIC_FEATURES     = ['temp', 'rain_1h', 'snow_1h', 'clouds_all']
CATEGORICAL_OHE      = ['holiday', 'weather_main', 'hour', 'day_of_week', 'month']
CATEGORICAL_BASE     = ['holiday', 'weather_main']
CYCLICAL_FEATURES    = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                        'month_sin', 'month_cos']


def build_features(train_df, val_df, test_df, config='cyclical'):
    """
    Fit scaler and encoder on train only, apply to val and test.

    config='onehot'   — one-hot encode all temporal columns (Config A)
    config='cyclical' — use sin/cos for temporal, OHE only for non-ordinal cats (Config B)
    """
    scaler = CustomStandardScaler()
    X_train_num = scaler.fit_transform(train_df[NUMERIC_FEATURES])
    X_val_num   = scaler.transform(val_df[NUMERIC_FEATURES])
    X_test_num  = scaler.transform(test_df[NUMERIC_FEATURES])

    if config == 'onehot':
        ohe = CustomOneHotEncoder()
        X_train_cat = ohe.fit_transform(train_df[CATEGORICAL_OHE])
        X_val_cat   = ohe.transform(val_df[CATEGORICAL_OHE])
        X_test_cat  = ohe.transform(test_df[CATEGORICAL_OHE])
        feat_names = (NUMERIC_FEATURES +
                      ohe.get_feature_names(CATEGORICAL_OHE))
        X_train = np.hstack([X_train_num, X_train_cat])
        X_val   = np.hstack([X_val_num,   X_val_cat])
        X_test  = np.hstack([X_test_num,  X_test_cat])

    else:  # cyclical
        ohe = CustomOneHotEncoder()
        X_train_cat = ohe.fit_transform(train_df[CATEGORICAL_BASE])
        X_val_cat   = ohe.transform(val_df[CATEGORICAL_BASE])
        X_test_cat  = ohe.transform(test_df[CATEGORICAL_BASE])

        X_train_cyc = train_df[CYCLICAL_FEATURES].values
        X_val_cyc   = val_df[CYCLICAL_FEATURES].values
        X_test_cyc  = test_df[CYCLICAL_FEATURES].values

        feat_names = (NUMERIC_FEATURES + CYCLICAL_FEATURES +
                      ohe.get_feature_names(CATEGORICAL_BASE))
        X_train = np.hstack([X_train_num, X_train_cyc, X_train_cat])
        X_val   = np.hstack([X_val_num,   X_val_cyc,   X_val_cat])
        X_test  = np.hstack([X_test_num,  X_test_cyc,  X_test_cat])

    return X_train, X_val, X_test, feat_names


def main():
    os.makedirs('../results', exist_ok=True)
    os.makedirs('../images', exist_ok=True)

    print("Loading splits...")
    train_df = pd.read_csv('../data/train.csv')
    val_df   = pd.read_csv('../data/val.csv')
    test_df  = pd.read_csv('../data/test.csv')

    y_train = train_df['traffic_volume'].values
    y_val   = val_df['traffic_volume'].values
    y_test  = test_df['traffic_volume'].values

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Feature config comparison: one-hot vs cyclical (Config A vs B)
    # We compare both on val to motivate our encoding choice.

    alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    print("\n--- Config A: One-Hot Temporal Features ---")
    X_train_a, X_val_a, X_test_a, feat_names_a = build_features(
        train_df, val_df, test_df, config='onehot')

    best_val_a, best_alpha_a = float('inf'), None
    for alpha in alphas:
        m = CustomRidgeRegression(alpha=alpha)
        m.fit(X_train_a, y_train)
        v = rmse(y_val, m.predict(X_val_a))
        print(f"  alpha={alpha:<8} val_rmse={v:.2f}")
        if v < best_val_a:
            best_val_a, best_alpha_a = v, alpha
    print(f"  Best alpha (Config A): {best_alpha_a}  val_rmse={best_val_a:.2f}")

    print("\n--- Config B: Cyclical Temporal Features ---")
    X_train_b, X_val_b, X_test_b, feat_names_b = build_features(
        train_df, val_df, test_df, config='cyclical')

    best_val_b, best_alpha_b = float('inf'), None
    for alpha in alphas:
        m = CustomRidgeRegression(alpha=alpha)
        m.fit(X_train_b, y_train)
        v = rmse(y_val, m.predict(X_val_b))
        print(f"  alpha={alpha:<8} val_rmse={v:.2f}")
        if v < best_val_b:
            best_val_b, best_alpha_b = v, alpha
    print(f"  Best alpha (Config B): {best_alpha_b}  val_rmse={best_val_b:.2f}")

    # Choose best config based on val performance
    if best_val_b <= best_val_a:
        best_config = 'cyclical'
        X_train, X_val, X_test, feat_names = X_train_b, X_val_b, X_test_b, feat_names_b
        best_alpha = best_alpha_b
        print("\nUsing Config B (cyclical) — better or equal val RMSE.")
    else:
        best_config = 'onehot'
        X_train, X_val, X_test, feat_names = X_train_a, X_val_a, X_test_a, feat_names_a
        best_alpha = best_alpha_a
        print("\nUsing Config A (one-hot) — better val RMSE.")


    print(f"\n--- 5-Fold TimeSeriesSplit CV on train (alpha={best_alpha}) ---")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_rmses = []
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_train)):
        m = CustomRidgeRegression(alpha=best_alpha)
        m.fit(X_train[tr_idx], y_train[tr_idx])
        fold_rmse = rmse(y_train[te_idx], m.predict(X_train[te_idx]))
        cv_rmses.append(fold_rmse)
        print(f"  Fold {fold+1}: RMSE={fold_rmse:.2f}")
    cv_mean = float(np.mean(cv_rmses))
    cv_std  = float(np.std(cv_rmses))
    print(f"  CV RMSE: {cv_mean:.2f} ± {cv_std:.2f}")

    # Alpha grid results table (for report)
    grid_rows = []
    for alpha in alphas:
        m = CustomRidgeRegression(alpha=alpha)
        m.fit(X_train, y_train)
        val_r = rmse(y_val, m.predict(X_val))

        cv_r_list = []
        for tr_idx, te_idx in tscv.split(X_train):
            m2 = CustomRidgeRegression(alpha=alpha)
            m2.fit(X_train[tr_idx], y_train[tr_idx])
            cv_r_list.append(rmse(y_train[te_idx], m2.predict(X_train[te_idx])))

        grid_rows.append({
            'alpha': alpha,
            'val_rmse': round(val_r, 4),
            'cv_mean_rmse': round(float(np.mean(cv_r_list)), 4),
            'cv_std_rmse':  round(float(np.std(cv_r_list)),  4),
        })

    pd.DataFrame(grid_rows).to_csv('../results/ridge_alpha_grid.csv', index=False)

    # Final model: refit on train+val combined, evaluate on test once
    print("\nRefitting best model on train+val...")
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    t0 = time.time()
    final_model = CustomRidgeRegression(alpha=best_alpha)
    final_model.fit(X_trainval, y_trainval)
    train_time = time.time() - t0

    y_train_pred = final_model.predict(X_train)
    y_test_pred  = final_model.predict(X_test)

    metrics = {
        'config':      best_config,
        'alpha':       best_alpha,
        'train_rmse':  round(rmse(y_train, y_train_pred), 4),
        'val_rmse':    round(best_val_b if best_config == 'cyclical' else best_val_a, 4),
        'test_rmse':   round(rmse(y_test, y_test_pred), 4),
        'test_mae':    round(mae(y_test, y_test_pred), 4),
        'test_mse':    round(mse(y_test, y_test_pred), 4),
        'cv_mean_rmse': round(cv_mean, 4),
        'cv_std_rmse':  round(cv_std, 4),
        'train_time_sec': round(train_time, 4),
        'n_features':  int(X_train.shape[1]),
    }

    print("\n--- Final Ridge Results (test set) ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Save artifacts
    np.save('../results/ridge_predictions_test.npy', y_test_pred)
    np.save('../results/ridge_y_test.npy', y_test)
    np.save('../results/ridge_coef.npy', final_model.coef_)
    with open('../results/ridge_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    with open('../results/ridge_feature_names.json', 'w') as f:
        json.dump(feat_names, f, indent=2)

    # Plot: val RMSE and CV RMSE vs log(alpha) — bias-variance visualization
    # High alpha → underfitting (high bias). Low alpha → overfitting (high variance).
    grid_df = pd.DataFrame(grid_rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogx(grid_df['alpha'], grid_df['val_rmse'],
                marker='o', label='Val RMSE', color='coral', linewidth=2)
    ax.semilogx(grid_df['alpha'], grid_df['cv_mean_rmse'],
                marker='s', label='CV Mean RMSE (train)', color='steelblue',
                linewidth=2, linestyle='--')
    ax.fill_between(grid_df['alpha'],
                    grid_df['cv_mean_rmse'] - grid_df['cv_std_rmse'],
                    grid_df['cv_mean_rmse'] + grid_df['cv_std_rmse'],
                    alpha=0.2, color='steelblue', label='CV ± 1 std')
    ax.axvline(best_alpha, color='green', linestyle=':', linewidth=2,
               label=f'Best α={best_alpha}')
    ax.set_xlabel('Regularization Strength α (log scale)')
    ax.set_ylabel('RMSE')
    ax.set_title('Ridge Regression: Bias–Variance Trade-off\n'
                 '← low α (overfits) · · · high α (underfits) →')
    ax.legend()
    plt.tight_layout()
    plt.savefig('../images/ridge_alpha_vs_rmse.png', dpi=150)
    plt.close()
    print("Saved ../images/ridge_alpha_vs_rmse.png")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()
