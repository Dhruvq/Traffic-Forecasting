import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_json(path):
    with open(path) as f:
        return json.load(f)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def main():
    os.makedirs('../results', exist_ok=True)
    os.makedirs('../images', exist_ok=True)

    ridge_preds = np.load('../results/ridge_predictions_test.npy')
    fnn_preds   = np.load('../results/fnn_predictions_test.npy')
    lstm_preds  = np.load('../results/lstm_predictions_test.npy')
    y_test_full = np.load('../results/ridge_y_test.npy')
    y_test_lstm = np.load('../results/lstm_y_test_aligned.npy')

    ridge_m = load_json('../results/ridge_metrics.json')
    fnn_m   = load_json('../results/fnn_metrics.json')
    lstm_m  = load_json('../results/lstm_metrics.json')

    fnn_tr_l  = np.load('../results/fnn_train_losses.npy')
    fnn_va_l  = np.load('../results/fnn_val_losses.npy')
    lstm_tr_l = np.load('../results/lstm_train_losses.npy')
    lstm_va_l = np.load('../results/lstm_val_losses.npy')

    ridge_coef  = np.load('../results/ridge_coef.npy')
    with open('../results/ridge_feature_names.json') as f:
        feat_names = json.load(f)

    rows = [
        {
            'Model':           f"Ridge (α={ridge_m['alpha']})",
            'MSE':             ridge_m['test_mse'],
            'RMSE':            ridge_m['test_rmse'],
            'MAE':             ridge_m['test_mae'],
            'Train Time (s)':  ridge_m['train_time_sec'],
        },
        {
            'Model':           'FNN',
            'MSE':             fnn_m['test_mse'],
            'RMSE':            fnn_m['test_rmse'],
            'MAE':             fnn_m['test_mae'],
            'Train Time (s)':  fnn_m['train_time_sec'],
        },
        {
            'Model':           'LSTM',
            'MSE':             lstm_m['test_mse'],
            'RMSE':            lstm_m['test_rmse'],
            'MAE':             lstm_m['test_mae'],
            'Train Time (s)':  lstm_m['train_time_sec'],
        },
    ]
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv('../results/model_comparison_table.csv', index=False)

    print("\n=== Model Comparison (Test Set) ===")
    print(cmp_df.to_string(index=False))

    n_lstm = len(lstm_preds)
    y_shared = y_test_full[-n_lstm:]  

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    configs = [
        (ridge_preds[-n_lstm:], y_shared, f"Ridge (α={ridge_m['alpha']})", 'coral'),
        (fnn_preds[-n_lstm:],   y_shared, 'FNN',                           'steelblue'),
        (lstm_preds,            y_test_lstm, 'LSTM',                       'seagreen'),
    ]
    for ax, (preds, y_true, label, color) in zip(axes, configs):
        r2_val = r2(y_true, preds)
        ax.scatter(y_true, preds, alpha=0.15, s=4, color=color)
        lims = [min(y_true.min(), preds.min()), max(y_true.max(), preds.max())]
        ax.plot(lims, lims, 'k--', linewidth=1.2, label='Perfect prediction')
        ax.set_title(f"{label}\nRMSE={rmse(y_true, preds):.1f}  R²={r2_val:.3f}",
                     fontsize=11)
        ax.set_xlabel('Actual Traffic Volume')
        ax.set_ylabel('Predicted Traffic Volume')
        ax.legend(fontsize=8)
    plt.suptitle('Predicted vs Actual Traffic Volume', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../images/scatter_all_models.png', dpi=150)
    plt.close()
    print("Saved scatter_all_models.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (preds, y_true, label, color) in zip(axes, configs):
        residuals = preds - y_true
        ax.hist(residuals, bins=60, color=color, alpha=0.7, density=True)
        ax.axvline(residuals.mean(), color='black', linestyle='--',
                   linewidth=1.5, label=f'Mean={residuals.mean():.1f}')
        ax.set_title(f"{label} Residuals\nstd={residuals.std():.1f}", fontsize=11)
        ax.set_xlabel('Residual (Predicted − Actual)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
    plt.suptitle('Residual Distributions  (narrower = less variance, mean≈0 = unbiased)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../images/residual_distributions.png', dpi=150)
    plt.close()
    print("Saved residual_distributions.png")


    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (tr_l, va_l, label) in zip(axes, [
        (fnn_tr_l,  fnn_va_l,  'FNN'),
        (lstm_tr_l, lstm_va_l, 'LSTM'),
    ]):
        epochs = range(1, len(tr_l) + 1)
        ax.plot(epochs, np.sqrt(tr_l), label='Train RMSE', color='coral')
        ax.plot(epochs, np.sqrt(va_l), label='Val RMSE',   color='steelblue')
        stop = len(tr_l)
        ax.axvline(stop, color='green', linestyle=':', linewidth=1.5,
                   label=f'Early stop (epoch {stop})')
        ax.set_title(f'{label} Learning Curve', fontsize=12)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.legend()
    plt.suptitle('Training vs Validation RMSE per Epoch', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../images/learning_curves.png', dpi=150)
    plt.close()
    print("Saved learning_curves.png")

    grid_df = pd.read_csv('../results/ridge_alpha_grid.csv')
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
    best_alpha = ridge_m['alpha']
    ax.axvline(best_alpha, color='green', linestyle=':', linewidth=2,
               label=f'Best α={best_alpha}')
    ax.annotate('← Low α\n  (overfitting\n   high variance)',
                xy=(grid_df['alpha'].iloc[0], grid_df['val_rmse'].iloc[0]),
                xytext=(grid_df['alpha'].iloc[0] * 1.5,
                        grid_df['val_rmse'].iloc[0] + 30),
                fontsize=9, color='dimgray')
    ax.annotate('High α →\n(underfitting\n  high bias)',
                xy=(grid_df['alpha'].iloc[-1], grid_df['val_rmse'].iloc[-1]),
                xytext=(grid_df['alpha'].iloc[-2],
                        grid_df['val_rmse'].iloc[-1] + 30),
                fontsize=9, color='dimgray')
    ax.set_xlabel('Regularisation Strength α (log scale)')
    ax.set_ylabel('RMSE')
    ax.set_title('Ridge Regression: Bias–Variance Trade-off\n'
                 'Finding the sweet spot between underfitting and overfitting')
    ax.legend()
    plt.tight_layout()
    plt.savefig('../images/ridge_bias_variance.png', dpi=150)
    plt.close()
    print("Saved ridge_bias_variance.png")

    abs_coef = np.abs(ridge_coef)
    top_idx  = np.argsort(abs_coef)[-20:][::-1]
    top_names = [feat_names[i] for i in top_idx]
    top_vals  = ridge_coef[top_idx]

    colors = ['steelblue' if v >= 0 else 'tomato' for v in top_vals]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(len(top_names)), top_vals, color=colors)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=9)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Coefficient Value')
    ax.set_title('Ridge Regression — Top 20 Feature Coefficients\n'
                 '(blue = positive effect on traffic, red = negative)')
    plt.tight_layout()
    plt.savefig('../images/ridge_feature_importance.png', dpi=150)
    plt.close()
    print("Saved ridge_feature_importance.png")

    model_names = [f"Ridge\n(α={ridge_m['alpha']})", 'FNN', 'LSTM']
    test_rmses  = [ridge_m['test_rmse'], fnn_m['test_rmse'], lstm_m['test_rmse']]
    colors_bar  = ['coral', 'steelblue', 'seagreen']

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(model_names, test_rmses, color=colors_bar, width=0.5)
    for bar, val in zip(bars, test_rmses):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5, f'{val:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Test RMSE (vehicles/hour)')
    ax.set_title('Test RMSE by Model\n(lower is better)')
    ax.set_ylim(0, max(test_rmses) * 1.15)
    plt.tight_layout()
    plt.savefig('../images/rmse_comparison.png', dpi=150)
    plt.close()
    print("Saved rmse_comparison.png")

    print("\nAll plots saved to ../images/")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()
