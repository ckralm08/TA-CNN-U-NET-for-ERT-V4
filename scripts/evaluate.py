# =============================================================================
# scripts/evaluate.py
# Environment : TensorFlow
# Fungsi      : Evaluasi model inversi dan surrogate pada test set.
#
# Output:
#   results/metrics_summary.txt   -- ringkasan angka semua metrik
#   results/plots/sample_xx.png   -- visualisasi N sampel test
#   results/history_plots/        -- kurva training loss
# =============================================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import tensorflow as tf
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from utils.preprocessing  import load_config, ERTDataset, model_to_flat, denormalize
from utils.metrics        import evaluate_batch, print_metrics
from models.cnn_inversion   import CNNInversion
from models.forward_surrogate import SurrogateForward


# =============================================================================
# KONFIGURASI
# =============================================================================
CFG          = load_config()
RESULTS_DIR  = Path(CFG["evaluate"]["results_dir"])
N_PLOT       = CFG["evaluate"]["n_samples_plot"]
INV_PATH     = CFG["inversion"]["save_path"]
SURR_PATH    = CFG["surrogate"]["save_path"]
INV_HIST     = CFG["inversion"]["history_path"]
SURR_HIST    = CFG["surrogate"]["history_path"]

(RESULTS_DIR / "plots").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "history_plots").mkdir(parents=True, exist_ok=True)


# =============================================================================
# PLOT SATU SAMPEL
# =============================================================================
def plot_sample(idx: int,
                d_obs:  np.ndarray,
                m_true: np.ndarray,
                m_pred: np.ndarray,
                d_pred: np.ndarray,
                d_obs_vec: np.ndarray,
                metrics: dict,
                save_path: Path) -> None:
    """Buat figure 2x3 untuk satu sampel test."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Test Sample #{idx:04d}", fontsize=13, fontweight="bold")

    nz, nx = CFG["grid"]["nz"], CFG["grid"]["nx"]
    cmap   = "jet"
    vmin, vmax = 0.0, 1.0        # skala normalisasi

    def show(ax, data, title, cmap=cmap, vmin=vmin, vmax=vmax):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect="auto", origin="upper")
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("x (piksel)")
        ax.set_ylabel("z (piksel)")

    # Baris 1: domain model (resistivitas)
    show(axes[0, 0], d_obs[:, :, 0],  "d_obs  (pseudosection input)")
    show(axes[0, 1], m_true[:, :, 0], "m_true (true model)")
    show(axes[0, 2], m_pred[:, :, 0], "m_pred (CNN prediction)")

    # Baris 2: domain data dan error
    diff_model = np.abs(m_true[:, :, 0] - m_pred[:, :, 0])
    show(axes[1, 0], diff_model, "|m_true - m_pred|", cmap="hot_r", vmin=0, vmax=0.2)

    # Scatter d_obs vs d_pred
    n = min(len(d_obs_vec), len(d_pred))
    axes[1, 1].scatter(d_obs_vec[:n], d_pred[:n], s=4, alpha=0.5, color="#7F77DD")
    axes[1, 1].plot([0, 1], [0, 1], "r--", lw=1)
    axes[1, 1].set_xlabel("d_obs (norm)")
    axes[1, 1].set_ylabel("d_pred surrogate (norm)")
    axes[1, 1].set_title("Data domain: d_obs vs d_pred")
    axes[1, 1].set_xlim(0, 1); axes[1, 1].set_ylim(0, 1)

    # Teks metrik
    txt = "\n".join([
        f"MSE model  : {metrics['mse_model']:.6f}",
        f"MAE model  : {metrics['mae_model']:.6f}",
        f"R2         : {metrics['r2_model']:.4f}",
        f"SSIM       : {metrics['ssim']:.4f}",
        f"Rel.Err(m) : {metrics['rel_err_model']:.2f}%",
        f"MSE data   : {metrics['mse_data']:.6f}",
        f"Rel.Err(d) : {metrics['rel_err_data']:.2f}%",
    ])
    axes[1, 2].axis("off")
    axes[1, 2].text(0.1, 0.5, txt, transform=axes[1, 2].transAxes,
                    fontsize=10, fontfamily="monospace",
                    verticalalignment="center",
                    bbox=dict(boxstyle="round", facecolor="#f5f5f5", alpha=0.8))
    axes[1, 2].set_title("Metrik evaluasi")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# PLOT KURVA TRAINING
# =============================================================================
def plot_history(history: dict, title: str, keys_loss: list,
                 save_path: Path) -> None:
    fig, axes = plt.subplots(1, len(keys_loss), figsize=(5 * len(keys_loss), 4))
    if len(keys_loss) == 1:
        axes = [axes]
    for ax, (train_key, val_key, label) in zip(axes, keys_loss):
        if train_key in history:
            ax.plot(history[train_key], label=f"train {label}", color="#7F77DD")
        if val_key in history:
            ax.plot(history[val_key],   label=f"val {label}",   color="#1D9E75")
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "="*60)
    print("  EVALUASI MODEL ERT SURROGATE INVERSION")
    print("="*60)

    # Load model 
    for path, label in [(INV_PATH, "Inversion"), (SURR_PATH, "Surrogate")]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{label} model tidak ditemukan: {path}\n"
                f"Jalankan dulu train_forward.py dan train_inversion.py"
            )

    inversion = CNNInversion.load(INV_PATH)
    surrogate = SurrogateForward.load(SURR_PATH)
    surrogate.model.trainable = False
    print(f"  Inversion dimuat dari : {INV_PATH}")
    print(f"  Surrogate dimuat dari : {SURR_PATH}\n")

    # Test dataset 
    test_ds = ERTDataset("test", CFG)
    print(f"  Test samples : {len(test_ds)}\n")

    # Evaluasi seluruh test set
    all_m_true, all_m_pred = [], []
    all_d_obs,  all_d_pred = [], []

    for X, y, d_obs in test_ds.iter_batches(4, shuffle=False):
        X_tf  = tf.constant(X)
        m_pred = inversion(X_tf, training=False).numpy()

        B      = m_pred.shape[0]
        m_flat = m_pred.reshape(B, -1)
        d_pred = surrogate(tf.constant(m_flat), training=False).numpy()

        all_m_true.append(y)
        all_m_pred.append(m_pred)
        all_d_obs.append(d_obs)
        all_d_pred.append(d_pred)

    all_m_true = np.concatenate(all_m_true)
    all_m_pred = np.concatenate(all_m_pred)
    all_d_obs  = np.concatenate(all_d_obs)
    all_d_pred = np.concatenate(all_d_pred)

    # Sesuaikan panjang vektor d agar MSE bisa dihitung
    n_min = min(all_d_obs.shape[1], all_d_pred.shape[1])
    all_d_obs_trim  = all_d_obs[:,  :n_min]
    all_d_pred_trim = all_d_pred[:, :n_min]

    # Hitung metrik 
    metrics = evaluate_batch(all_m_true, all_m_pred,
                             all_d_obs_trim, all_d_pred_trim)
    print_metrics(metrics, title="Test Set Evaluation Results")

    # Simpan ke file teks
    with open(RESULTS_DIR / "metrics_summary.txt", "w") as f:
        f.write("ERT Surrogate Inversion -- Test Set Metrics\n")
        f.write("=" * 50 + "\n")
        for k, v in metrics.items():
            f.write(f"{k:<20}: {v:.6f}\n")
    print(f"  Metrik disimpan ke : {RESULTS_DIR / 'metrics_summary.txt'}")

    # Plot N sampel 
    print(f"\n  Membuat plot untuk {N_PLOT} sampel...")
    for i in range(min(N_PLOT, len(all_m_true))):
        from utils.metrics import evaluate_sample
        s_metrics = evaluate_sample(
            all_m_true[i], all_m_pred[i],
            all_d_obs_trim[i], all_d_pred_trim[i]
        )
        # Load d_obs asli (grid form) untuk plot
        X_i, y_i, d_i = test_ds.load(i)
        nz, nx = CFG["grid"]["nz"], CFG["grid"]["nx"]
        plot_sample(
            idx      = i,
            d_obs    = X_i,
            m_true   = y_i,
            m_pred   = all_m_pred[i],
            d_pred   = all_d_pred[i],
            d_obs_vec = all_d_obs_trim[i],
            metrics  = s_metrics,
            save_path = RESULTS_DIR / "plots" / f"sample_{i:04d}.png"
        )
        print(f"  Plot {i+1}/{N_PLOT}: R2={s_metrics['r2_model']:.4f}"
              f"  SSIM={s_metrics['ssim']:.4f}"
              f"  RelErr_data={s_metrics['rel_err_data']:.2f}%")

    # Plot kurva training
    print("\n  Membuat plot kurva training...")

    if Path(INV_HIST).exists():
        with open(INV_HIST, "rb") as f:
            inv_h = pickle.load(f)
        plot_history(inv_h, "CNN Inversion Training History",
            keys_loss=[
                ("train_total", "val_total", "Total Loss"),
                ("train_data",  "val_data",  "L_data (model)"),
                ("train_phys",  "val_phys",  "L_phys (physics)"),
            ],
            save_path=RESULTS_DIR / "history_plots" / "inversion_history.png")
        print(f"  Inversion history -> {RESULTS_DIR/'history_plots'/'inversion_history.png'}")

    if Path(SURR_HIST).exists():
        with open(SURR_HIST, "rb") as f:
            surr_h = pickle.load(f)
        plot_history(surr_h, "Surrogate Forward Training History",
            keys_loss=[
                ("train_loss", "val_mse", "MSE"),
            ],
            save_path=RESULTS_DIR / "history_plots" / "surrogate_history.png")
        print(f"  Surrogate history  -> {RESULTS_DIR/'history_plots'/'surrogate_history.png'}")

    print("\n" + "="*60)
    print("  EVALUASI SELESAI")
    print("="*60)
    print(f"  MSE model  : {metrics['mse_model']:.6f}")
    print(f"  MAE model  : {metrics['mae_model']:.6f}")
    print(f"  R2         : {metrics['r2_model']:.4f}")
    print(f"  SSIM       : {metrics['ssim']:.4f}")
    print(f"  MSE data   : {metrics['mse_data']:.6f}")
    print(f"  Semua hasil tersimpan di: {RESULTS_DIR}/")
    print("="*60)


if __name__ == "__main__":
    main()
