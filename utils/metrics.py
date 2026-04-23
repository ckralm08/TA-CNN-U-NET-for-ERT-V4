# =============================================================================
# utils/metrics.py
# Fungsi evaluasi kuantitatif untuk model inversi dan surrogate forward.
# =============================================================================

import numpy as np
import tensorflow as tf


# -----------------------------------------------------------------------------
# METRIK REKONSTRUKSI MODEL
# -----------------------------------------------------------------------------
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Koefisien determinasi R^2."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-10))


def relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Rata-rata kesalahan relatif (dalam persen)."""
    return float(np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-10)) * 100)


def structural_similarity_index(y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 data_range: float = 1.0) -> float:
    """
    Structural Similarity Index (SSIM) sederhana tanpa library eksternal.
    Mengukur kesamaan struktural antara dua gambar (grid model resistivitas).
    Nilai mendekati 1.0 berarti sangat mirip.
    """
    mu_t  = np.mean(y_true)
    mu_p  = np.mean(y_pred)
    sig_t = np.std(y_true)
    sig_p = np.std(y_pred)
    sig_tp = np.mean((y_true - mu_t) * (y_pred - mu_p))

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    num = (2 * mu_t * mu_p + c1) * (2 * sig_tp + c2)
    den = (mu_t ** 2 + mu_p ** 2 + c1) * (sig_t ** 2 + sig_p ** 2 + c2)
    return float(num / (den + 1e-10))


# -----------------------------------------------------------------------------
# METRIK SURROGATE (domain data)
# -----------------------------------------------------------------------------
def data_misfit(d_obs: np.ndarray, d_pred: np.ndarray) -> float:
    """
    Data misfit ternormalisasi — ukuran konsistensi fisika.
    phi_d = (1/N) * sum((d_obs - d_pred)^2)
    """
    return float(np.mean((d_obs - d_pred) ** 2))


def data_misfit_relative(d_obs: np.ndarray, d_pred: np.ndarray) -> float:
    """Data misfit relatif dalam persen."""
    return float(np.mean(np.abs(d_obs - d_pred) / (np.abs(d_obs) + 1e-10)) * 100)


# -----------------------------------------------------------------------------
# EVALUASI SATU SAMPEL
# -----------------------------------------------------------------------------
def evaluate_sample(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    d_obs: np.ndarray,
                    d_pred: np.ndarray) -> dict:
    """
    Hitung semua metrik untuk satu sampel.

    Parameters
    ----------
    y_true  : (NZ, NX) atau (NZ, NX, 1)  model resistivitas sejati (norm)
    y_pred  : sama shape, model prediksi
    d_obs   : (n_data,) vektor rhoa observasi (norm)
    d_pred  : (n_data,) vektor rhoa prediksi surrogate (norm)

    Returns
    -------
    dict berisi semua metrik
    """
    y_t = y_true.squeeze()
    y_p = y_pred.squeeze()

    return {
        # Domain model
        "mse_model"     : mse(y_t, y_p),
        "mae_model"     : mae(y_t, y_p),
        "rmse_model"    : rmse(y_t, y_p),
        "r2_model"      : r2_score(y_t, y_p),
        "rel_err_model" : relative_error(y_t, y_p),
        "ssim"          : structural_similarity_index(y_t, y_p),
        # Domain data (fisika)
        "mse_data"      : data_misfit(d_obs, d_pred),
        "rel_err_data"  : data_misfit_relative(d_obs, d_pred),
    }


# -----------------------------------------------------------------------------
# EVALUASI BATCH 
# -----------------------------------------------------------------------------
def evaluate_batch(y_true_batch: np.ndarray,
                   y_pred_batch: np.ndarray,
                   d_obs_batch:  np.ndarray,
                   d_pred_batch: np.ndarray) -> dict:
    """
    Hitung rata-rata metrik untuk seluruh batch/dataset.
    """
    results = [
        evaluate_sample(y_true_batch[i], y_pred_batch[i],
                        d_obs_batch[i],  d_pred_batch[i])
        for i in range(len(y_true_batch))
    ]
    keys = results[0].keys()
    return {k: float(np.mean([r[k] for r in results])) for k in keys}


# -----------------------------------------------------------------------------
# CETAK LAPORAN
# -----------------------------------------------------------------------------
def print_metrics(metrics: dict, title: str = "Evaluation Results") -> None:
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  {title}")
    print(f"{sep}")
    print(f"  --- Domain Model (Resistivitas) ---")
    print(f"  MSE          : {metrics['mse_model']:.6f}")
    print(f"  MAE          : {metrics['mae_model']:.6f}")
    print(f"  RMSE         : {metrics['rmse_model']:.6f}")
    print(f"  R2           : {metrics['r2_model']:.4f}")
    print(f"  Rel. Error   : {metrics['rel_err_model']:.2f}%")
    print(f"  SSIM         : {metrics['ssim']:.4f}")
    print(f"  --- Domain Data (Fisika/Surrogate) ---")
    print(f"  MSE data     : {metrics['mse_data']:.6f}")
    print(f"  Rel. Error   : {metrics['rel_err_data']:.2f}%")
    print(f"{sep}\n")
