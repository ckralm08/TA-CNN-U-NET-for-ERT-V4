# =============================================================================
# utils/preprocessing.py
# Utilitas normalisasi, denormalisasi, dan loading dataset.
# Digunakan oleh semua script training dan evaluasi.
# =============================================================================

import numpy as np
import os
from pathlib import Path
import yaml


# -----------------------------------------------------------------------------
# LOAD CONFIG
# -----------------------------------------------------------------------------
def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# -----------------------------------------------------------------------------
# NORMALISASI / DENORMALISASI
# Skema: log10(rho) dipetakan ke [0, 1]
#   norm(rho) = (log10(rho) - log10(rho_min)) / (log10(rho_max) - log10(rho_min))
# -----------------------------------------------------------------------------
def get_log_range(cfg: dict):
    rho_min = cfg["domain"]["rho_min"]
    rho_max = cfg["domain"]["rho_max"]
    return np.log10(rho_min), np.log10(rho_max)


def normalize(rho: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Normalisasi resistivitas ke [0, 1] dalam skala log.
    Berlaku untuk nilai rho (Ohm.m) maupun rhoa (apparent resistivity).
    """
    log_min, log_max = get_log_range(cfg)
    rho   = np.clip(rho, cfg["domain"]["rho_min"], cfg["domain"]["rho_max"])
    return (np.log10(rho) - log_min) / (log_max - log_min)


def denormalize(rho_norm: np.ndarray, cfg: dict) -> np.ndarray:
    """Kembalikan nilai normalisasi ke resistivitas Ohm.m."""
    log_min, log_max = get_log_range(cfg)
    return 10.0 ** (log_min + rho_norm * (log_max - log_min))


def normalize_rhoa_vector(rhoa: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Normalisasi vektor rhoa (output forward modeling) ke [0, 1].
    Sama formula dengan normalize() — dipisah untuk kejelasan semantik.
    """
    return normalize(rhoa, cfg)


def denormalize_rhoa_vector(rhoa_norm: np.ndarray, cfg: dict) -> np.ndarray:
    """Kembalikan vektor rhoa ternormalisasi ke Ohm.m."""
    return denormalize(rhoa_norm, cfg)


# -----------------------------------------------------------------------------
# DATASET LOADER
# -----------------------------------------------------------------------------
class ERTDataset:
    """
    Loader dataset ERT dari direktori processed.

    Struktur direktori:
        processed/
          train/
            X/       X_0000.npy  ...  pseudosection grid  (NZ, NX, 1)  [0,1]
            y/       y_0000.npy  ...  true model grid     (NZ, NX, 1)  [0,1]
            d_obs/   d_0000.npy  ...  rhoa vektor obs     (n_data,)    [0,1]
          val/  ...
          test/ ...
    """
    def __init__(self, split: str, cfg: dict):
        base  = Path(cfg["dataset"]["processed_dir"]) / split
        self.X_dir    = base / "X"
        self.y_dir    = base / "y"
        self.d_dir    = base / "d_obs"
        self.files    = sorted(os.listdir(self.X_dir))
        self.split    = split
        self.n        = len(self.files)

    def __len__(self):
        return self.n

    def load(self, idx: int):
        """
        Load satu sampel.

        Returns
        -------
        X     : np.ndarray (NZ, NX, 1)   input CNN (pseudosection, norm)
        y     : np.ndarray (NZ, NX, 1)   label model resistivitas (norm)
        d_obs : np.ndarray (n_data,)     vektor rhoa observasi (norm)
        """
        fname = self.files[idx]
        X     = np.load(self.X_dir / fname)
        y     = np.load(self.y_dir / fname.replace("X_", "y_"))
        d_obs = np.load(self.d_dir / fname.replace("X_", "d_"))
        return X.astype(np.float32), y.astype(np.float32), d_obs.astype(np.float32)

    def load_batch(self, indices):
        """Load sekumpulan sampel sekaligus."""
        Xs, ys, ds = [], [], []
        for i in indices:
            X, y, d = self.load(i)
            Xs.append(X); ys.append(y); ds.append(d)
        return (
            np.array(Xs, dtype=np.float32),
            np.array(ys, dtype=np.float32),
            np.array(ds, dtype=np.float32),
        )

    def iter_batches(self, batch_size: int, shuffle: bool = True):
        """Generator yang menghasilkan batch (X, y, d_obs) hingga dataset habis."""
        idx = np.arange(self.n)
        if shuffle:
            np.random.shuffle(idx)
        for start in range(0, self.n, batch_size):
            yield self.load_batch(idx[start:start + batch_size])


# -----------------------------------------------------------------------------
# MODEL FLATTENER (untuk surrogate)
# -----------------------------------------------------------------------------
def model_to_flat(y_grid: np.ndarray) -> np.ndarray:
    """
    Ratakan grid model (NZ, NX, 1) atau (B, NZ, NX, 1) menjadi vektor 1D.

    Returns shape (NZ*NX,) atau (B, NZ*NX).
    """
    if y_grid.ndim == 3:
        return y_grid.reshape(-1)
    return y_grid.reshape(y_grid.shape[0], -1)


def flat_to_model(flat: np.ndarray, nz: int, nx: int) -> np.ndarray:
    """
    Kembalikan vektor flat menjadi grid (NZ, NX, 1) atau (B, NZ, NX, 1).
    """
    if flat.ndim == 1:
        return flat.reshape(nz, nx, 1)
    return flat.reshape(flat.shape[0], nz, nx, 1)
