# =============================================================================
# scripts/train_inversion.py
# Environment : TensorFlow
# Fungsi      : Training CNN inversion  g_phi: d_obs -> m_pred
#               dengan surrogate forward  f_theta: m_pred -> d_pred
#               yang sudah terlatih dan dibekukan.
#
# Pipeline per batch:
#   d_obs  --(g_phi)--> m_pred --(f_theta)--> d_pred
#
# Loss:
#   L_data  = MSE(m_pred, m_true)        domain model
#   L_phys  = MSE(d_pred, d_obs)         domain data (physics consistency)
#   L_total = lambda_data * L_data
#           + lambda_phys * L_phys
# Urutan training :
#   1. python scripts/generate_dataset.py      (env PyGIMLi)
#   2. python scripts/train_forward.py         (env TF)
#   3. python scripts/train_inversion.py       (env TF)
#   4. python scripts/evaluate.py              (env TF)
# =============================================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import tensorflow as tf
import pickle
from pathlib import Path

from utils.preprocessing import load_config, ERTDataset, model_to_flat
from models.cnn_inversion   import CNNInversion
from models.forward_surrogate import SurrogateForward


# =============================================================================
# KONFIGURASI
# =============================================================================
CFG          = load_config()
INV_CFG      = CFG["inversion"]
EPOCHS       = INV_CFG["epochs"]
BATCH_SIZE   = INV_CFG["batch_size"]
LR           = INV_CFG["learning_rate"]
PATIENCE     = INV_CFG["patience"]
LAM_DATA     = INV_CFG["lambda_data"]
LAM_PHYS     = INV_CFG["lambda_physics"]
SAVE_PATH    = INV_CFG["save_path"]
HISTORY_PATH = INV_CFG["history_path"]

SURROGATE_PATH = CFG["surrogate"]["save_path"]

Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)


# =============================================================================
# TRAINING STEP — FULLY DIFFERENTIABLE
# =============================================================================
@tf.function
def train_step(inversion, surrogate_model,
               X_obs, m_true, d_obs_vec,
               optimizer,
               lam_data: float, lam_phys: float):
    """
    Satu langkah gradient descent — fully differentiable.

    Pipeline: X_obs --(g_phi)--> m_pred --(f_theta)--> d_pred
    Seluruhnya berada dalam satu GradientTape sehingga gradien
    L_phys mengalir melalui f_theta ke parameter g_phi.

    Parameters
    ----------
    inversion       : CNNInversion   g_phi
    surrogate_model : tf.keras.Model f_theta (dibekukan, tidak diupdate)
    X_obs    : tf.Tensor (B, NZ, NX, 1)  pseudosection grid  [input CNN]
    m_true   : tf.Tensor (B, NZ, NX, 1)  model resistivitas sejati
    d_obs_vec: tf.Tensor (B, n_data)      vektor rhoa observasi [dari d_xxxx.npy]
    optimizer       : tf.keras.optimizers
    lam_data : float  bobot L_data  (domain model)
    lam_phys : float  bobot L_phys  (domain data / fisika)

    Returns
    -------
    l_total, l_data, l_phys, d_pred
    """
    with tf.GradientTape() as tape:
        # g_phi: pseudosection -> model resistivitas prediksi
        m_pred = inversion(X_obs, training=True)           # (B, NZ, NX, 1)

        # Ratakan untuk surrogate
        B      = tf.shape(m_pred)[0]
        m_flat = tf.reshape(m_pred, [B, -1])               # (B, NZ*NX)

        # f_theta: model prediksi -> apparent resistivity prediksi
        # Dibekukan tapi di dalam tape -> gradien L_phys tetap mengalir
        # melalui f_theta ke bobot g_phi
        d_pred = surrogate_model(m_flat, training=False)   # (B, n_data)

        # L_data: CNN cocok dengan label true model
        l_data  = tf.reduce_mean(tf.square(m_true - m_pred))
        # L_phys: prediksi konsisten secara fisika dengan data observasi
        l_phys  = tf.reduce_mean(tf.square(d_obs_vec - d_pred))
        l_total = lam_data * l_data + lam_phys * l_phys

    # Update hanya bobot g_phi (inversion), bukan f_theta (surrogate)
    grads = tape.gradient(l_total, inversion.trainable_variables)
    optimizer.apply_gradients(zip(grads, inversion.trainable_variables))

    return l_total, l_data, l_phys, d_pred
# =============================================================================
# VALIDASI
# =============================================================================
def validate_inversion(inversion, surrogate_model, val_ds, lam_data, lam_phys):
    total = {"l_total": 0.0, "l_data": 0.0, "l_phys": 0.0, "mae": 0.0}
    n = 0
    for X, y, d_obs in val_ds.iter_batches(INV_CFG["batch_size"], shuffle=False):
        X_tf  = tf.constant(X)
        y_tf  = tf.constant(y)

        m_pred    = inversion(X_tf, training=False)
        B         = tf.shape(m_pred)[0]
        m_flat    = tf.reshape(m_pred, [B, -1])
        d_pred    = surrogate_model(m_flat, training=False)
        d_obs_tf  = tf.constant(d_obs)

        n_data    = tf.shape(d_pred)[1]
        d_obs_vec = tf.reshape(d_obs_tf, [B, -1])[:, :n_data]

        l_data  = float(tf.reduce_mean(tf.square(y_tf - m_pred)))
        l_phys  = float(tf.reduce_mean(tf.square(d_obs_vec - d_pred)))
        l_total = lam_data * l_data + lam_phys * l_phys
        mae_v   = float(tf.reduce_mean(tf.abs(y_tf - m_pred)))

        total["l_total"] += l_total
        total["l_data"]  += l_data
        total["l_phys"]  += l_phys
        total["mae"]     += mae_v
        n += 1

    return {k: v / max(n, 1) for k, v in total.items()}


# =============================================================================
# LAPORAN PER EPOCH
# =============================================================================
def print_report(epoch, epochs, ep, val, best, patience_cnt, history):
    def delta(cur, prev):
        if prev is None:
            return "  (epoch pertama)"
        d   = cur - prev
        pct = d / abs(prev) * 100
        arr = "v" if d < 0 else "^"
        bar = "|" * min(int(abs(pct) / 2), 20) or "."
        tag = "[TURUN SIGNIFIKAN]" if pct <= -1 else \
              "[turun sedikit]"    if pct <= -0.1 else \
              "[STAGNAN]"          if abs(pct) < 0.1 else \
              "[NAIK -- periksa!]"
        return f"  {d:+.6f}  ({pct:+.1f}%)  [{arr}] {bar}  {tag}"

    prev = lambda key: history[key][-2] if len(history[key]) >= 2 else None

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  INVERSION EPOCH {epoch}/{epochs}")
    print(f"  lam_data={LAM_DATA:.4f}  lam_phys={LAM_PHYS:.4f}")
    print(f"{sep}")
    print(f"  train_total : {ep['l_total']:.6f}" + delta(ep["l_total"], prev("train_total")))
    print(f"  train_data  : {ep['l_data']:.6f}"  + delta(ep["l_data"],  prev("train_data")))
    print(f"  train_phys  : {ep['l_phys']:.6f}"  + delta(ep["l_phys"],  prev("train_phys")))
    print(f"  kontrib_d   : {LAM_DATA*ep['l_data']:.6f}  (lam_data x L_data)")
    print(f"  kontrib_p   : {LAM_PHYS*ep['l_phys']:.6f}  (lam_phys x L_phys)")
    print(f"  {'─'*56}")
    print(f"  val_total   : {val['l_total']:.6f}" + delta(val["l_total"], prev("val_total")))
    print(f"  val_data    : {val['l_data']:.6f}"  + delta(val["l_data"],  prev("val_data")))
    print(f"  val_phys    : {val['l_phys']:.6f}")
    print(f"  val_mae     : {val['mae']:.6f}")
    print(f"  {'─'*56}")
    gap = val["l_data"] - ep["l_data"]
    if gap < 0:
        gap_tag = "[val < train -- dropout effect]"
    elif gap > ep["l_data"] * 0.5:
        gap_tag = "[gap besar -- indikasi overfitting]"
    else:
        gap_tag = "[gap wajar]"
    print(f"  gap         : {gap:+.6f}  {gap_tag}")
    print(f"  best val    : {best:.6f}  patience={patience_cnt}/{PATIENCE}")
    if len(history["val_total"]) >= 3:
        t  = history["val_total"][-3:]
        tr = "menurun" if t[-1] < t[0] else "naik/stagnan"
        print(f"  tren(3ep)   : {t[0]:.6f} -> {t[1]:.6f} -> {t[2]:.6f}  ({tr})")
    print(f"{sep}\n")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "="*60)
    print("  TRAINING CNN INVERSION  g_phi: d_obs -> m_pred")
    print("  dengan surrogate forward  f_theta: m_pred -> d_pred")
    print("="*60)
    print(f"  Surrogate    : {SURROGATE_PATH}")
    print(f"  lambda_data  : {LAM_DATA}")
    print(f"  lambda_phys  : {LAM_PHYS}")
    print(f"  Epochs       : {EPOCHS}")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  LR           : {LR}")
    print("="*60 + "\n")

    # ── Load surrogate (dibekukan — bobotnya tidak ikut diupdate) ─────────────
    if not Path(SURROGATE_PATH).exists():
        raise FileNotFoundError(
            f"Surrogate tidak ditemukan: {SURROGATE_PATH}\n"
            f"Jalankan dulu: python scripts/train_forward.py"
        )
    surrogate     = SurrogateForward.load(SURROGATE_PATH)
    surrogate.model.trainable = False     # BEKUKAN surrogate
    print(f"  Surrogate dimuat dan dibekukan dari: {SURROGATE_PATH}")
    print(f"  Surrogate trainable: {surrogate.model.trainable}\n")

    # ── CNN inversion ─────────────────────────────────────────────────────────
    inversion = CNNInversion.from_config(CFG)
    inversion.model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_ds = ERTDataset("train", CFG)
    val_ds   = ERTDataset("val",   CFG)
    print(f"  Train samples : {len(train_ds)}")
    print(f"  Val   samples : {len(val_ds)}\n")

    # ── History ───────────────────────────────────────────────────────────────
    history = {
        "train_total": [], "train_data": [], "train_phys": [],
        "val_total":   [], "val_data":   [], "val_phys":   [],
        "val_mae":     [],
        "lam_data":    [], "lam_phys":   [],
    }
    best_val  = np.inf
    patience  = 0

    # =========================================================================
    # EPOCH LOOP
    # =========================================================================
    for epoch in range(1, EPOCHS + 1):

        # ── Training ──────────────────────────────────────────────────────────
        ep = {"l_total": 0.0, "l_data": 0.0, "l_phys": 0.0}
        n_batch = 0

        for X, y, d_obs in train_ds.iter_batches(BATCH_SIZE, shuffle=True):
            X_tf     = tf.constant(X)
            y_tf     = tf.constant(y)
            d_obs_tf = tf.constant(d_obs)   # (B, n_data) vektor rhoa asli

            l_total, l_data, l_phys, _ = train_step(
                inversion, surrogate.model,
                X_tf, y_tf, d_obs_tf,
                optimizer,
                float(LAM_DATA), float(LAM_PHYS)
            )
            ep["l_total"] += float(l_total)
            ep["l_data"]  += float(l_data)
            ep["l_phys"]  += float(l_phys)
            n_batch += 1

        for k in ep:
            ep[k] /= max(n_batch, 1)

        # ── Validasi ──────────────────────────────────────────────────────────
        val = validate_inversion(
            inversion, surrogate.model, val_ds, LAM_DATA, LAM_PHYS
        )

        # ── Rekam history ─────────────────────────────────────────────────────
        history["train_total"].append(ep["l_total"])
        history["train_data"].append(ep["l_data"])
        history["train_phys"].append(ep["l_phys"])
        history["val_total"].append(val["l_total"])
        history["val_data"].append(val["l_data"])
        history["val_phys"].append(val["l_phys"])
        history["val_mae"].append(val["mae"])
        history["lam_data"].append(LAM_DATA)
        history["lam_phys"].append(LAM_PHYS)

        # ── Checkpoint & early stopping ───────────────────────────────────────
        monitor = val["l_data"]       # pantau domain model untuk checkpoint
        if monitor < best_val:
            best_val  = monitor
            patience  = 0
            inversion.save(SAVE_PATH)
            saved = True
        else:
            patience += 1
            saved = False

        print_report(epoch, EPOCHS, ep, val, best_val, patience, history)

        if saved:
            print(f"  [OK] Best inversion tersimpan  val_data={monitor:.6f}")
        else:
            print(f"  [..] Tidak ada perbaikan  patience={patience}/{PATIENCE}")
            if patience >= PATIENCE:
                print(f"\n  [STOP] Early stopping epoch {epoch}")
                break

    # ── Simpan history ────────────────────────────────────────────────────────
    with open(HISTORY_PATH, "wb") as f:
        pickle.dump(history, f)

    print("\n" + "="*60)
    print("  INVERSION TRAINING SELESAI")
    print("="*60)
    print(f"  Best val_data   : {best_val:.6f}")
    print(f"  Model tersimpan : {SAVE_PATH}")
    print(f"  History         : {HISTORY_PATH}")
    print("="*60)


if __name__ == "__main__":
    main()
