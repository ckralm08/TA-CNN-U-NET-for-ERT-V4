# =============================================================================
# scripts/train_forward.py
# Environment : TensorFlow
# Fungsi      : Training surrogate forward model  f_theta: m -> d
#
# Surrogate dilatih terpisah sebelum training inversion.
# Input  : model resistivitas diratakan  y_grid -> flat  (NZ*NX,)  [0,1]
# Output : vektor apparent resistivity                    (n_data,) [0,1]
#
# Loss  : MSE(d_true, d_pred) 
#
# Urutan training:
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
from models.forward_surrogate import SurrogateForward


# =============================================================================
# KONFIGURASI
# =============================================================================
CFG          = load_config()
S_CFG        = CFG["surrogate"]
EPOCHS       = S_CFG["epochs"]
BATCH_SIZE   = S_CFG["batch_size"]
LR           = S_CFG["learning_rate"]
PATIENCE     = S_CFG["patience"]
SAVE_PATH    = S_CFG["save_path"]
HISTORY_PATH = S_CFG["history_path"]

Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)


# =============================================================================
# LOSS & TRAINING STEP
# =============================================================================
@tf.function
def train_step(model, optimizer, m_flat, d_true):
    with tf.GradientTape() as tape:
        d_pred = model(m_flat, training=True)
        loss   = tf.reduce_mean(tf.square(d_true - d_pred))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def validate_surrogate(surrogate, val_ds):
    total_mse, total_mae, n = 0.0, 0.0, 0
    for X, y, d_obs in val_ds.iter_batches(S_CFG["batch_size"], shuffle=False):
        m_flat = model_to_flat(y)                            # (B, NZ*NX)
        d_pred = surrogate(tf.constant(m_flat), training=False).numpy()
        total_mse += float(np.mean((d_obs - d_pred) ** 2))
        total_mae += float(np.mean(np.abs(d_obs - d_pred)))
        n += 1
    return total_mse / n, total_mae / n


# =============================================================================
# LAPORAN PER EPOCH
# =============================================================================
def print_report(epoch: int, epochs: int,
                 loss: float, val_mse: float, val_mae: float,
                 best: float, patience: int, history: dict) -> None:
    prev_loss    = history["train_loss"][-2]  if len(history["train_loss"]) >= 2 else None
    prev_val_mse = history["val_mse"][-2]     if len(history["val_mse"])    >= 2 else None

    def delta(cur, prev):
        if prev is None:
            return "(epoch pertama)"
        d   = cur - prev
        pct = d / abs(prev) * 100
        arr = "v" if d < 0 else "^"
        bar = "|" * min(int(abs(pct) / 2), 20) or "."
        tag = "[TURUN SIGNIFIKAN]" if pct <= -1 else \
              "[turun sedikit]"    if pct <= -0.1 else \
              "[STAGNAN]"          if abs(pct) < 0.1 else \
              "[NAIK -- periksa!]"
        return f"  {d:+.6f}  ({pct:+.1f}%)  [{arr}] {bar}  {tag}"

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  SURROGATE EPOCH {epoch}/{epochs}")
    print(f"{sep}")
    print(f"  train_loss : {loss:.6f}" + delta(loss, prev_loss))
    print(f"  val_mse    : {val_mse:.6f}" + delta(val_mse, prev_val_mse))
    print(f"  val_mae    : {val_mae:.6f}")
    print(f"  best_val   : {best:.6f}  patience={patience}/{PATIENCE}")
    if len(history["val_mse"]) >= 3:
        t = history["val_mse"][-3:]
        tr = "menurun" if t[-1] < t[0] else "naik/stagnan"
        print(f"  tren(3ep)  : {t[0]:.6f} -> {t[1]:.6f} -> {t[2]:.6f}  ({tr})")
    print(f"{sep}\n")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "="*60)
    print("  TRAINING SURROGATE FORWARD MODEL  f_theta: m -> d")
    print("="*60)
    print(f"  Input dim    : {S_CFG['input_dim']}")
    print(f"  Hidden dims  : {S_CFG['hidden_dims']}")
    print(f"  Output dim   : {S_CFG['output_dim']}")
    print(f"  Epochs       : {EPOCHS}")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  LR           : {LR}")
    print("="*60 + "\n")

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_ds = ERTDataset("train", CFG)
    val_ds   = ERTDataset("val",   CFG)
    print(f"  Train samples : {len(train_ds)}")
    print(f"  Val   samples : {len(val_ds)}\n")

    # ── Model & optimizer ─────────────────────────────────────────────────────
    surrogate = SurrogateForward.from_config(CFG)
    surrogate.model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

    # ── History ───────────────────────────────────────────────────────────────
    history = {"train_loss": [], "val_mse": [], "val_mae": []}
    best_val  = np.inf
    patience  = 0

    # =========================================================================
    # EPOCH LOOP
    # =========================================================================
    for epoch in range(1, EPOCHS + 1):

        # ── Training batches ──────────────────────────────────────────────────
        ep_loss = 0.0
        n_batch = 0
        for X, y, d_obs in train_ds.iter_batches(BATCH_SIZE, shuffle=True):
            # Surrogate input = model diratakan (bukan pseudosection)
            m_flat = tf.constant(model_to_flat(y))   # (B, NZ*NX)
            d_true = tf.constant(d_obs)               # (B, n_data)

            loss    = train_step(surrogate.model, optimizer, m_flat, d_true)
            ep_loss += float(loss)
            n_batch += 1

        ep_loss /= max(n_batch, 1)

        # ── Validasi ──────────────────────────────────────────────────────────
        val_mse, val_mae = validate_surrogate(surrogate, val_ds)

        history["train_loss"].append(ep_loss)
        history["val_mse"].append(val_mse)
        history["val_mae"].append(val_mae)

        # ── Checkpoint & early stopping ───────────────────────────────────────
        if val_mse < best_val:
            best_val = val_mse
            patience = 0
            surrogate.save(SAVE_PATH)
            saved = True
        else:
            patience += 1
            saved = False

        print_report(epoch, EPOCHS, ep_loss, val_mse, val_mae,
                     best_val, patience, history)

        if saved:
            print(f"  [OK] Best surrogate tersimpan  val_mse={val_mse:.6f}")
        else:
            print(f"  [..] Tidak ada perbaikan  patience={patience}/{PATIENCE}")
            if patience >= PATIENCE:
                print(f"\n  [STOP] Early stopping epoch {epoch}")
                break

    # ── Simpan history ────────────────────────────────────────────────────────
    with open(HISTORY_PATH, "wb") as f:
        pickle.dump(history, f)

    print("\n" + "="*60)
    print("  SURROGATE TRAINING SELESAI")
    print("="*60)
    print(f"  Best val_mse  : {best_val:.6f}")
    print(f"  Model tersimpan : {SAVE_PATH}")
    print(f"  History         : {HISTORY_PATH}")
    print("="*60)


if __name__ == "__main__":
    main()
