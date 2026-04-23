# =============================================================================
# models/forward_surrogate.py
# Surrogate Forward Model  f_theta: m_pred  -->  d_pred
#
# Menggantikan PyGIMLi ert.simulate() dengan jaringan MLP yang terlatih.
# Input  : model resistivitas diratakan  (NZ * NX,)  ternormalisasi [0,1]
# Output : vektor apparent resistivity    (n_data,)   ternormalisasi [0,1]
#
# Keunggulan vs subprocess PyGIMLi:
#   - Differentiable secara penuh -> gradien L_phys mengalir langsung ke CNN
#   - Cepat: inferensi milidetik vs beberapa detik per batch
#   - Berjalan dalam satu environment TensorFlow
# =============================================================================

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, BatchNormalization, Activation,
    Dropout, LayerNormalization
)
from tensorflow.keras.models import Model
import numpy as np



# -----------------------------------------------------------------------------
# HELPER: hitung output_dim dari jumlah elektroda
# -----------------------------------------------------------------------------
def n_wenner_alpha(n_electrodes: int) -> int:
    """
    Hitung jumlah konfigurasi Wenner-Alpha untuk n_electrodes elektroda.

    Rumus: sum_{a=1}^{floor((N-1)/3)} max(N - 3a, 0)
    Contoh: N=48 -> 360,  N=36 -> 198,  N=60 -> 570
    """
    total = 0
    for a in range(1, n_electrodes):
        n = n_electrodes - 3 * a
        if n <= 0:
            break
        total += n
    return total


# -----------------------------------------------------------------------------
# BLOK RESIDUAL MLP
# Menggunakan skip connection agar gradien mengalir lebih lancar
# -----------------------------------------------------------------------------
def residual_block(x: tf.Tensor, units: int, dropout: float = 0.1) -> tf.Tensor:
    """
    Blok residual sederhana untuk MLP:
      x -> Dense -> LayerNorm -> ReLU -> Dropout -> Dense -> LayerNorm -> ReLU
      + skip connection (linear projection jika dimensi berbeda)
    """
    shortcut = x

    h = Dense(units, use_bias=False)(x)
    h = LayerNormalization()(h)
    h = Activation("relu")(h)
    h = Dropout(dropout)(h)
    h = Dense(units, use_bias=False)(h)
    h = LayerNormalization()(h)

    # Proyeksikan shortcut jika dimensi tidak sama
    if shortcut.shape[-1] != units:
        shortcut = Dense(units, use_bias=False)(shortcut)

    h = h + shortcut
    h = Activation("relu")(h)
    return h


# -----------------------------------------------------------------------------
# SURROGATE FORWARD MODEL
# -----------------------------------------------------------------------------
def build_surrogate(
    input_dim:   int   = 9280,     # NZ * NX = 40 * 232
    hidden_dims: list  = None,     # [1024, 512, 256]
    output_dim:  int   = 360,      # jumlah konfigurasi Wenner-Alpha (48 elektroda)
    dropout:     float = 0.1,
) -> Model:
    """
    Bangun surrogate forward model  f_theta: m  -->  d.

    Arsitektur: Input -> [ResidualBlock] x N -> Output (sigmoid)

    Parameters
    ----------
    input_dim   : dimensi input = NZ * NX (model diratakan)
    hidden_dims : list ukuran layer tersembunyi
    output_dim  : dimensi output = jumlah konfigurasi ERT (n_data)
    dropout     : dropout rate di setiap residual block

    Notes
    -----
    Jumlah konfigurasi Wenner-Alpha untuk N elektroda:
      n_data = sum_{a=1}^{floor((N-1)/3)} (N - 3a)
    Untuk N=48: n_data = 360  (bukan 658 -- rumus: sum_{a=1}^{15}(48-3a))
    """
    if hidden_dims is None:
        hidden_dims = [1024, 512, 256]

    inputs = Input(shape=(input_dim,), name="model_flat_input")

    # Proyeksi awal ke dimensi pertama
    x = Dense(hidden_dims[0], use_bias=False)(inputs)
    x = LayerNormalization()(x)
    x = Activation("relu")(x)

    # Residual blocks
    for units in hidden_dims:
        x = residual_block(x, units, dropout=dropout)

    # Output: sigmoid -> [0, 1] (skala normalisasi rhoa)
    outputs = Dense(output_dim, activation="sigmoid", name="rhoa_output")(x)

    return Model(inputs, outputs, name="ERT_Surrogate")


# -----------------------------------------------------------------------------
# WRAPPER KELAS untuk kemudahan penggunaan
# -----------------------------------------------------------------------------
class SurrogateForward:
    """
    Wrapper surrogate forward model dengan metode helper.

    Digunakan oleh train_inversion.py untuk menghitung L_phys
    secara differentiable dalam GradientTape.
    """
    def __init__(self, model: Model):
        self.model = model

    def __call__(self, m_flat: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Hitung d_pred dari m_flat.

        Parameters
        ----------
        m_flat   : tf.Tensor shape (B, NZ*NX)  model diratakan [0,1]
        training : bool  aktifkan dropout saat training

        Returns
        -------
        d_pred : tf.Tensor shape (B, n_data)   rhoa prediksi [0,1]
        """
        return self.model(m_flat, training=training)

    def predict_from_grid(self, m_grid: tf.Tensor,
                          training: bool = False) -> tf.Tensor:
        """
        Shortcut: terima m_grid (B, NZ, NX, 1) -> ratakan -> prediksi.
        """
        b    = tf.shape(m_grid)[0]
        flat = tf.reshape(m_grid, [b, -1])
        return self.model(flat, training=training)

    @classmethod
    def from_config(cls, cfg: dict) -> "SurrogateForward":
        """Bangun dari konfigurasi yaml."""
        model = build_surrogate(
            input_dim   = cfg["surrogate"]["input_dim"],
            hidden_dims = cfg["surrogate"]["hidden_dims"],
            output_dim  = cfg["surrogate"]["output_dim"],
            dropout     = cfg["surrogate"]["dropout"],
        )
        return cls(model)

    @classmethod
    def load(cls, path: str) -> "SurrogateForward":
        """Load model tersimpan dari file .keras."""
        model = tf.keras.models.load_model(path)
        return cls(model)

    def save(self, path: str) -> None:
        self.model.save(path)


# -----------------------------------------------------------------------------
# QUICK TEST
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from utils.preprocessing import load_config

    cfg   = load_config()
    surr  = SurrogateForward.from_config(cfg)
    surr.model.summary()

    # Test forward pass
    dummy_model = tf.random.uniform((4, cfg["surrogate"]["input_dim"]))
    d_pred = surr(dummy_model, training=False)
    print(f"\nInput shape  : {dummy_model.shape}")
    print(f"Output shape : {d_pred.shape}")
    assert d_pred.shape == (4, cfg["surrogate"]["output_dim"])
    print("Surrogate OK")
