# =============================================================================
# models/cnn_inversion.py
# CNN Inversion Model  g_phi: d_obs  -->  m_pred
#
# Arsitektur U-Net dengan skip connections.
# Input  : pseudosection apparent resistivity  (NZ, NX, 1)  [0, 1]
# Output : distribusi resistivitas prediksi    (NZ, NX, 1)  [0, 1]
# =============================================================================

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D,
    Conv2DTranspose, Concatenate,
    BatchNormalization, Activation, Dropout
)
from tensorflow.keras.models import Model


# -----------------------------------------------------------------------------
# BLOK KONVOLUSI GANDA  (Conv -> BN -> ReLU) x 2
# -----------------------------------------------------------------------------
def conv_block(x: tf.Tensor,
               filters: int,
               dropout_rate: float = 0.0) -> tf.Tensor:
    """
    Dua lapisan Conv2D 3x3 dengan Batch Normalization dan ReLU.
    Dropout opsional setelah blok kedua (digunakan di bottleneck).
    """
    for _ in range(2):
        x = Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    if dropout_rate > 0.0:
        x = Dropout(dropout_rate)(x)
    return x


# -----------------------------------------------------------------------------
# U-NET INVERSION
# -----------------------------------------------------------------------------
def build_cnn_inversion(
    input_shape:        tuple = (40, 232, 1),
    base_filters:       int   = 16,
    dropout_bottleneck: float = 0.3,
) -> Model:
    """
    Bangun U-Net untuk inversi ERT.

    Arsitektur
    ----------
    Encoder    : 3 level   (16 -> 32 -> 64 filter)  + MaxPool2D
    Bottleneck : 128 filter + Dropout
    Decoder    : 3 level   Conv2DTranspose + skip connection (Concatenate)
    Output     : Conv2D 1x1 + sigmoid  -> [0, 1]

    Parameters
    ----------
    input_shape        : (NZ, NX, 1)
    base_filters       : jumlah filter awal; berlipat 2x tiap level
    dropout_bottleneck : dropout rate di bottleneck
    """
    f      = base_filters
    inputs = Input(shape=input_shape, name="d_obs_input")

    # ── Encoder ──────────────────────────────────────────────────────────────
    c1 = conv_block(inputs, f * 1)
    p1 = MaxPooling2D(name="pool1")(c1)

    c2 = conv_block(p1, f * 2)
    p2 = MaxPooling2D(name="pool2")(c2)

    c3 = conv_block(p2, f * 4)
    p3 = MaxPooling2D(name="pool3")(c3)

    # ── Bottleneck ────────────────────────────────────────────────────────────
    b  = conv_block(p3, f * 8, dropout_rate=dropout_bottleneck)

    # ── Decoder ───────────────────────────────────────────────────────────────
    u3 = Conv2DTranspose(f * 4, 2, strides=2, padding="same", name="up3")(b)
    u3 = Concatenate(name="skip3")([u3, c3])
    c4 = conv_block(u3, f * 4)

    u2 = Conv2DTranspose(f * 2, 2, strides=2, padding="same", name="up2")(c4)
    u2 = Concatenate(name="skip2")([u2, c2])
    c5 = conv_block(u2, f * 2)

    u1 = Conv2DTranspose(f * 1, 2, strides=2, padding="same", name="up1")(c5)
    u1 = Concatenate(name="skip1")([u1, c1])
    c6 = conv_block(u1, f * 1)

    # ── Output ────────────────────────────────────────────────────────────────
    outputs = Conv2D(
        1, kernel_size=1,
        activation="sigmoid",
        name="m_pred_output"
    )(c6)

    return Model(inputs, outputs, name="ERT_CNN_Inversion")


# -----------------------------------------------------------------------------
# DENORMALISASI HELPER (TensorFlow)
# -----------------------------------------------------------------------------
_LOG_MIN = tf.constant(
    tf.math.log(0.5)   / tf.math.log(10.0), dtype=tf.float32
)
_LOG_MAX = tf.constant(
    tf.math.log(450.0) / tf.math.log(10.0), dtype=tf.float32
)


def denormalize_rho(rho_norm: tf.Tensor) -> tf.Tensor:
    """
    Kembalikan nilai ternormalisasi ke resistivitas dalam Ohm.m.
    rho_norm in [0, 1]  ->  10^(LOG_MIN + rho_norm * (LOG_MAX - LOG_MIN))
    """
    log_rho = _LOG_MIN + rho_norm * (_LOG_MAX - _LOG_MIN)
    return tf.pow(10.0, log_rho)


# -----------------------------------------------------------------------------
# WRAPPER KELAS
# -----------------------------------------------------------------------------
class CNNInversion:
    """
    Wrapper U-Net inversion dengan metode helper.
    """
    def __init__(self, model: Model):
        self.model = model

    def __call__(self, d_obs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Prediksi model resistivitas dari pseudosection.

        Parameters
        ----------
        d_obs    : tf.Tensor (B, NZ, NX, 1)  pseudosection [0,1]
        training : bool

        Returns
        -------
        m_pred : tf.Tensor (B, NZ, NX, 1)  model prediksi [0,1]
        """
        return self.model(d_obs, training=training)

    @classmethod
    def from_config(cls, cfg: dict) -> "CNNInversion":
        shape  = cfg["inversion"]["input_shape"]
        model  = build_cnn_inversion(
            input_shape        = tuple(shape),
            base_filters       = cfg["inversion"]["base_filters"],
            dropout_bottleneck = cfg["inversion"]["dropout_bottleneck"],
        )
        return cls(model)

    @classmethod
    def load(cls, path: str) -> "CNNInversion":
        model = tf.keras.models.load_model(path)
        return cls(model)

    def save(self, path: str) -> None:
        self.model.save(path)

    @property
    def trainable_variables(self):
        return self.model.trainable_variables


# -----------------------------------------------------------------------------
# QUICK TEST
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from utils.preprocessing import load_config

    cfg = load_config()
    inv = CNNInversion.from_config(cfg)
    inv.model.summary()

    shape = tuple(cfg["inversion"]["input_shape"])
    dummy = tf.zeros((2, *shape))
    out   = inv(dummy, training=False)
    print(f"\nInput  : {dummy.shape}")
    print(f"Output : {out.shape}")
    assert out.shape == (2, *shape)
    print("CNN Inversion OK")
