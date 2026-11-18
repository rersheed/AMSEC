"""Optional TensorFlow/Keras builders copied from the PhD notebook.

These are the authoritative graph definitions for notebook reproduction when
TensorFlow is installed (Python ≤3.12 recommended). On Python 3.14 hosts without
TF, callers must use the PyTorch ports and label ``backend=pytorch_notebook_port``.
"""
from __future__ import annotations


def keras_available() -> bool:
    try:
        import tensorflow  # noqa: F401

        return True
    except Exception:
        return False


def notebook_flops(param_count: int) -> int:
    """Notebook complexity rule used in cells 2/4: FLOPs = 2 × parameters."""
    return int(param_count) * 2


def build_mobilenetv4_keras(input_dim: int, num_classes: int = 2, use_attention: bool = False):
    if not keras_available():
        raise RuntimeError("TensorFlow/Keras is not available in this environment")
    from tensorflow.keras import Model, regularizers
    from tensorflow.keras.layers import (
        BatchNormalization,
        Concatenate,
        Conv1D,
        Dense,
        Dropout,
        Flatten,
        GlobalAveragePooling1D,
        GlobalMaxPooling1D,
        Input,
        LayerNormalization,
        MaxPooling1D,
        MultiHeadAttention,
        Reshape,
    )
    from tensorflow.keras.regularizers import l2

    inputs = Input(shape=(input_dim,))
    x = BatchNormalization()(inputs)
    x = Reshape((input_dim, 1))(x)
    for filters, drop in ((48, 0.12), (64, 0.12)):
        x = Conv1D(filters, 3, padding="same", activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(5e-5))(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(drop)(x)
    x = Conv1D(96, 3, padding="same", activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(5e-5))(x)
    x = BatchNormalization()(x)
    x = Concatenate()([GlobalAveragePooling1D()(x), GlobalMaxPooling1D()(x)])
    x = Dropout(0.2)(x)
    x = Dense(128, activation="relu", kernel_regularizer=l2(5e-5), kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Dense(64, activation="relu", kernel_regularizer=l2(5e-5), kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    if use_attention:
        attn_input = Reshape((1, 64))(x)
        attn_out = MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(attn_input, attn_input)
        attn_out = Flatten()(attn_out)
        attn_out = LayerNormalization()(attn_out)
        x = Concatenate()([x, attn_out])
        x = Dense(64, activation="relu", kernel_regularizer=l2(5e-5), kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.15)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs)


def build_lightweight_keras(input_dim: int, num_classes: int = 2):
    if not keras_available():
        raise RuntimeError("TensorFlow/Keras is not available in this environment")
    from tensorflow.keras import Model
    from tensorflow.keras.layers import (
        BatchNormalization,
        Concatenate,
        Conv1D,
        Dense,
        Dropout,
        GlobalAveragePooling1D,
        GlobalMaxPooling1D,
        Input,
        MaxPooling1D,
        Reshape,
    )
    from tensorflow.keras.regularizers import l2

    inputs = Input(shape=(input_dim,))
    x = BatchNormalization()(inputs)
    x = Reshape((input_dim, 1))(x)
    for filters, drop in ((32, 0.1), (48, 0.1)):
        x = Conv1D(filters, 2, padding="same", activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(drop)(x)
    x = Concatenate()([GlobalAveragePooling1D()(x), GlobalMaxPooling1D()(x)])
    x = Dropout(0.15)(x)
    x = Dense(64, activation="relu", kernel_regularizer=l2(1e-4), kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu", kernel_regularizer=l2(1e-4), kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs)
