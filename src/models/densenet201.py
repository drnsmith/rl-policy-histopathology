"""
src/models/densenet201.py

DenseNet201 classifier for binary breast-cancer classification.

Architecture rationale (paper Section 3):
  DenseNet201 was selected as the single backbone. Its dense connectivity —
  each layer receiving feature maps from all preceding layers — promotes
  feature reuse across spatial scales, well-suited to histopathological
  texture where fine-grained patterns at multiple resolutions carry
  diagnostic information. Dense connectivity also reduces parameter
  redundancy, lowering overfitting risk on a moderately-sized dataset.
  Prior work on BreakHis consistently places DenseNet201 among the
  strongest single-model performers (Voon et al., 2022).

Head architecture:
  DenseNet201 base (ImageNet weights, fully fine-tuned)
  -> Conv2D(256, 3×3, ReLU)
  -> BatchNormalization
  -> GlobalAveragePooling2D
  -> Dropout(0.5)
  -> Dense(1, sigmoid)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import DenseNet201


def build_model(image_size:          tuple = (224, 224),
                dropout_rate:        float = 0.5,
                extra_conv_filters:  int   = 256,
                extra_conv_kernel:   int   = 3,
                learning_rate:       float = 1e-4,
                class_weight_pos:    float = 1.0,
                class_weight_neg:    float = 1.0) -> Model:
    """
    Build and compile the DenseNet201 classifier.

    class_weight_pos : weight for positive (malignant) class in loss
    class_weight_neg : weight for negative (benign)    class in loss
    Weights are recomputed per fold via splits.compute_class_weights().
    """
    inputs = layers.Input(shape=(*image_size, 3))

    base = DenseNet201(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
    )
    base.trainable = True  # full fine-tuning

    x = base.output
    x = layers.Conv2D(extra_conv_filters, extra_conv_kernel,
                      padding="same", activation="relu",
                      name="extra_conv")(x)
    x = layers.BatchNormalization(name="extra_bn")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs,
                  name="BreakHis_DenseNet201")

    def weighted_bce(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce    = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        weight = y_true * class_weight_pos + (1.0 - y_true) * class_weight_neg
        return tf.reduce_mean(weight * bce)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=weighted_bce,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def get_callbacks(early_stopping_patience: int   = 10,
                  lr_reduce_patience:      int   = 5,
                  lr_reduce_factor:        float = 0.5,
                  checkpoint_path:         str   = None) -> list:
    cbs = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=early_stopping_patience,
            mode="max", restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", factor=lr_reduce_factor,
            patience=lr_reduce_patience, mode="max",
            min_lr=1e-7, verbose=1,
        ),
    ]
    if checkpoint_path:
        cbs.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path, monitor="val_auc",
                save_best_only=True, mode="max", verbose=0,
            )
        )
    return cbs
