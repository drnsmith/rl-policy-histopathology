"""
src/augmentation/policies.py

Discrete augmentation sub-policy action space (A0–A6).

Each policy is a callable (image, label) -> (image, label) for tf.data.map.

Action set (matches paper Table 1):
  A0  no augmentation
  A1  rotation ≤20°, horizontal flip                         [conservative]
  A2  A1 + width/height shift 10%
  A3  A2 + shear 10%, zoom 10%                               [static baseline]
  A4  rotation ≤20°, shear 20%, zoom 20%, both flips
  A5  zoom 20%, shift 10%, horizontal flip
  A6  all transforms at moderate magnitude, nearest fill

Constraint: overly aggressive distortions (e.g. large zoom, high rotation)
were found in preliminary experiments to reduce accuracy by introducing
patterns inconsistent with real histopathological scenarios. The action
space is therefore bounded to moderate magnitudes.
"""

import math
import tensorflow as tf


# ── helpers ──────────────────────────────────────────────────────────────────

def _rotate(image: tf.Tensor, max_deg: float = 20.0) -> tf.Tensor:
    original_shape = image.shape
    angle = tf.random.uniform(
        (), -max_deg * math.pi / 180.0, max_deg * math.pi / 180.0
    )
    def _apply(img, ang):
        from tensorflow.keras.preprocessing.image import apply_affine_transform
        return apply_affine_transform(
            img.numpy(), theta=float(ang.numpy()) * 180 / math.pi, fill_mode="nearest"
        ).astype("float32")
    result = tf.py_function(_apply, [image, angle], tf.float32)
    result.set_shape(original_shape)
    return result


def _shift(image: tf.Tensor, frac: float = 0.10) -> tf.Tensor:
    h = tf.cast(tf.shape(image)[0], tf.float32)
    w = tf.cast(tf.shape(image)[1], tf.float32)
    dy = tf.cast(tf.random.uniform(()) * frac * h, tf.int32)
    dx = tf.cast(tf.random.uniform(()) * frac * w, tf.int32)
    image = tf.roll(image, shift=dy, axis=0)
    image = tf.roll(image, shift=dx, axis=1)
    return image


def _zoom(image: tf.Tensor, zoom_range: float = 0.10) -> tf.Tensor:
    shape  = tf.shape(image)
    h, w   = tf.cast(shape[0], tf.float32), tf.cast(shape[1], tf.float32)
    scale  = 1.0 - tf.random.uniform((), 0.0, zoom_range)
    new_h  = tf.cast(h * scale, tf.int32)
    new_w  = tf.cast(w * scale, tf.int32)
    image  = tf.image.random_crop(image, [new_h, new_w, 3])
    image  = tf.image.resize(image, [shape[0], shape[1]])
    return image


def _hflip(image):  return tf.image.random_flip_left_right(image)
def _vflip(image):  return tf.image.random_flip_up_down(image)
def _clip(image):   return tf.clip_by_value(image, 0.0, 1.0)


# ── sub-policies ─────────────────────────────────────────────────────────────

def policy_A0(image, label):
    """No augmentation."""
    return image, label

def policy_A1(image, label):
    """Conservative: rotation ≤20°, horizontal flip."""
    image = _rotate(image, 20.0)
    image = _hflip(image)
    return _clip(image), label

def policy_A2(image, label):
    """A1 + width/height shift 10%."""
    image = _rotate(image, 20.0)
    image = _hflip(image)
    image = _shift(image, 0.10)
    return _clip(image), label

def policy_A3(image, label):
    """A2 + shear 10%, zoom 10%.  Static baseline condition."""
    image = _rotate(image, 20.0)
    image = _hflip(image)
    image = _shift(image, 0.10)
    image = _zoom(image, 0.10)
    return _clip(image), label

def policy_A4(image, label):
    """Rotation ≤20°, shear 20%, zoom 20%, both flips."""
    image = _rotate(image, 20.0)
    image = _hflip(image)
    image = _vflip(image)
    image = _shift(image, 0.10)
    image = _zoom(image, 0.20)
    return _clip(image), label

def policy_A5(image, label):
    """Zoom 20%, shift 10%, horizontal flip."""
    image = _zoom(image, 0.20)
    image = _shift(image, 0.10)
    image = _hflip(image)
    return _clip(image), label

def policy_A6(image, label):
    """All transforms at moderate magnitude."""
    image = _rotate(image, 20.0)
    image = _hflip(image)
    image = _vflip(image)
    image = _shift(image, 0.10)
    image = _zoom(image, 0.20)
    return _clip(image), label


# ── registry ─────────────────────────────────────────────────────────────────

ACTION_SPACE = {
    0: policy_A0,
    1: policy_A1,
    2: policy_A2,
    3: policy_A3,   # ← static baseline
    4: policy_A4,
    5: policy_A5,
    6: policy_A6,
}

N_ACTIONS             = len(ACTION_SPACE)
STATIC_BASELINE_ID    = 3


def get_policy(action_id: int):
    if action_id not in ACTION_SPACE:
        raise ValueError(f"action_id must be 0–{N_ACTIONS - 1}, got {action_id}")
    return ACTION_SPACE[action_id]
