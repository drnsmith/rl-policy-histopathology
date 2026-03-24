"""
src/data/dataset.py

Build tf.data.Dataset pipelines from DataFrames.
Augmentation is passed as a callable so the RL agent can swap policies
without rebuilding the pipeline.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Callable, Optional


def _load_image(path: str,
                label: int,
                image_size: tuple) -> tuple:
    raw  = tf.io.read_file(path)
    img  = tf.image.decode_png(raw, channels=3)
    img  = tf.image.resize(img, image_size)
    img  = tf.cast(img, tf.float32) / 255.0
    return img, label


def build_dataset(df:         pd.DataFrame,
                  image_size: tuple,
                  batch_size: int,
                  augment_fn: Optional[Callable] = None,
                  shuffle:    bool = True,
                  seed:       int  = 42) -> tf.data.Dataset:
    """
    Build a batched, prefetched tf.data.Dataset.

    Parameters
    ----------
    df          DataFrame with 'path' and 'label' columns
    image_size  (H, W)
    batch_size  mini-batch size
    augment_fn  optional (image, label) -> (image, label) applied per sample
    shuffle     shuffle before batching (True for train, False for val/test)
    seed        shuffle seed
    """
    paths  = df["path"].values
    labels = df["label"].values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p, l: _load_image(p, l, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if augment_fn is not None:
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
