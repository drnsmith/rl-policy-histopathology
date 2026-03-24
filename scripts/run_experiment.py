"""
scripts/run_experiment.py

Main entry point. Runs all experimental conditions:
  1. no augmentation
  2. static augmentation (A3)
  3. RL-AUC   — RL-guided, reward = delta AUC
  4. RL-Acc   — RL-guided, reward = delta accuracy
  5. RL-F1    — RL-guided, reward = delta F1

Split protocol:
  Patient-level by default (config: split.patient_level = true).
  Set to false to reproduce the image-level baseline for the ablation table.

Usage:
  python scripts/run_experiment.py --config configs/config.yaml

  # Run only specific conditions:
  python scripts/run_experiment.py --conditions none static rl_auc

  # Image-level split (ablation only):
  python scripts/run_experiment.py --image-level-split
"""

import argparse
import os
import sys
import yaml
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.load_breakhis     import load_breakhis
from src.data.splits            import (add_patient_ids,
                                        train_test_split_by_patient,
                                        train_test_split_by_image,
                                        kfold_splits_by_patient,
                                        compute_class_weights)
from src.data.dataset           import build_dataset
from src.models.densenet201     import build_model, get_callbacks
from src.augmentation.policies  import get_policy, STATIC_BASELINE_ID, N_ACTIONS
from src.rl.agent               import QLearningAgent
from src.rl.env                 import run_rl_search
from src.evaluation.evaluate    import (compute_metrics, print_summary,
                                        save_results, plot_roc,
                                        plot_pr, plot_confusion)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


ALL_CONDITIONS = ["none", "static", "rl_auc", "rl_acc", "rl_f1"]

REWARD_METRIC_MAP = {
    "rl_auc": "auc",
    "rl_acc": "accuracy",
    "rl_f1":  "f1",
}


def run_fold(condition:       str,
             fold:            int,
             fold_train_df,
             fold_val_df,
             test_df,
             initial_weights: list,
             cfg:             dict) -> dict:
    """Train and evaluate one fold under one condition. Returns metric dict."""
    dc   = cfg["data"]
    tc   = cfg["training"]
    rlc  = cfg["rl"]
    evc  = cfg["evaluation"]
    mc   = cfg["model"]
    seed = dc["random_seed"]
    image_size = tuple(dc["image_size"])
    batch_size = tc["batch_size"]

    # Class weights from this fold's training partition
    class_weights = compute_class_weights(fold_train_df["label"].values)
    w_pos = class_weights[1]
    w_neg = class_weights[0]

    # Fresh model for every fold × condition
    model = build_model(
        image_size=image_size,
        dropout_rate=mc["dropout_rate"],
        extra_conv_filters=mc["extra_conv_filters"],
        extra_conv_kernel=mc["extra_conv_kernel"],
        learning_rate=tc["learning_rate"],
        class_weight_pos=w_pos,
        class_weight_neg=w_neg,
    )
    model.set_weights(initial_weights)

    val_ds  = build_dataset(fold_val_df,  image_size, batch_size,
                            shuffle=False, seed=seed)
    test_ds = build_dataset(test_df,       image_size, batch_size,
                            shuffle=False, seed=seed)

    ckpt_dir = evc["checkpoints_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir,
                             f"fold{fold+1}_{condition}.keras")

    callbacks = get_callbacks(
        tc["early_stopping_patience"],
        tc["lr_reduce_patience"],
        tc["lr_reduce_factor"],
        ckpt_path,
    )

    # ── augmentation policy selection ─────────────────────────────────────
    if condition == "none":
        aug_fn = None

    elif condition == "static":
        aug_fn = get_policy(STATIC_BASELINE_ID)

    elif condition in REWARD_METRIC_MAP:
        reward_metric = REWARD_METRIC_MAP[condition]
        agent = QLearningAgent(
            n_actions=N_ACTIONS,
            alpha=rlc["alpha"], gamma=rlc["gamma"],
            epsilon_start=rlc["epsilon_start"],
            epsilon_end=rlc["epsilon_end"],
            epsilon_decay=rlc["epsilon_decay"],
            seed=seed + fold,
        )
        def train_ds_fn(action_id):
            return build_dataset(fold_train_df, image_size, batch_size,
                                 augment_fn=get_policy(action_id),
                                 shuffle=True, seed=seed + fold)

        best_action = run_rl_search(
            model=model,
            train_ds_fn=train_ds_fn,
            val_ds=val_ds,
            agent=agent,
            reward_metric=reward_metric,
            n_steps=rlc["search_steps"],
            inner_epochs=rlc["inner_epochs"],
            auc_low=rlc["auc_low_threshold"],
            auc_high=rlc["auc_high_threshold"],
        )

        # Retrain from scratch with best policy
        print(f"\n  Retraining from scratch with A{best_action} …")
        model.set_weights(initial_weights)
        aug_fn = get_policy(best_action)

    else:
        raise ValueError(f"Unknown condition: {condition!r}")

    # ── full training ──────────────────────────────────────────────────────
    train_ds = build_dataset(fold_train_df, image_size, batch_size,
                             augment_fn=aug_fn, shuffle=True, seed=seed)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=tc["full_epochs"],
        callbacks=callbacks,
        verbose=1,
    )

    # ── test evaluation ────────────────────────────────────────────────────
    y_true, y_prob = [], []
    for images, labels in test_ds:
        y_prob.extend(model.predict(images, verbose=0).ravel())
        y_true.extend(labels.numpy().tolist())

    metrics = compute_metrics(np.array(y_true), np.array(y_prob))
    print(f"  Fold {fold+1} | {condition} | "
          f"AUC={metrics['roc_auc']:.4f}  "
          f"acc={metrics['accuracy']:.4f}  "
          f"recall={metrics['recall']:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--conditions", nargs="+", default=ALL_CONDITIONS,
                        choices=ALL_CONDITIONS,
                        help="Conditions to run (default: all)")
    parser.add_argument("--image-level-split", action="store_true",
                        help="Use image-level split (ablation only)")
    args = parser.parse_args()

    cfg  = load_config(args.config)
    seed = cfg["data"]["random_seed"]
    tf.random.set_seed(seed)
    np.random.seed(seed)

    for d in (cfg["evaluation"]["figures_dir"],
              cfg["evaluation"]["reports_dir"],
              cfg["evaluation"]["checkpoints_dir"]):
        os.makedirs(d, exist_ok=True)

    # ── load data ──────────────────────────────────────────────────────────
    print("\nLoading BreakHis …")
    df = load_breakhis(cfg["data"]["dataset_path"])
    df = add_patient_ids(df)

    use_patient_split = (cfg["split"]["patient_level"]
                         and not args.image_level_split)

    if use_patient_split:
        print("\nSplit protocol: PATIENT-LEVEL (correct)")
        train_df, test_df = train_test_split_by_patient(
            df, test_size=cfg["data"]["test_split"], random_seed=seed)
    else:
        print("\nSplit protocol: IMAGE-LEVEL (ablation only — inflated metrics)")
        train_df, test_df = train_test_split_by_image(
            df, test_size=cfg["data"]["test_split"], random_seed=seed)

    n_folds    = cfg["split"]["n_folds"]
    image_size = tuple(cfg["data"]["image_size"])
    mc         = cfg["model"]
    tc         = cfg["training"]

    # ── initial weights (shared across all folds and conditions) ──────────
    print("\nInitialising model …")
    init_model = build_model(image_size=image_size,
                             dropout_rate=mc["dropout_rate"],
                             extra_conv_filters=mc["extra_conv_filters"],
                             extra_conv_kernel=mc["extra_conv_kernel"],
                             learning_rate=tc["learning_rate"])
    initial_weights = init_model.get_weights()
    del init_model

    # ── run all conditions ─────────────────────────────────────────────────
    all_results = {}

    for condition in args.conditions:
        print(f"\n\n{'#'*65}")
        print(f"#  CONDITION: {condition.upper()}")
        print(f"{'#'*65}")

        fold_metrics = []
        fold_gen = kfold_splits_by_patient(train_df, n_folds=n_folds,
                                           random_seed=seed)
        for fold, fold_train_df, fold_val_df in fold_gen:
            metrics = run_fold(
                condition=condition,
                fold=fold,
                fold_train_df=fold_train_df,
                fold_val_df=fold_val_df,
                test_df=test_df,
                initial_weights=initial_weights,
                cfg=cfg,
            )
            fold_metrics.append(metrics)

        all_results[condition] = fold_metrics
        print_summary(fold_metrics, condition=condition)
        save_results(fold_metrics, condition=condition,
                     reports_dir=cfg["evaluation"]["reports_dir"])

    print("\n\nAll conditions complete.")
    print(f"Results in: {cfg['evaluation']['reports_dir']}")


if __name__ == "__main__":
    main()
