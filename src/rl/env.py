"""
src/rl/env.py

RL environment: wraps one CV fold's training loop.

The environment exposes a step() interface:
  - receives action (augmentation policy id)
  - trains the model for inner_epochs (no early stopping in search phase)
  - evaluates on validation set using the configured reward metric
  - returns (next_state, reward, info)

Search phase and final evaluation are strictly separated:
  search phase  → inner_epochs per step, no early stopping
  retrain phase → full_epochs, early stopping, best policy from agent
"""

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from src.rl.agent import encode_state, QLearningAgent
from src.augmentation.policies import get_policy, STATIC_BASELINE_ID


def _compute_metric(y_true: list, y_prob: list, metric: str) -> float:
    """Compute the reward metric from validation predictions."""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    try:
        if metric == "auc":
            return float(roc_auc_score(y_true, y_prob))
        elif metric == "accuracy":
            return float(accuracy_score(y_true, (y_prob >= 0.5).astype(int)))
        elif metric == "f1":
            return float(f1_score(y_true, (y_prob >= 0.5).astype(int),
                                  zero_division=0))
        else:
            raise ValueError(f"Unknown reward metric: {metric!r}. "
                             "Choose 'auc', 'accuracy', or 'f1'.")
    except Exception:
        return 0.5


class AugmentationEnv:
    """
    MDP environment for one cross-validation fold.

    Parameters
    ----------
    model         compiled Keras model (weights reset before each fold)
    train_ds_fn   callable(action_id) -> tf.data.Dataset
    val_ds        tf.data.Dataset — fixed validation set (no augmentation)
    reward_metric "auc" | "accuracy" | "f1"
    inner_epochs  training epochs per RL step
    auc_low/high  AUC bucket thresholds for state encoding
    """

    def __init__(self, model, train_ds_fn, val_ds,
                 reward_metric: str   = "auc",
                 inner_epochs:  int   = 5,
                 auc_low:       float = 0.95,
                 auc_high:      float = 0.98):
        self.model         = model
        self.train_ds_fn   = train_ds_fn
        self.val_ds        = val_ds
        self.reward_metric = reward_metric
        self.inner_epochs  = inner_epochs
        self.auc_low       = auc_low
        self.auc_high      = auc_high

        self._prev_metric:    float = 0.0
        self._current_metric: float = 0.0
        self._current_action: int   = 0

    def reset(self) -> tuple:
        self._prev_metric    = 0.0
        self._current_metric = 0.0
        self._current_action = 0
        return encode_state(0, 0.0, self.auc_low, self.auc_high)

    def step(self, action_id: int) -> tuple:
        self.model.fit(
            self.train_ds_fn(action_id),
            epochs=self.inner_epochs,
            verbose=0,
        )

        y_true, y_prob = [], []
        for images, labels in self.val_ds:
            preds = self.model.predict(images, verbose=0).ravel()
            y_true.extend(labels.numpy().tolist())
            y_prob.extend(preds.tolist())

        new_metric = _compute_metric(y_true, y_prob, self.reward_metric)
        reward     = new_metric - self._prev_metric

        self._prev_metric    = self._current_metric
        self._current_metric = new_metric
        self._current_action = action_id

        next_state = encode_state(action_id, new_metric,
                                  self.auc_low, self.auc_high)
        return next_state, reward, {"metric": new_metric, "reward": reward}


def run_rl_search(model,
                  train_ds_fn,
                  val_ds,
                  agent:         QLearningAgent,
                  reward_metric: str = "auc",
                  n_steps:       int = 20,
                  inner_epochs:  int = 5,
                  auc_low:       float = 0.95,
                  auc_high:      float = 0.98) -> int:
    """
    Run the RL policy search phase for one fold.

    Returns the best action_id observed during search.
    The caller is responsible for retraining from scratch with that policy.
    """
    env   = AugmentationEnv(model, train_ds_fn, val_ds,
                            reward_metric=reward_metric,
                            inner_epochs=inner_epochs,
                            auc_low=auc_low, auc_high=auc_high)
    state = env.reset()

    print(f"  [RL] search: {n_steps} steps × {inner_epochs} epochs  "
          f"reward={reward_metric}")

    for step in range(n_steps):
        action                = agent.select_action(state)
        next_state, reward, info = env.step(action)
        agent.update(state, action, reward, next_state)
        agent.record(action, info["metric"])
        agent.step_epsilon()
        state = next_state

        print(f"  [RL] step {step+1:02d}  "
              f"action=A{action}  "
              f"{reward_metric}={info['metric']:.4f}  "
              f"reward={reward:+.4f}  "
              f"ε={agent.epsilon:.3f}")

    best = agent.best_action()
    best_metric = max(m for _, m in agent._history)
    print(f"  [RL] best policy: A{best}  "
          f"(peak {reward_metric}={best_metric:.4f})")
    return best
