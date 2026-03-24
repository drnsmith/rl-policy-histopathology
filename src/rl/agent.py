"""
src/rl/agent.py

Tabular epsilon-greedy Q-learning agent for augmentation policy search.

State  : (last_action_id, auc_bucket)
           auc_bucket  0 = low (<0.95)
                       1 = medium (0.95–0.98)
                       2 = high (>0.98)
Action : int in {0 … N_ACTIONS-1}
Reward : delta_metric = val_metric_t − val_metric_{t−1}

         Three reward variants are compared in the paper:
           "auc"      → delta AUC      (recommended — threshold-independent)
           "accuracy" → delta accuracy (inflated by class imbalance)
           "f1"       → delta F1       (threshold-dependent at τ=0.5)

Q update:
  Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]
"""

import numpy as np


def _auc_bucket(auc: float,
                low:  float = 0.95,
                high: float = 0.98) -> int:
    if auc < low:   return 0
    if auc < high:  return 1
    return 2


def encode_state(action_id: int,
                 metric:    float,
                 low:       float = 0.95,
                 high:      float = 0.98) -> tuple:
    return (action_id, _auc_bucket(metric, low, high))


class QLearningAgent:
    """
    Epsilon-greedy tabular Q-learning agent.

    Parameters
    ----------
    n_actions       size of the discrete action space
    alpha           Q-learning rate
    gamma           discount factor
    epsilon_start   initial exploration probability
    epsilon_end     minimum exploration probability
    epsilon_decay   multiplicative decay per step
    seed            RNG seed for reproducibility
    """

    def __init__(self,
                 n_actions:     int   = 7,
                 alpha:         float = 0.1,
                 gamma:         float = 0.9,
                 epsilon_start: float = 0.3,
                 epsilon_end:   float = 0.05,
                 epsilon_decay: float = 0.92,
                 seed:          int   = 42):
        self.n_actions     = n_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.rng           = np.random.default_rng(seed)

        self.Q: dict[tuple, np.ndarray] = {}
        self._history: list[tuple[int, float]] = []   # (action, metric)

    # ── Q-table ──────────────────────────────────────────────────────────

    def _q(self, state: tuple) -> np.ndarray:
        if state not in self.Q:
            self.Q[state] = np.zeros(self.n_actions, dtype=np.float64)
        return self.Q[state]

    def select_action(self, state: tuple) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self._q(state)))

    def update(self, state: tuple, action: int,
               reward: float, next_state: tuple):
        q_sa     = self._q(state)[action]
        q_next   = np.max(self._q(next_state))
        td_error = reward + self.gamma * q_next - q_sa
        self._q(state)[action] += self.alpha * td_error

    def step_epsilon(self):
        self.epsilon = max(self.epsilon_end,
                           self.epsilon * self.epsilon_decay)

    def record(self, action: int, metric: float):
        self._history.append((action, metric))

    def best_action(self) -> int:
        """Action that achieved the highest metric during the search phase."""
        if not self._history:
            return 3   # fallback = static baseline (A3)
        return max(self._history, key=lambda t: t[1])[0]

    def reset(self, seed: int = None):
        self.Q        = {}
        self._history = []
        if seed is not None:
            self.rng = np.random.default_rng(seed)
