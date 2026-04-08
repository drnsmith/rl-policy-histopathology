## **Adaptive augmentation policy search via reinforcement learning for breast cancer histopathology classification.**



---

## What this is

This repository implements the experiments for my paper:

> *Adaptive Augmentation Policy Search via Reinforcement Learning for Robust Medical Image Classification* (preprint forthcoming)

The central argument is not that RL-guided augmentation is novel in itself — AutoAugment established that in 2019. The argument is that the **choice of reward metric used to guide the RL agent is a measurement validity problem**, not merely a hyperparameter choice. An agent rewarded by accuracy on an imbalanced dataset learns a different — and worse — policy than one rewarded by AUC. This is demonstrated empirically by running the same agent under three reward variants (AUC, accuracy, F1) and reporting the consequences for recall, specificity, and false negative rate.

A secondary contribution is methodological: unlike the majority of published BreakHis studies, **folds are stratified by patient ID**, not by image. The BreakHis dataset contains up to 32 images per patient across four magnifications. Image-level splitting allows the same patient's tissue to appear in both training and validation, inflating all reported metrics. This repository corrects that, and provides an image-level baseline for comparison.

---

## Results status

Experiments are in progress. Tables in the paper currently contain placeholder values pending experimental completion. This repository will be updated with final results upon completion.

---

## Dataset

[BreakHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/) — 7,909 breast histopathology images from 82 patients, 4 magnifications (40×, 100×, 200×, 400×), binary classification (benign / malignant).

Download the dataset and extract it so the structure is:

```
BreaKHis_v1/
  histology_slides/breast/
    benign/SOB/<subtype>/<patient>/<mag>X/*.png
    malignant/SOB/<subtype>/<patient>/<mag>X/*.png
```

---

## Setup

```bash
git clone https://github.com/drnsmith/rl-policy-histopathology.git
cd rl-policy-histopathology
pip install -r requirements.txt
```

---

## Configuration

Edit **one line** in `configs/config.yaml`:

```yaml
data:
  dataset_path: "/path/to/your/BreaKHis_v1"
```

| Environment | Path |
|---|---|
| External drive (macOS) | `/Volumes/MyDrive/KaggleData/BreaKHis_v1` |
| External drive (Linux) | `/media/natalya/MyDrive/KaggleData/BreaKHis_v1` |
| Google Colab | `/content/drive/MyDrive/KaggleData/BreaKHis_v1` |
| Cloud GPU | `/home/user/data/BreaKHis_v1` |

---

## Running experiments

**All conditions** (no augmentation, static, RL-AUC, RL-Acc, RL-F1):

```bash
python scripts/run_experiment.py
```

**Specific conditions only:**

```bash
python scripts/run_experiment.py --conditions rl_auc rl_acc rl_f1
```

**Image-level split** (ablation comparison — produces inflated metrics):

```bash
python scripts/run_experiment.py --image-level-split
```

Results are saved to `results/reports/` as CSVs, one per condition.

---

## Experimental conditions

| Condition | Description |
|---|---|
| `none` | No augmentation |
| `static` | Fixed sub-policy A3 throughout training |
| `rl_auc` | RL-guided, reward = ΔAUC *(recommended)* |
| `rl_acc` | RL-guided, reward = Δaccuracy |
| `rl_f1` | RL-guided, reward = ΔF1 |

---

## Why patient-level splits matter

The BreakHis dataset ships with an official 5-fold patient-level split (Spanhol et al., 2016). Despite its availability, the majority of published studies bypass it in favour of `train_test_split` on image arrays — allowing the same patient's tissue at different magnifications to appear in both training and test. This inflates every reported metric.

`src/data/splits.py` extracts patient IDs from filenames and uses `StratifiedGroupKFold` to ensure complete patient separation. The function `train_test_split_by_image()` is preserved for the ablation comparison only.

The gap between image-level and patient-level metrics is reported in the paper's Table 2 and is itself evidence for the measurement validity argument.

---

## Repository structure

```
rl-policy-histopathology/
├── src/
│   ├── data/
│   │   ├── load_breakhis.py     walks BreakHis directory tree
│   │   ├── splits.py            patient-level stratified splits
│   │   └── dataset.py           tf.data pipeline builder
│   ├── augmentation/
│   │   └── policies.py          7 discrete sub-policies (A0–A6)
│   ├── models/
│   │   └── densenet201.py       DenseNet201 + weighted BCE
│   ├── rl/
│   │   ├── agent.py             tabular Q-learning agent
│   │   └── env.py               MDP environment + search loop
│   └── evaluation/
│       └── evaluate.py          metrics, plots, CSV export
├── configs/
│   └── config.yaml              all hyperparameters
├── scripts/
│   └── run_experiment.py        main entry point
├── results/                     outputs (gitignored except .gitkeep)
└── data/raw/                    put BreaKHis_v1 here (gitignored)
```

---

## Citation

If you use this code, please cite the accompanying paper (preprint link forthcoming) and the BreakHis dataset:

```
Spanhol, F.A., Oliveira, L.S., Petitjean, C., Heutte, L. (2016).
A dataset for breast cancer histopathological image classification.
IEEE Transactions on Biomedical Engineering, 63(7), 1455–1462.
```

---

## License

MIT — see [LICENSE](LICENSE).

---

## Note on prior work

This repository is a methodologically corrected reconstruction of earlier student work (available at [drnsmith/Histopathology-AI-BreastCancer](https://github.com/drnsmith/Histopathology-AI-BreastCancer)), which used image-level splitting consistent with most published BreakHis literature at the time. The reconstruction corrects the split protocol and adds the RL reward metric comparison.
