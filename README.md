# Human Activity Recognition with Deep Learning (UCI HAR)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jennie3306/uci-har-deep-learning/blob/main/notebooks/UCI-HAR_DL.ipynb)

Can deep learning models trained on **raw smartphone sensor signals** recognise daily activities better than classical models trained on **561 hand-crafted features**, when tested on **people they have never seen**?

This repository compares a 1D-CNN, a bidirectional LSTM and a CNN-LSTM in **PyTorch** against a scikit-learn baseline, using a leakage-free, subject-wise evaluation. Target use case: daily-activity monitoring for elderly people.

## Results

Subject-wise 5-fold cross-validation (GroupKFold). Primary metric: macro-F1, mean ± std over folds.

| Model | Input | Macro-F1 | Accuracy | Params |
|---|---|---|---|---|
| **LinearSVC** (scikit-learn) | 561 features | **0.960 ± 0.017** | **0.960** | – |
| Logistic Regression (scikit-learn) | 561 features | 0.957 ± 0.016 | 0.958 | – |
| Random Forest (scikit-learn) | 561 features | 0.936 ± 0.027 | 0.939 | – |
| 1D-CNN | raw 9 × 128 | 0.930 ± 0.017 | 0.927 | 116,550 |
| BiLSTM | raw 9 × 128 | 0.919 ± 0.031 | 0.917 | 539,142 |
| **CNN-LSTM** | raw 9 × 128 | **0.935 ± 0.016** | **0.933** | **57,414** |

Main findings:

- **Hand-crafted features still win on this dataset.** LinearSVC on the 561 features beats the best deep model (CNN-LSTM) by about 0.025 macro-F1, and it is better on **all 5 folds** (same test subjects for every model), so the gap is consistent rather than noise.
- **Deep models get close without any feature engineering.** All three reach 0.92–0.94 macro-F1 from raw signals. CNN-LSTM has the best mean with the fewest parameters (57k), but the differences between the three deep models are within one standard deviation, so none is a clear winner.
- **The hardest classes are SITTING ↔ STANDING** (per-class F1 0.81–0.86 for the deep models vs ≥ 0.94 for every other activity): both are static postures that differ mainly in phone orientation.

Full run with outputs: [`notebooks/UCI-HAR_DL.ipynb`](notebooks/UCI-HAR_DL.ipynb) (Colab, T4 GPU). Per-fold scores and confusion matrices: [`results/`](results/).

## Dataset

[UCI HAR](https://archive.ics.uci.edu/dataset/240): 30 volunteers aged 19–48 wearing a smartphone (Samsung Galaxy S II) on the waist. Accelerometer and gyroscope sampled at 50 Hz, cut into 2.56 s windows (128 samples) with 50 % overlap, giving 10,299 windows. Six activities: walking, walking upstairs, walking downstairs, sitting, standing, lying.

This project uses the 9 raw inertial channels (`total_acc`, `body_acc`, `body_gyro` × x, y, z). The script downloads the dataset automatically into `data/`, which is git-ignored.

## Method

| Step | What | Why |
|---|---|---|
| Split | Merge official train/test, then GroupKFold(5) by subject | Overlapping windows and per-person movement style make random splits over-optimistic |
| Early stopping | Inner subject-wise validation split (20 % of training subjects), patience 6 | Choose the epoch without touching the test fold |
| Normalisation | Per-channel z-score fitted on training subjects only | Avoid leaking test statistics; channels have different units |
| Metric | Macro-F1 + per-class F1 + pooled confusion matrix | Every activity matters equally for monitoring |
| Training | AdamW (lr 1e-3, weight decay 1e-4), batch 64, cross-entropy, ≤ 30 epochs | Standard, reproducible setup (fixed seeds) |

Models:

- **1D-CNN**: 4 Conv1d–BatchNorm–ReLU blocks, max pooling, global average pooling, dropout, linear head.
- **BiLSTM**: 2-layer bidirectional LSTM (hidden 128), mean-pooled over time.
- **CNN-LSTM**: 2 conv blocks extract local patterns, an LSTM (hidden 64) models their order.

## How to run

```bash
pip install -r requirements.txt
python har_pytorch.py --synthetic --epochs 2 --n-splits 3   # smoke test, no download
python har_pytorch.py --model cnn                            # one model
python har_pytorch.py --model all                            # cnn, lstm, cnn_lstm
```

On Google Colab: *Runtime → Change runtime type → T4 GPU*, upload `har_pytorch.py`, then run the same commands prefixed with `!`.

If the automatic download fails, download the dataset from the UCI page and extract it so that `data/UCI HAR Dataset/train/Inertial Signals/` exists.

Outputs in `results/`:

- `<model>_folds.csv`: per-fold test subjects, best epoch, accuracy, macro-F1
- `<model>_summary.json`: mean ± std, per-class F1, confusion matrix, full config

## Limitations

- Volunteers were 19–48 years old and recorded in lab conditions with a fixed waist position. Results may not transfer to elderly users or free-living conditions without new data.
- Windows are classified independently. Smoothing predictions over consecutive windows could help in a real system.

## Repository structure

```
├── har_pytorch.py      # data loading, models, training, subject-wise CV
├── notebooks/          # Colab notebook of the full run, with outputs
├── requirements.txt
├── results/            # CSV / JSON outputs committed after each run
└── README.md
```

## Citation

Dataset: D. Anguita, A. Ghio, L. Oneto, X. Parra and J. L. Reyes-Ortiz. *A Public Domain Dataset for Human Activity Recognition Using Smartphones.* ESANN 2013. The dataset is distributed by its authors for non-commercial use.

## Author

Lê Như Nhã Uyên, Computer Science, Ho Chi Minh City University of Technology (HCMUT).