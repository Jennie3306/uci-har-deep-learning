# Human Activity Recognition with Deep Learning (UCI HAR)

Can deep learning models trained on **raw smartphone sensor signals** recognise daily activities better than classical models trained on **561 hand-crafted features**, when tested on **people they have never seen**?

This repository compares a 1D-CNN, a bidirectional LSTM and a CNN-LSTM in **PyTorch** against a scikit-learn baseline, using a leakage-free, subject-wise evaluation. Target use case: daily-activity monitoring for elderly people.

## Results

Subject-wise 5-fold cross-validation (GroupKFold). Primary metric: macro-F1, mean ± std over folds.

| Model | Input | Macro-F1 | Accuracy | Params |
|---|---|---|---|---|
| Best scikit-learn model: [name] | 561 features | [ ] | [ ] | – |
| 1D-CNN | raw 9 × 128 | [ ] | [ ] | [ ] |
| BiLSTM | raw 9 × 128 | [ ] | [ ] | [ ] |
| CNN-LSTM | raw 9 × 128 | [ ] | [ ] | [ ] |

Main finding: [one sentence, e.g. which model wins, by how much, and which classes are still confused].

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
├── requirements.txt
├── results/            # CSV / JSON outputs committed after each run
└── README.md
```

## Citation

Dataset: D. Anguita, A. Ghio, L. Oneto, X. Parra and J. L. Reyes-Ortiz. *A Public Domain Dataset for Human Activity Recognition Using Smartphones.* ESANN 2013. The dataset is distributed by its authors for non-commercial use.

## Author

Lê Như Nhã Uyên, Computer Science, Ho Chi Minh City University of Technology (HCMUT).
