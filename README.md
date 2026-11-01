# Wireless Intrusion Detection System (WIDS) using AWID3

A prototype WIDS that uses scikit-learn to detect deauthentication and disassociation floods in Wi-Fi traffic and raise alerts.

## Dataset
Uses the public AWID3 dataset (Chatzoglou, Kambourakis, Kolias, IEEE Access, 2021). The data is **not included** in this repository; request access from the AWID project page (icsdweb.aegean.gr/awid). I used 8 Deauth and 9 Disass CSV slices. No attacks were generated for this project.

## How it works
AWID3 per-frame CSV -> 1-second windows per device -> 13 features -> classifier -> alert and log.

## Run order
Put the AWID3 CSVs in data/Deauth and data/Disass, then run:

    pip install -r requirements.txt
    python build_dataset.py
    python train.py label_bin
    python train.py label
    python train_ablation.py label_bin
    python cv.py
    python replay.py data/Deauth/Deauth_14.csv data/Deauth/Deauth_31.csv data/Disass/Disas_30.csv

What each step does: build_dataset.py builds dataset.csv; train.py label_bin compares models for normal vs attack; train.py label classifies the attack type; train_ablation.py repeats the test without the count features; cv.py runs 5-fold cross-validation by file; replay.py replays recorded traffic one second at a time, prints alerts and writes alerts.log. It is a replay, not live capture.

## Results (normal vs attack, 925 test windows, 128 attacks)
| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| Random Forest | 99.89% | 1.000 | 0.992 |
| Logistic Regression | 99.46% | 1.000 | 0.961 |
| Simple rule (>=10/s) | 99.35% | 1.000 | 0.953 |
| Decision Tree | 99.03% | 0.992 | 0.938 |
| Naive Bayes | 95.89% | 0.771 | 1.000 |
| Isolation Forest | 92.11% | 0.802 | 0.570 |

5-fold cross-validation by file: 261 of 263 attack windows caught, 0 false alarms in 3,060 normal windows. Full outputs and figures are in results/.

## Limitations
One testbed, small test set, replay instead of live capture, spoofable source addresses, no prevention (WIDS only). Rogue_AP was dropped (too few attack frames).

## AI usage
I used Claude (Anthropic) as a coding assistant to help plan the pipeline, write and debug the Python scripts, and interpret results. I ran all the code and can explain how it works.
