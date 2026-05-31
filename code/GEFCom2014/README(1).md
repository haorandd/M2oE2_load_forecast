# GRU-M²oE² on GEFCom2014-L

This repository contains the GEFCom2014-L experiments for **GRU-M²oE²**, an external-data-enhanced probabilistic load forecasting model. The current version follows a **day-ahead 00:00 forecasting protocol** and reports raw-scale probabilistic metrics on the 2011 test set.

Two weather-input settings are provided:

- **Weather station: 25 → 1**: average temperature across all 25 weather stations.
- **Weather station: 25 → 3**: average temperatures of 3 station clusters derived from the 25 weather stations.

The current checkpoint and evaluation files have been renamed to short, paper-friendly names.

---

## Repository Structure

```text
.
├── 1weather/
│   ├── Model_1cluster_GEF.py
│   ├── Main_1cluster_GEF.py
│   ├── M2OE2_1cluster.pt
│   └── M2OE2_1cluster_eval_results.json
├── 3weather/
│   ├── Model_3cluster_GEF.py
│   ├── Main_3clusterGEF.py
│   ├── M2OE2_3cluster.pt
│   └── M2OE2_3cluster_eval_results.json
├── metrics_check.json
└── README.md
```

`metrics_check.json` verifies that the saved evaluation files reproduce the reported table values after rounding to two decimals.

---

## Dataset

The experiment uses the **GEFCom2014-L** load forecasting dataset.

The dataset was originally released with:

> Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli, and Rob J. Hyndman,  
> "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond,"  
> *International Journal of Forecasting*, 32(3), 896--913, 2016.  
> DOI: `10.1016/j.ijforecast.2016.02.001`

Please cite the original GEFCom2014 paper when using this dataset.

The scripts expect the GEFCom2014-L zip file at:

```python
zip_path = "/content/drive/MyDrive/M2oE2_For_Zhe/data/GEFCom2014-L_V2.zip"
```

If your local path is different, update `zip_path` in:

```text
1weather/Main_1cluster_GEF.py
3weather/Main_3clusterGEF.py
```

---

## Experimental Protocol

The scripts use the following protocol:

1. Read the GEFCom2014-L load and temperature data from the zip file.
2. Use **2006--2009** as the training period, **2010** as the validation period, and **2011** as the test period.
3. Build weekly sequence-to-sequence samples.
4. Use the previous 168 hours as the encoder input.
5. Use a 168-hour decoder week, but evaluate only day-ahead origins at:

```python
forecast_indices = [0, 24, 48, 72, 96, 120, 144]
```

6. Forecast the next `K = 24` hours at each daily 00:00 origin.
7. Report metrics after inverse transformation to the original load scale.

---

## Model Variants

### Weather station: 25 → 1

This setting uses one average-temperature feature computed from all 25 weather stations.

Main files:

```bash
cd 1weather
python Main_1cluster_GEF.py
```

Saved checkpoint and result file:

```text
M2OE2_1cluster.pt
M2OE2_1cluster_eval_results.json
```

Key external inputs:

```text
temp, future 24-hour temperature path, workday, holiday, month labels
```

The saved evaluation file reports `external_raw_dim = 31`.

### Weather station: 25 → 3

This setting clusters the 25 weather stations into 3 groups using training-year temperature correlation distance and average-linkage agglomerative clustering.

Main files:

```bash
cd 3weather
python Main_3clusterGEF.py
```

Saved checkpoint and result file:

```text
M2OE2_3cluster.pt
M2OE2_3cluster_eval_results.json
```

The saved evaluation file reports `external_raw_dim = 81` and `k_temp = 3`.

The station clusters are:

| Cluster | Weather stations |
|---:|---|
| 0 | w1, w2, w3, w4, w5, w7, w8, w10, w11, w12, w13, w15, w16, w17, w19, w20, w22, w23, w24, w25 |
| 1 | w6, w14, w18, w21 |
| 2 | w9 |

---

## Experimental Settings for GEFCom2014

| Setting | Description |
|---|---|
| Dataset | GEFCom2014 |
| Training period | 2006--2009 |
| Validation period | 2010 |
| Test period | 2011 |
| Weather feature (Weather station: 25 → 1) | Average temperature across 25 weather stations |
| Weather feature (Weather station: 25 → 3) | Average temperatures of 3 station clusters derived from the 25 weather stations |
| Calendar features | Workday, holiday, and month labels |
| Forecast horizon | `K = 24` |
| Learning rate | `10^-3` |
| KL weight | `0.0001` |
| Batch size | `13` |
| Maximum training epochs | `1500` |
| Early stopping epoch (validation metric) | `564` for Weather station: 25 → 1; `537` for Weather station: 25 → 3 |
| Dropout rate | `0.1` |
| Training scale | Min--max normalization |
| Evaluation scale | Original scale after inverse transformation |
| Main metrics | QS, WS (50%), and WS (90%) |

---

## Probabilistic Load Forecasting Performance on GEFCom2014

The following values are test-set raw-scale metrics on 2011. Lower values are better.

| Model | QS | WS (50%) | WS (90%) |
|---|---:|---:|---:|
| iQRNN [1] | 2.71 | 23.94 | 47.11 |
| Q-ResNet [2] | 2.69 | 23.64 | 46.50 |
| ResNetPlus [3] | 2.52 | 22.41 | 42.63 |
| Basic QRNN [4] | 2.45 | 21.75 | 41.80 |
| AHLC-QRNN [4] | 2.42 | 21.43 | 40.84 |
| **GRU-M²oE² (Weather station: 25 → 1)** | **2.31** | **20.41** | **39.72** |
| **GRU-M²oE² (Weather station: 25 → 3)** | **2.27** | **20.06** | **39.35** |

The saved JSON values before rounding are:

| Variant | `pinball_allq_daily` | `winkler_50_daily` | `winkler_90_daily` |
|---|---:|---:|---:|
| Weather station: 25 → 1 | 2.309159 | 20.410744 | 39.724805 |
| Weather station: 25 → 3 | 2.268537 | 20.059014 | 39.349204 |

---

## Metrics

The main probabilistic metrics are computed after inverse normalization on the raw load scale.

- **QS / Pinball**: averaged pinball loss over quantiles from 0.01 to 0.99.
- **WS (50%)**: Winkler score for the central 50% prediction interval.
- **WS (90%)**: Winkler score for the central 90% prediction interval.

The evaluation scripts also report MSE, RMSE, NLL, CRPS, quantile loss, and peak-value error.

The main values used in the paper table are stored in:

```text
test.paper3_raw_day_ahead.pinball_allq_daily
test.paper3_raw_day_ahead.winkler_50_daily
test.paper3_raw_day_ahead.winkler_90_daily
```

---

## How to Run

Install the required packages:

```bash
pip install numpy pandas torch scikit-learn matplotlib holidays
```

Run the 25 → 1 weather-station setting:

```bash
cd 1weather
python Main_1cluster_GEF.py
```

Run the 25 → 3 weather-station setting:

```bash
cd 3weather
python Main_3clusterGEF.py
```

If the corresponding `.pt` checkpoint exists, the script loads the saved model and evaluates it. If the checkpoint is missing, the script trains the model and saves a new checkpoint.

---

## Reproducibility Notes

The scripts fix random seeds for Python, NumPy, and PyTorch:

```python
seed = 42
```

Evaluation is deterministic by default:

```python
deterministic_eval = True
z = mu
```

The training setup uses:

```text
Optimizer: AdamW
Learning-rate scheduler: warmup cosine
Warmup ratio: 0.04
Minimum LR ratio: 0.1
Gradient clipping norm: 1.0
Weight decay: 1e-4
KL annealing epochs: 50
Top-k gating: 4 / 4 experts
```

Because `.pt` checkpoint files are binary files, Git LFS is recommended if this repository is shared publicly.

---

## Citation

If you use this repository or the M²oE² model, please cite the M²oE² paper:

```bibtex
@article{li2025m2oe2,
  title={External Data-Enhanced Meta-Representation for Adaptive Probabilistic Load Forecasting},
  author={Li, Haoran and Guo, Muhao and Ilic, Marija and Weng, Yang and Ruan, Guangchun},
  journal={arXiv preprint arXiv:2506.23201},
  year={2025}
}
```

If you use the GEFCom2014-L dataset, please cite:

```bibtex
@article{hong2016gefcom2014,
  title={Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond},
  author={Hong, Tao and Pinson, Pierre and Fan, Shu and Zareipour, Hamidreza and Troccoli, Alberto and Hyndman, Rob J.},
  journal={International Journal of Forecasting},
  volume={32},
  number={3},
  pages={896--913},
  year={2016},
  doi={10.1016/j.ijforecast.2016.02.001}
}
```
