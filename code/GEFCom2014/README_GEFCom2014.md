# M2OE2 on GEFCom2014-L

This document provides a benchmark-style description of running **M2OE2** on the **GEFCom2014-L** load forecasting dataset.

The corresponding script is:

```bash
python M2OE2_for_GEFCom2014_papersetting.py
```

The goal of this experiment is to evaluate M2OE2 under the commonly used GEFCom2014-L data split and probabilistic forecasting metrics. The model uses historical load, average temperature information from 25 weather stations, workday labels, and season labels to generate probabilistic load forecasts.

---

## Dataset

The experiment uses the **GEFCom2014-L** load forecasting dataset.

The GEFCom2014 data were originally released as the appendix / supplementary material of the GEFCom2014 paper:

> Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli, and Rob J. Hyndman,  
> "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond,"  
> *International Journal of Forecasting*, 32(3), 896--913, 2016.  
> DOI: 10.1016/j.ijforecast.2016.02.001

Please cite the original GEFCom2014 paper when using this dataset.

The expected local data structure is:

```text
data/
└── GEFCom2014 Data/
    └── GEFCom2014-L_V2/
        └── Load/
            ├── Task 1/
            │   └── L1-train.csv
            ├── Task 2/
            │   └── L2-train.csv
            ├── ...
            └── Task 15/
                └── L15-train.csv
```

If your local path is different, modify the `root` argument in:

```python
get_data_GEFCom2014_avgtemp_latest(...)
```

---

## Adopted Experimental Settings

The following settings are used for the GEFCom2014-L experiment:

| Setting | Adopted value |
|---|---|
| Dataset | GEFCom2014-L |
| Forecast horizon | `K = 24` |
| Learning rate | `1e-3` |
| KL weight | `0.001` |
| Training period | 2006--2009 |
| Validation period | 2010 |
| Test period | 2011 |
| Weather feature | Average temperature from 25 weather stations |
| Calendar features | Workday and season |
| Scaling | Global MinMax scaling |
| Model selection | Validation CRPS early stopping |
| Early stopping patience | 100 epochs |
| Evaluation scale | Raw load scale after inverse normalization |
| Main metrics | QS / Pinball, WS(50%), WS(90%) |

---

## Evaluation Protocol

The reported main results use **rolling first-step evaluation**.

For each rolling window, the model outputs a 24-hour probabilistic forecast. However, the reported rolling first-step metrics only evaluate the first forecast horizon:

```python
mu_first = mu_preds[:, :, 0]
tgt_first = tgt[:, :, 0]
```

This follows the original rolling short-term forecasting behavior of the M2OE2 sequence-to-sequence setup.

**Important note:**  
The reported comparison demonstrates rolling short-term probabilistic forecasting performance under the GEFCom2014-L data split and raw-scale metric setting. It is not a strict full 24-hour direct day-ahead evaluation.

---

## Metrics

The script reports both normalized-scale and raw-scale metrics. The main comparison metrics are computed after inverse normalization on the raw load scale.

The three benchmark-style probabilistic metrics are:

- **QS / Pinball**: averaged pinball loss over quantiles from 0.01 to 0.99
- **WS(50%)**: Winkler score for the central 50% prediction interval
- **WS(90%)**: Winkler score for the central 90% prediction interval

The script also reports:

- MSE
- RMSE
- CRPS
- QuantileLoss over `[0.1, 0.5, 0.9]`
- 90% Winkler score
- Peak Value Error

---

## How to Run

Install the required packages:

```bash
pip install numpy pandas torch scikit-learn matplotlib
```

Then run:

```bash
python M2OE2_for_GEFCom2014_papersetting.py
```

The script will:

1. Load and clean the GEFCom2014-L data.
2. Build weekly sequence-to-sequence samples.
3. Split the data into 2006--2009 training, 2010 validation, and 2011 test sets.
4. Train M2OE2 with validation-CRPS early stopping.
5. Save the best model checkpoint.
6. Evaluate validation and test performance on both normalized and raw scales.
7. Print rolling first-step benchmark-style probabilistic metrics.

---

## Checkpoint

The default checkpoint name is:

```text
GEFCom2014_M2OE2_v1_24hours_protocol_avgtemp_wds_minmax_kl0001_ep1500_valCRPS_es100_det_best_model.pt
```

If this file already exists, the script loads it directly and skips retraining. To retrain from scratch, delete the checkpoint file or change the `model_name` in the script.

---

## Current Result

Using validation-CRPS early stopping and deterministic VAE evaluation, the best model was selected at epoch 435. On the 2011 test set, the rolling first-step raw-scale probabilistic metrics are:

| Model | QS / Pinball | WS (50%) | WS (90%) |
|---|---:|---:|---:|
| iQRNN | 2.71 | 23.94 | 47.11 |
| Q-ResNet | 2.69 | 23.64 | 46.50 |
| ResNetPlus | 2.52 | 22.41 | 42.63 |
| Basic QRNN | 2.45 | 21.75 | 41.80 |
| AHLC-QRNN | 2.42 | 21.43 | 40.84 |
| **M2OE2** | **1.99** | **17.84** | **30.37** |

Again, the M2OE2 result is based on rolling first-step evaluation. It should not be interpreted as a strict full 24-hour direct day-ahead comparison.

---

## Reproducibility Notes

The script fixes random seeds for Python, NumPy, and PyTorch:

```python
seed = 42
```

The VAE evaluation is deterministic by default:

```python
z = mu
```

This avoids random latent sampling during validation and test evaluation.

---

## Citation

If you use this repository or the M2OE2 model, please cite the M2OE2 paper:

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
