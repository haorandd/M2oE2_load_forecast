# M2oE2-GL: A Family of Probabilistic Load Forecasters That Scales to Massive Customers

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## Overview

This repository contains the official implementation of **M2oE2-GL**, a scalable family of probabilistic load forecasters for massive customer populations. It extends the **M2oE2** (Many-to-One Elastic Ensemble) framework with a global-to-local adaptation strategy to address the critical challenge of forecasting localized power grid loads (e.g., specific distribution transformers) under conditions of data scarcity, heterogeneity, and high volatility.

Our approach leverages **Transfer Learning** combined with **Low-Rank Adaptation (LoRA)** to efficiently adapt large-scale pre-trained models (trained on aggregated data) to specific target domains (individual transformers). The framework supports multiple deep learning architectures—including **Variational Autoencoders (VAE)**, **CNN-GRU**, **LSTM**, and **RNNs**—and integrates a novel **Peak-Weighted Loss** mechanism to enhance grid resilience prediction during extreme demand events.

## Key Features

* **Probabilistic Forecasting:** Quantifies aleatoric uncertainty using Gaussian predictive distributions, rigorously evaluated via CRPS, NLL, and Winkler Scores.
* **Parameter-Efficient Fine-Tuning (PEFT):** Implements **LoRA** to adapt base models to target domains with minimal computational overhead (<1% trainable parameters), preventing catastrophic forgetting.
* **Peak-Aware Optimization:** Introduces a quantile-based weighted loss function to prioritize forecast accuracy during critical peak load periods, essential for grid stability analysis.
* **Multi-Architecture Support:** Provides standardized implementations for:
    * **VAE (M2oE2):** Generative probabilistic modeling.
    * **CNN-GRU:** Hybrid feature extraction and temporal modeling.
    * **LSTM / RNN:** Robust recurrent baselines.
* **Flexible Covariate Integration:** Seamlessly handles external features such as weather data and temporal embeddings.



## Latest Model Variant: Residential-Only Base Model (No LoRA)

The latest model used in our recent experiments is a **base-only M2oE2 variant for the residential setting**, and it does **not** use LoRA.

This design choice is intentional. In this setup, we focus on a more homogeneous residential load pattern rather than cross-domain adaptation between heterogeneous customer groups. As a result, we use a **single base model pipeline** instead of the earlier **global-to-local LoRA adaptation** workflow.

In the current implementation, the corresponding training script is `M2OE2_Base_Only.py`, which is explicitly marked as:

- **NO LoRA**
- base-model training only
- base-only evaluation/export workflow

The exported results and plotting pipeline for this latest variant are also base-only. For example, the companion plotting script `M2oE2_base_draw.py` is written for **base-only** forecasts and uses base-model keys only.

### What is different from the earlier LoRA version?

Earlier versions of this repository supported a **global pretraining + LoRA fine-tuning** workflow for adapting a pretrained model to a target transformer or target domain. In contrast, the latest residential-focused model:

- uses a **base-only** training/evaluation workflow,
- does **not** attach LoRA layers,
- does **not** perform LoRA fine-tuning,
- and should be interpreted as a residential/base benchmark rather than a PEFT-adapted model.

### Practical note

If you are reproducing the latest residential results, please use the **base-only scripts and outputs**, not the LoRA-based workflow.

## External Data Naming Note

### Legacy naming vs. actual Oncor covariates

The variable names used in the older Oncor scripts follow a legacy convention and do **not** exactly match the actual semantics of the external channels currently loaded by the preprocessing pipeline.

In the current Oncor loader, the four base channels are returned as:

- `load` → load / kWh target
- `temp` → `SURDPOINTTEMPFAHRENHEIT`
- `workday` → `RELATIVEHUMIDITY`
- `season` → `HEATINDEXFAHRENHEIT`

This means that, in the current codebase, the names `temp`, `workday`, and `season` should be understood as **compatibility placeholders**, rather than literal descriptions of the raw covariates.

### Implication for `temp_fc_*`

The `v1_temp` version additionally constructs the following forecast-summary features:

- `temp_fc_mean24`
- `temp_fc_max24`
- `temp_fc_min24`
- `temp_fc_ramp24`

These are computed from the channel currently named `temp`. Under the present Oncor preprocessing, these `temp_fc_*` features are therefore derived from `SURDPOINTTEMPFAHRENHEIT`, not from true ambient air temperature.

### Recommended cleanup

For future cleanup, we recommend updating the preprocessing and README names so that the external channels reflect their actual meanings. For example:

- rename `temp` to `dewpoint_f`
- rename `workday` to `relative_humidity`
- rename `season` to `heat_index_f`

At the same time, we recommend **keeping** the humidity- and heat-related channels in the model, since they still provide useful weather context.

A better long-term solution is to **add the true external variables explicitly**, rather than replacing the current ones. In particular, future versions should consider including:

- true air temperature
- relative humidity
- heat index
- calendar / day-type indicator
- season label

This would make the README, code, and data semantics consistent, while preserving the useful weather variables already present in the current pipeline.

---

## Warmup Schedule

This training setup uses two warmup mechanisms:

### 1. Model warmup

A short internal warmup is applied at the model level:

- `warmup_ep = 10`

This warmup is passed into the model forward call and is only active during the early training stage.

### 2. Peak-loss warmup

The peak-aware loss terms are **not** applied at full strength from the start. Instead, they are linearly ramped up during the first 300 epochs:

- `V5_PEAK_WARM = 300`

The following peak-related loss weights increase gradually from 0 to their configured values over epochs 1 to 300:

- `lam_thr`
- `lam_q`
- `lam_time`

In other words, during the first 300 epochs, the model gradually transitions from primarily learning the base probabilistic forecasting objective to increasingly emphasizing peak fidelity. After epoch 300, these peak-loss weights reach their full configured values and remain fixed for the rest of training.



## Data Pipeline

### Weekly `.npz` Files

The preprocessing utilities generate one compressed weekly file per transformer:

- `processed_data/weeks_tensor_<XFMR>.npz`

Each file stores:

- `tensor`: weekly feature tensor of shape `(W, 168, F)`
- `features`
- `time_index`
- `time_index_str`
- `week_start`
- `week_end`
- `xfmr` : the data we used

Each weekly tensor is built by:

- sorting records by timestamp,
- creating a continuous hourly timeline,
- splitting into consecutive **168-hour weeks**,
- keeping only fully observed weeks. :contentReference[oaicite:24]{index=24}

## Combined Dataset

`Combine_Final.py` combines selected transformer weekly files into:

- `processed_data/weeks_tensor_all.npz`

The current combined dataset is built from four customer-group datasets:

- `residential+small_commercial`
- `small_commercial1`
- `small_commercial2`
- `large_commercial`

## ONCOR Feature Mapping Used by the Current Main Script

In the current ONCOR loader, the main script uses:

- `load  <- KWH`
- `temp  <- SURDPOINTTEMPFAHRENHEIT`
- `workday <- RELATIVEHUMIDITY`
- `season  <- HEATINDEXFAHRENHEIT`

**Important note:** in the current ONCOR implementation, `workday` and `season` are variable-slot names in the training pipeline, but for this dataset they are mapped to humidity and heat-index features rather than literal weekday/season labels.

## Forecast Horizon and Sample Construction

The current variational main script uses:

- `encoder_len_weeks = 1`
- `decoder_len_weeks = 1`
- `num_in_week = 168`
- `output_len = 3`

This means:

- each encoder input contains **1 week = 168 hourly points**,
- each decoder week also has **168 hourly points**,
- the model predicts **3 hours ahead per decoder position**,
- the number of rolling decoder positions is `168 - 3 + 1 = 166`.

So the model is **not** producing a single 166-hour horizon. Instead, it produces **166 rolling 3-hour forecasts** within the decoder week.

## Important Note on Graphs and Model Outputs

Several evaluation plots in the current code can be misread if their meaning is not stated clearly.

During evaluation, the plotting code extracts:

- `mu_first = mu_preds[:, :, 0]`
- `tgt_first = tgt[:, :, 0]`

That means the default graphs visualize **only the first horizon** from each rolling 3-hour forecast window.

### What the Default Plots Actually Show

- **`pred_only` plots**  
  Show the target first-horizon series and predicted first-horizon mean/std across all rolling decoder positions.

- **`with_hist` plots**  
  Show:
  - the 168-point encoder history,
  - followed by the sequence of **166 first-horizon rolling predictions**,
  - plus uncertainty bands.  
  These plots are useful, but the post-history segment is **not a single 166-hour rollout**. It is the first-step slice of all rolling decoder forecasts.

- **`global_overlay` plot**  
  Overlays first-horizon targets and predictions for all samples.

### Normalization Note

All features in `process_seq2seq_data()` are normalized using `MinMaxScaler`, so the default plots are displayed in **normalized space**, not original engineering units.

## LoRA Adaptation Details

The current LoRA workflow:

- loads the pretrained base checkpoint,
- replaces `decoder.head_mu` with `LoRALinear(...)`,
- replaces `decoder.head_logvar` with `LoRALinear(...)`,
- freezes all original model parameters,
- trains only LoRA parameters.

In the current script, LoRA fine-tuning is performed on the target transformer:

- `small_commercial2`


## Acknowledgements

🙋 **Acknowledgements** Developed by **Haoran Li, Muhao Guo, and Zhe Cheng**. If you use this repo in your work, please cite or acknowledge the project.

If you have any questions, please contact [haoran@asu.edu](mailto:haoran@asu.edu), [haorandd@mit.edu](mailto:haorandd@mit.edu), or [zcheng55@asu.edu](mailto:zcheng55@asu.edu).

## References

1. **H. Li, Z. Cheng, M. Guo, Y. Weng, Y. Sun, V. Tran, and J. Chainaranont**, *M2OE2-GL: A Family of Probabilistic Load Forecasters That Scales to Massive Customers*, arXiv preprint arXiv:2511.17623, 2025. [arXiv](https://arxiv.org/abs/2511.17623)

2. **H. Li, M. Guo, Y. Weng, M. Ilic, and G. Ruan**, *ExARNN: An Environment-Driven Adaptive RNN for Learning Non-Stationary Power Dynamics*, 2025 IEEE Power & Energy Society General Meeting (PESGM), pp. 1–5, 2025. [arXiv](https://arxiv.org/abs/2505.17488)

3. **H. Li, M. Guo, M. Ilic, Y. Weng, and G. Ruan**, *External Data-Enhanced Meta-Representation for Adaptive Probabilistic Load Forecasting*, arXiv preprint arXiv:2506.23201, 2025. [arXiv](https://arxiv.org/abs/2506.23201)

## Repository Structure

```text
M2oE2_load_forecast/
├── code/
│   ├── lora_utils_Final.py                  # LoRA implementation (Linear layers, parameter freezing)
│   ├── main_variational_peak_Final.py  # VAE model training/eval with Peak-Weighted Loss
│   ├── main_CNNGRU_var_v1.py          # CNN-GRU probabilistic model workflow
│   ├── main_LSTM_var_v1.py            # LSTM probabilistic model workflow
│   ├── main_RNN_var_v1_ft.py          # RNN/GRU probabilistic model workflow
│   ├── peak_metrics.py                # Auxiliary metrics for peak load analysis (ERCOT/PJM style)
│   ├── data_utils_Final.py                  # Data loading and preprocessing utilities
│   └── model_Final.py                    # Base model definitions (VAE/M2oE2)
└── README.md






