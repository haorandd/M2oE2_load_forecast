# M2oE2: Probabilistic Load Forecasting Framework via Low-Rank Adaptation (LoRA)

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## Overview

This repository contains the official implementation of the **M2oE2** (Many-to-One Elastic Ensemble) probabilistic load forecasting framework. This project addresses the critical challenge of forecasting localized power grid loads (e.g., specific distribution transformers) under conditions of data scarcity and high volatility.

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

## Repository Structure

```text
M2oE2_load_forecast/
├── code/
│   ├── lora_utils.py                  # LoRA implementation (Linear layers, parameter freezing)
│   ├── main_variational_base_peak.py  # VAE model training/eval with Peak-Weighted Loss
│   ├── main_CNNGRU_var_v1.py          # CNN-GRU probabilistic model workflow
│   ├── main_LSTM_var_v1.py            # LSTM probabilistic model workflow
│   ├── main_RNN_var_v1_ft.py          # RNN/GRU probabilistic model workflow
│   ├── peak_metrics.py                # Auxiliary metrics for peak load analysis (ERCOT/PJM style)
│   ├── data_utils.py                  # Data loading and preprocessing utilities
│   └── model_v1.py                    # Base model definitions (VAE/M2oE2)
└── README.md