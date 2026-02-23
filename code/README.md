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
│   ├── lora_utils.py                  # LoRA implementation (Linear layers, parameter freezing)
│   ├── main_variational_base_peak.py  # VAE model training/eval with Peak-Weighted Loss
│   ├── main_CNNGRU_var_v1.py          # CNN-GRU probabilistic model workflow
│   ├── main_LSTM_var_v1.py            # LSTM probabilistic model workflow
│   ├── main_RNN_var_v1_ft.py          # RNN/GRU probabilistic model workflow
│   ├── peak_metrics.py                # Auxiliary metrics for peak load analysis (ERCOT/PJM style)
│   ├── data_utils.py                  # Data loading and preprocessing utilities
│   └── model_v1.py                    # Base model definitions (VAE/M2oE2)
└── README.md


