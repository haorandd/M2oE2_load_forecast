# M2oE2_load_forecast
# 🧠 Probabilistic Seq2Seq Forecasting with Mixture-of-Experts

Paper Link: [External Data-Enhanced Meta-Representation for
Adaptive Probabilistic Load Forecasting](https://arxiv.org/pdf/2506.23201)

Link to Hugging Face: [Data](https://huggingface.co/datasets/MuhaoGuo/M2oE2TimeSeriesData), [Model](https://huggingface.co/MuhaoGuo/M2oE2). 

This repository implements a **variational sequence-to-sequence forecasting model** with a **Mixture-of-Experts (MoE)** architecture for multivariate energy time series (e.g., solar or building load). The model provides probabilistic forecasts with uncertainty quantification and is dynamically parameterized by contextual features (temperature, workday, season).

---

## 🔍 Key Features

- Encoder-decoder Seq2Seq model using GRUs
- MetaNet + GatingNet for expert-specific dynamic parameter generation
- Variational inference with reparameterization
- Two decoder types: fixed or predicted variance
- Evaluation metrics: **MSE**, **CRPS**, and **NLL**
- Supports multiple datasets (Solar, Residential, Building, etc.)

---

## 📁 Directory Structure


├── main_M2oE2_prob.py # Training and evaluation pipeline

├── model.py # Model architecture (MetaNet, Variational Seq2Seq)

├── data_utils.py # Data loading, normalization, batching

├── result/ # Folder for saving plots

├── README.md # Project documentation


---

## 📊 Datasets

Supports the following datasets with weekly resolution:
- **Solar**
- **Residential**
- **Building**
- **Spanish**
- **Consumption**

Each dataset includes:
- Energy load (`load`)
- Temperature (`temp`)
- Workday indicator (`workday`)
- Seasonal index (`season`)

All features are normalized and Gaussian-smoothed before training.

---

## 🚀 Getting Started

Step 1: Clone the Repository

Step 2: Install Dependencies

Step 3: Download Data

Download the dataset [here](https://zenodo.org/records/15767099) and place it into the appropriate `data/` directory.


Step 4: Run the Model

python main_M2oE2_prob.py

This will:

Train the model (unless a saved model exists)

Save the best checkpoint (Solar_M2OE2_best_model.pt, etc.)

Evaluate on test data and generate forecast plots with uncertainty bands in ./result/

📈 Metrics
MSE (Mean Squared Error): for accuracy

NLL (Negative Log-Likelihood): under-predicted Gaussian

CRPS (Continuous Ranked Probability Score): for probabilistic forecast quality

🧠 Model Architecture
MetaNet: Dynamically generates projection matrices from contextual features

GatingNet: Learns soft expert selection

Encoder: Processes input with GRU and outputs a latent distribution

Decoder: Generates multi-step probabilistic forecasts

Two decoder variants:

VariationalDecoder_meta_fixvar: fixed uncertainty

VariationalDecoder_meta_predvar: predicted uncertainty (used by default)

📌 Example Results
Forecasts are visualized with predicted mean and ±1 std confidence band.
Saved plots can be found in the ./result/ folder after training.


## 📊 GEFCom2014 Benchmark-Style Comparison

We additionally evaluate M2OE2 on the GEFCom2014-L load forecasting benchmark under a setting aligned with prior probabilistic forecasting studies. The model uses a 24-hour forecasting horizon (`K=24`), a learning rate of `1e-3`, average temperature information from 25 weather stations, and calendar features including workday and season labels. The data split follows the benchmark protocol: 2006–2009 for training, 2010 for validation, and 2011 for testing.

The evaluation metrics are computed after inverse normalization on the raw load scale. We report Quantile Score / Pinball loss and Winkler Scores for 50% and 90% prediction intervals.

| Model | QS / Pinball | WS (50%) | WS (90%) |
|---|---:|---:|---:|
| iQRNN | 2.71 | 23.94 | 47.11 |
| Q-ResNet | 2.69 | 23.64 | 46.50 |
| ResNetPlus | 2.52 | 22.41 | 42.63 |
| Basic QRNN | 2.45 | 21.75 | 41.80 |
| AHLC-QRNN | 2.42 | 21.43 | 40.84 |
| **M2OE2** | **1.99** | **17.84** | **30.37** |

Note: The M2OE2 result is evaluated under a rolling first-step setting, where only the first forecast horizon of each rolling prediction window is used. Therefore, this comparison demonstrates rolling short-term probabilistic forecasting performance under the benchmark data split and metric scale, rather than a strict full 24-hour direct day-ahead evaluation.

📜 License
This repository is licensed under the MIT License.

🙋‍♂️ Acknowledgements
Developed by [Haoran Li and Muhao Guo]. If you use this repo in your work, please cite or acknowledge the project.

If you have any questions, please contact lhaoran@asu.edu or haorandd@mit.edu







