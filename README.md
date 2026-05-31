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

| Model | QS | WS (50%) | WS (90%) |
|---|---:|---:|---:|
| iQRNN [1] | 2.71 | 23.94 | 47.11 |
| Q-ResNet [2] | 2.69 | 23.64 | 46.50 |
| ResNetPlus [3] | 2.52 | 22.41 | 42.63 |
| Basic QRNN [4] | 2.45 | 21.75 | 41.80 |
| AHLC-QRNN [4] | 2.42 | 21.43 | 40.84 |
| **GRU-M²oE² (Weather station: 25 → 1)** | **2.31** | **20.41** | **39.72** |
| **GRU-M²oE² (Weather station: 25 → 3)** | **2.27** | **20.06** | **39.35** |

To reproduce the GEFCom2014 benchmark-style experiment, go to code/GEFCom2014 for detail information.

The GEFCom2014-L data were originally released as the appendix / supplementary material of the GEFCom2014 paper. Please cite the original paper when using this dataset:

Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli, and Rob J. Hyndman, "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond," International Journal of Forecasting, 32(3), 896–913, 2016.

The dataset can be accessed through the paper supplementary material. A download link is also provided by Tao Hong's GEFCom2014 load forecasting data page.






📜 License
This repository is licensed under the MIT License.

🙋‍♂️ Acknowledgements
Developed by [Haoran Li, Zhe Cheng and Muhao Guo]. If you use this repo in your work, please cite or acknowledge the project.

If you have any questions, please contact lhaoran@asu.edu, haorandd@mit.edu or zcheng55@asu.edu.

## Citation

If you use this repository or the M²oE² model, please cite the M²oE² paper:

```bibtex
@article{li2025m2oe2,
  title={External Data-Enhanced Meta-Representation for Adaptive Probabilistic Load Forecasting},
  author={Li, Haoran and Guo, Muhao and Ilic, Marija and Weng, Yang and Ruan, Guangchun},
  journal={arXiv preprint arXiv:2506.23201},
  year={2025}
}
@article{li2025m2oe2gl,
  title={M$^2$OE$^2$-GL: A Family of Probabilistic Load Forecasters That Scales to Massive Customers},
  author={Li, Haoran and Cheng, Zhe and Guo, Muhao and Weng, Yang and Sun, Yannan and Tran, Victor and Chainaranont, John},
  journal={arXiv preprint arXiv:2511.17623},
  year={2025},
  eprint={2511.17623},
  archivePrefix={arXiv}
}
@inproceedings{li2025exarnn,
  title={ExARNN: An Environment-Driven Adaptive RNN for Learning Non-Stationary Power Dynamics},
  author={Li, Haoran and Guo, Muhao and Weng, Yang and Ilic, Marija and Ruan, Guangchun},
  booktitle={2025 IEEE Power \& Energy Society General Meeting (PESGM)},
  pages={1--5},
  year={2025}
}
```







