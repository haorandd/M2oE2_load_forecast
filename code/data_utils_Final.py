import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import re, os
from typing import Optional, List, Tuple


import os
# --- Utility Functions ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- Data Loading and Initial Processing (from original) ---
def get_data_building_weather_weekly():
    # path = "C:\\Software\\Probabilistic_Forecasting\\Data\\ashrae-energy-prediction"
    # df_train = pd.read_csv(path + "\\train.csv")
    # df_weather = pd.read_csv(path + "\\weather_train.csv")
    # df_meta = pd.read_csv(path + "\\building_metadata.csv")

    df_train = pd.read_csv("/Users/muhaoguo/Documents/study/Paper_Projects/PESGM/data/ashrae-energy-prediction/train.csv")
    df_weather = pd.read_csv("/Users/muhaoguo/Documents/study/Paper_Projects/PESGM/data/ashrae-energy-prediction/weather_train.csv")
    df_meta = pd.read_csv("/Users/muhaoguo/Documents/study/Paper_Projects/PESGM/data/ashrae-energy-prediction/building_metadata.csv")


    df = df_train.merge(df_meta, on='building_id').merge(df_weather, on=['site_id', 'timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Filter for a specific building, meter, and a reduced date range for faster processing if needed
    df = df[(df['building_id'] == 2) & (df['meter'] == 0)]
    df = df[(df['timestamp'] >= '2016-01-04') & (df['timestamp'] < '2017-01-04')]  # Ensure enough data for ~50 weeks
    df['Date'] = df['timestamp'].dt.date
    df['day_of_week'] = df['timestamp'].dt.dayofweek  # Monday=0, Sunday=6

    def get_season(month):
        return {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}[month]

    # Ensure 'meter_reading' and 'air_temperature' are present and numeric
    df['meter_reading'] = pd.to_numeric(df['meter_reading'], errors='coerce').fillna(0)
    df['air_temperature'] = pd.to_numeric(df['air_temperature'], errors='coerce').fillna(method='ffill').fillna(
        method='bfill').fillna(15)

    measurement_columns = ['meter_reading', 'air_temperature', 'Date', 'timestamp']
    # Ensure columns exist, add placeholders if not
    for col in measurement_columns:
        if col not in df.columns and col not in ['Date']:  # Date is derived
            df[col] = 0 if col != 'timestamp' else pd.NaT

    grouped = df.groupby('Date')[measurement_columns + ['day_of_week']]

    array_3d, labels_3d, seasons_3d = [], [], []
    dates = sorted(grouped.groups.keys())
    if not dates:
        print("Warning: No data after filtering in get_data_building_weather_weekly.")
        # Return empty arrays with expected dimensions to avoid downstream errors immediately
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    for date_val in dates:
        group_df = grouped.get_group(date_val)
        if group_df.empty or len(group_df) != 24:  # Assuming hourly data, fill if not
            # Create a full day template
            full_day_timestamps = pd.to_datetime([f"{date_val} {h:02d}:00:00" for h in range(24)])
            template_df = pd.DataFrame({'timestamp': full_day_timestamps})
            group_df = pd.merge(template_df, group_df, on='timestamp', how='left')
            group_df['Date'] = group_df['timestamp'].dt.date
            group_df['day_of_week'] = group_df['timestamp'].dt.dayofweek
            for col in ['meter_reading', 'air_temperature']:
                group_df[col] = group_df[col].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
            group_df = group_df.fillna({'meter_reading': 0, 'air_temperature': 15})  # final fallback

        arr = group_df[measurement_columns].values
        label = 0 if group_df['day_of_week'].iloc[0] < 5 else 1  # Weekday/Weekend
        season = get_season(group_df['timestamp'].iloc[0].month)
        array_3d.append(arr)
        labels_3d.append(np.full(len(arr), label))
        seasons_3d.append(np.full(len(arr), season))

    n_full_weeks = len(array_3d) // 7
    if n_full_weeks == 0:
        print("Warning: Not enough daily data to form even one full week.")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    energy, temp, times, workday, season_feat = [], [], [], [], []
    for w in range(n_full_weeks):
        wk = slice(w * 7, (w + 1) * 7)
        week_data = array_3d[wk]
        week_labels = labels_3d[wk]
        week_seasons = seasons_3d[wk]

        e = np.concatenate([np.asarray(d[:, 0], dtype=float) for d in week_data])
        t = np.concatenate([np.asarray(d[:, 1], dtype=float) for d in week_data])
        ts = np.concatenate([np.asarray(d[:, 3]) for d in week_data])  # timestamp objects
        wl = np.concatenate([np.asarray(lbl, dtype=int) for lbl in week_labels])
        sl = np.concatenate([np.asarray(seas, dtype=int) for seas in week_seasons])

        if e.shape[0] != 168:  # Skip incomplete weeks silently or handle
            # print(f"Skipping week {w} due to incomplete data: {e.shape[0]} points")
            continue

        e = gaussian_filter1d(e, sigma=1)
        t = gaussian_filter1d(t, sigma=1)

        energy.append(e)
        temp.append(t)
        times.append(ts)
        workday.append(wl)
        season_feat.append(sl)

    return np.array(times, dtype=object), np.array(energy), np.array(temp), np.array(workday), np.array(season_feat)


def gaussian_nll_loss(mu, logvar, target):
    # mu, logvar, target → same shape [B, L+1, output_len, output_dim]
    nll = 0.5 * (logvar + np.log(2 * np.pi) + ((target - mu) ** 2) / logvar.exp())
    return nll.mean()  # average over all elements

def kl_loss(mu_z, logvar_z):
    return -0.5 * torch.mean(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())


def process_seq2seq_data(
        feature_dict,
        *,
        train_ratio        = 0.7,
        norm_features      = ('load', 'temp'),   # ignored: we now normalize ALL features
        output_len         = 24,          # steps each decoder step predicts
        encoder_len_weeks  = 1,
        decoder_len_weeks  = 1,
        num_in_week        = 168,
        device             = None):
    """
    feature_dict: {'load': np.ndarray [weeks, 168], 'featureX': same shape, ...}
    External features are all keys except 'load'.

    NOTE: This version normalizes *all* features with MinMaxScaler (per-feature, global),
    regardless of their names. The 'norm_features' parameter is ignored for compatibility.
    """
    # 1) flatten + scale (normalize ALL features)
    processed, scalers = {}, {}
    for k, arr in feature_dict.items():
        if arr.size == 0:
            raise ValueError(f"feature '{k}' is empty.")
        vec = np.asarray(arr, dtype=float).flatten()

        sc = MinMaxScaler()
        processed[k] = sc.fit_transform(vec.reshape(-1, 1)).flatten()
        scalers[k] = sc

    n_weeks = feature_dict['load'].shape[0]
    need_weeks = encoder_len_weeks + decoder_len_weeks
    if n_weeks < need_weeks:
        raise ValueError(f"Need ≥{need_weeks} consecutive weeks, found {n_weeks}.")

    enc_seq_len = encoder_len_weeks * num_in_week
    dec_seq_len = decoder_len_weeks * num_in_week
    L = dec_seq_len - output_len
    if L <= 0:
        raise ValueError("`output_len` must be smaller than decoder sequence length.")

    ext_keys = [k for k in feature_dict.keys() if k != 'load']
    K_ext = len(ext_keys)

    # 2) build samples (stride = 1 week)
    X_enc_l, X_dec_in_l, Y_dec_target = [], [], []
    X_enc_ext, X_dec_in_ext = [], []

    last_start = n_weeks - need_weeks
    for w in range(last_start + 1):
        enc_start =  w * num_in_week
        enc_end   = (w + encoder_len_weeks) * num_in_week
        dec_start =  enc_end
        dec_end   =  dec_start + dec_seq_len

        # load sequences
        enc_l = processed['load'][enc_start:enc_end]
        dec_full_l = processed['load'][dec_start:dec_end]

        # encoder/decoder externals
        if K_ext > 0:
            enc_ext = np.stack([processed[k][enc_start:enc_end] for k in ext_keys], axis=-1)  # [enc_len, K]
            dec_ext = np.stack([processed[k][dec_start: dec_start + L] for k in ext_keys], axis=-1)  # [L, K]
        else:
            enc_ext = np.empty((enc_seq_len, 0), dtype=np.float32)
            dec_ext = np.empty((L, 0), dtype=np.float32)

        # targets (sliding windows of length output_len across decoder horizon)
        targets = np.stack([dec_full_l[i:i+output_len] for i in range(L+1)], axis=0)  # [L+1, output_len]

        X_enc_l.append(enc_l)
        X_dec_in_l.append(dec_full_l[:L])
        X_enc_ext.append(enc_ext)
        X_dec_in_ext.append(dec_ext)
        Y_dec_target.append(targets)

    # 3) pack → tensors
    to_tensor = lambda a: torch.tensor(a, dtype=torch.float32).to(device)

    data_tensors = {
        'X_enc_l'      : to_tensor(np.array(X_enc_l)).unsqueeze(-1),         # [B, enc_len, 1]
        'X_enc_ext'    : to_tensor(np.array(X_enc_ext)),                      # [B, enc_len, K_ext]
        'X_dec_in_l'   : to_tensor(np.array(X_dec_in_l)).unsqueeze(-1),       # [B, L, 1]
        'X_dec_in_ext' : to_tensor(np.array(X_dec_in_ext)),                   # [B, L, K_ext]
        'Y_dec_target' : to_tensor(np.array(Y_dec_target)).unsqueeze(-1),     # [B, L+1, output_len, 1]
    }

    for k, v in data_tensors.items():
        print(f"{k:15s} {tuple(v.shape)}")

    # 4) split
    B = data_tensors['X_enc_l'].shape[0]
    split = int(train_ratio * B)
    train_dict = {k: v[:split] for k, v in data_tensors.items()}
    test_dict  = {k: v[split:] for k, v in data_tensors.items()}
    return train_dict, test_dict, scalers


def visualise_one_sample(data_dict, sample_idx=0):
    """Draw a single figure with three subplots:
       1) encoder load, 2) decoder load, 3) heat‑map of Y_dec_target_l."""
    enc = data_dict['X_enc_t'][sample_idx].cpu().numpy().squeeze(-1)
    dec = data_dict['X_dec_in_t'][sample_idx].cpu().numpy().squeeze(-1)
    tgt = data_dict['Y_dec_target_l'][sample_idx].cpu().numpy().squeeze(-1)  # [L, output_len]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)

    axes[0].plot(enc)
    axes[0].set_title("Encoder input")
    axes[0].set_xlabel("Time step"); axes[0].set_ylabel("scaled")

    axes[1].plot(dec)
    axes[1].set_title("Decoder input")
    axes[1].set_xlabel("Time step")


    axes[2].plot(tgt[0])
    axes[2].plot(tgt[1])
    axes[2].plot(tgt[2])
    axes[2].set_title("Decoder target")
    axes[2].set_xlabel("Time step")

    plt.show()


def plot_etth_weekly_features(times, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT, week_idx=0):
    """
    Plot all ETTh features (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT) for a given week.

    Parameters
    ----------
    times : np.ndarray (dtype=object)
        Array of timestamp arrays, one per week.
    HUFL, HULL, MUFL, MULL, LUFL, LULL, OT : np.ndarray
        Feature arrays, each of shape (n_weeks, 168).
    week_idx : int
        Index of the week to visualize (default=0).
    """
    if len(times) == 0:
        print("No data to plot.")
        return

    if week_idx >= len(times):
        raise IndexError(f"week_idx {week_idx} exceeds available weeks ({len(times)}).")

    # extract week data
    t  = pd.to_datetime(times[week_idx])
    h1 = HUFL[week_idx]
    h2 = HULL[week_idx]
    m1 = MUFL[week_idx]
    m2 = MULL[week_idx]
    l1 = LUFL[week_idx]
    l2 = LULL[week_idx]
    o  = OT[week_idx]

    plt.figure(figsize=(12, 6))
    plt.plot(t, h1, label="HUFL")
    plt.plot(t, h2, label="HULL")
    plt.plot(t, m1, label="MUFL")
    plt.plot(t, m2, label="MULL")
    plt.plot(t, l1, label="LUFL")
    plt.plot(t, l2, label="LULL")
    plt.plot(t, o,  label="OT", linewidth=2.0, color='black')

    # plt.title(f"ETTh1 Weekly Features (Week {week_idx})")
    plt.xlabel("Time")
    plt.ylabel("Normalized Value")
    plt.legend(loc="upper right", ncol=4, fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


    if len(times) == 0:
        print("No data to plot.")
        return
    # --- concatenate all weeks ---
    full_time = np.concatenate(times)
    full_HUFL = np.concatenate(HUFL)
    full_HULL = np.concatenate(HULL)
    full_MUFL = np.concatenate(MUFL)
    full_MULL = np.concatenate(MULL)
    full_LUFL = np.concatenate(LUFL)
    full_LULL = np.concatenate(LULL)
    full_OT   = np.concatenate(OT)

    feature_names = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    features = [full_HUFL, full_HULL, full_MUFL, full_MULL, full_LUFL, full_LULL, full_OT]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#000000"]

    # --- plot all features over continuous time ---
    plt.figure(figsize=(15, 6))
    for name, arr, color in zip(feature_names, features, colors):
        plt.plot(full_time, arr, label=name, linewidth=1.2, color=color)

    plt.title("ETTh1 All Features over Continuous Time", fontsize=13)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend(ncol=4, fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



# def plot_data(times, energy, temp, workday, season_feat,
#                    alpha=0.5, lw=1.0, cmap="viridis"):
#     """
#     Overlay *all* weeks in four side-by-side sub-figures.
#
#     Parameters
#     ----------
#     times, energy, temp, workday, season_feat : list/ndarray
#         Output from your get_data_…_weekly routine.
#     alpha : float
#         Per-curve transparency (≤1).  Lower → less clutter.
#     lw : float
#         Line width.
#     cmap : str or matplotlib Colormap
#         Used to give each week a slightly different colour.
#     """
#     n_weeks = len(times)
#     if n_weeks == 0:
#         print("Nothing to plot.")
#         return
#
#     cmap = "coolwarm"
#     colours = plt.cm.get_cmap(cmap, n_weeks)
#
#     ###########################
#     # # colour map to distinguish weeks (wraps if >256)
#
#     # fig, axes = plt.subplots(2, 1, figsize=(22, 8), sharex=False)
#     # fig.suptitle(f"Energy and Temperature (value-colored)", fontsize=14, fontweight='bold')
#     # # Date format: show Year + Month
#     # loc = mdates.AutoDateLocator()
#     # fmt = mdates.DateFormatter("%Y\n%b %d")
#     # # --- Plot each variable with color by value ---
#     # for ax, y_data, label in zip(
#     #     axes,
#     #     [energy, temp],
#     #     ["Energy (norm.)", "Temperature (norm.)"]
#     # ):
#     #     for w in range(n_weeks):
#     #         t = pd.to_datetime(times[w])
#     #         y = np.asarray(y_data[w])
#     #
#     #         # normalize y to [0, 1] for colormap scaling
#     #         norm_y = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-8)
#     #         colors = plt.cm.get_cmap(cmap)(norm_y)
#     #
#     #         # plot line with per-segment color
#     #         for i in range(len(t)-1):
#     #             ax.plot(t[i:i+2], y[i:i+2], color=colors[i], lw=lw, alpha=alpha)
#     #
#     #     ax.set_title(label)
#     #     ax.set_ylabel(label)
#     #     ax.xaxis.set_major_locator(loc)
#     #     ax.xaxis.set_major_formatter(fmt)
#     #     ax.tick_params(axis="x", rotation=0, labelsize=10)
#     # plt.tight_layout()
#     # plt.savefig("./result/load-temp.pdf")
#     # plt.show()
#
#     ###########################
#     fig, axes = plt.subplots(
#         nrows=2, ncols=1, figsize=(22, 8),
#         sharex = False, sharey = False,
#         gridspec_kw={"wspace": 1})
#     # date_fmt = mdates.DateFormatter("%b\n%d")
#     date_fmt = mdates.DateFormatter("%Y\n%b %d")  # ← includes year
#     # -------------------------------------------------------------
#     # iterate once, plotting the same week on all four axes
#     # -------------------------------------------------------------
#     for w in range(n_weeks):
#         c = colours(w)
#         axes[0].plot(times[w], energy[w],  color=c, alpha=alpha, lw=lw)
#         axes[1].plot(times[w], temp[w],    color=c, alpha=alpha, lw=lw)
#         # axes[2].step(times[w], workday[w], where="mid",
#         #              color=c, alpha=alpha, lw=lw)
#         # axes[3].step(times[w], season_feat[w], where="mid",
#         #              color=c, alpha=alpha, lw=lw)
#     axes[0].set_title("Energy (norm.)")
#     axes[0].set_ylabel("0–1")
#     axes[1].set_title("Temperature (norm.)")
#     # axes[2].set_title("Weekend flag")
#     # axes[2].set_ylim(-0.1, 1.1)
#     # axes[3].set_title("Season (0–3)")
#     # axes[3].set_ylim(-0.2, 3.2)
#
#     for ax in axes:
#         ax.xaxis.set_major_formatter(date_fmt)
#         ax.tick_params(axis="x", rotation=0, labelsize=14)
#     fig.suptitle(f"Overlay of {n_weeks} weeks", fontsize=15)
#     plt.tight_layout()
#     plt.show()
#
#     ##########################################################################################
#     fig, axes = plt.subplots(  nrows=1, ncols=4, figsize=(22, 4),  sharex=False, sharey=False,  gridspec_kw={"wspace": 0.25})
#     date_fmt = mdates.DateFormatter("%b\n%d")
#     # -------------------------------------------------------------
#     # iterate once, plotting the same week on all four axes
#     # -------------------------------------------------------------
#     for w in range(n_weeks):
#         c = colours(w)
#         axes[0].plot(energy[w], color=c, alpha=alpha, lw=lw)
#         axes[1].plot(temp[w], color=c, alpha=alpha, lw=lw)
#         axes[2].plot(workday[w], color=c, alpha=alpha, lw=lw)
#         axes[3].plot(season_feat[w], color=c, alpha=alpha, lw=lw)
#
#     # -------------------------------------------------------------
#     # cosmetics
#     # -------------------------------------------------------------
#     axes[0].set_title("Energy (norm.)")
#     axes[0].set_ylabel("0–1")
#     axes[1].set_title("Temperature (norm.)")
#     axes[2].set_title("Weekend flag")
#     axes[2].set_ylim(-0.1, 1.1)
#     axes[3].set_title("Season (0–3)")
#     axes[3].set_ylim(-0.2, 3.2)
#
#     for ax in axes:
#         ax.xaxis.set_major_formatter(date_fmt)
#         ax.tick_params(axis="x", rotation=45, labelsize=8)
#
#     fig.suptitle(f"Overlay of {n_weeks} weeks", fontsize=15, y=1.02)
#     plt.tight_layout()
#     plt.show()
#


def make_loader(data_dict, batch_size, shuffle=True):
    """
    Returns: batch =
        (enc_l, enc_t, enc_w, enc_s,
         dec_l, dec_t, dec_w, dec_s,
         tgt)
    Shapes:
        enc_* : [B, enc_seq, 1]
        dec_* : [B, L, 1]
        tgt   : [B, L+1, output_len, 1]
    """
    tensors = (
        data_dict['X_enc_l'], data_dict['X_enc_t'],
        data_dict['X_enc_w'], data_dict['X_enc_s'],
        data_dict['X_dec_in_l'], data_dict['X_dec_in_t'],
        data_dict['X_dec_in_w'], data_dict['X_dec_in_s'],
        data_dict['Y_dec_target_l']
    )
    ds = TensorDataset(*tensors)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def reconstruct_sequence(pred_seq):
    """
    Averages overlapping predictions from [L+1, output_len] into [L+output_len]
    Args:
        pred_seq: [L+1, output_len] – single sample prediction
    Returns:
        avg_pred: [L+output_len] – averaged sequence
    """
    L_plus_1, output_len = pred_seq.shape
    total_len = L_plus_1 + output_len - 1
    sum_seq = torch.zeros(total_len, device=pred_seq.device)
    count_seq = torch.zeros(total_len, device=pred_seq.device)

    for t in range(L_plus_1):
        sum_seq[t:t+output_len] += pred_seq[t]
        count_seq[t:t+output_len] += 1

    return sum_seq / count_seq  # [L+output_len]


def get_load_temperature_spanish():
    '''
    https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather
    '''
    # Load the energy dataset and weather features
    energy_df = pd.read_csv('data/Spanish/energy_dataset.csv')
    weather_df = pd.read_csv('data/Spanish/weather_features.csv')

    # Convert timestamp columns to datetime format for easier merging and plotting
    energy_df['time'] = pd.to_datetime(energy_df['time'])
    weather_df['time'] = pd.to_datetime(weather_df['dt_iso'])

    # Merge datasets on the 'timestamp' column
    merged_df = pd.merge(energy_df, weather_df, on='time', how='inner')
    merged_df = merged_df[['time', 'temp', 'total load actual']].dropna()
    merged_df = merged_df[::5]

    time = merged_df["time"].values
    temperature = (merged_df["temp"] - 273.15).values # from Kelvin (K) to degrees Celsius (°C),
    load = merged_df["total load actual"].values/1000  # from MW to (×10³ MW)

    temperature = gaussian_filter1d(temperature, sigma=2)
    load = gaussian_filter1d(load, sigma=2)

    # Plotting temperature and load on the same figure
    fig, ax1 = plt.subplots(figsize=(14, 6))
    # Plot temperature with left y-axis
    ax1.plot(time, temperature, label='Temperature', color='orange', linewidth=2)
    ax1.set_ylabel('Temperature (°C)', color='orange', fontsize=20)
    ax1.tick_params(axis='y', labelcolor='orange', labelsize=20)
    ax1.tick_params(axis='x', labelsize=20)
    # Create a second y-axis for load
    ax2 = ax1.twinx()
    ax2.plot(time, load, label='Power Load', color='darkblue', linewidth=2)
    ax2.set_ylabel('Power Load (×10³ MW)', color='darkblue', fontsize=20)
    ax2.tick_params(axis='y', labelcolor='darkblue', labelsize=20)
    ax2.tick_params(axis='x', labelsize=20)
    # Title and layout adjustments
    fig.suptitle('Temperature and Power Load Over Time', fontsize=20)
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    # plt.savefig("./results/raw_load_temp_spanish.pdf")
    plt.show()
    print(time.shape, load.shape, temperature.shape)
    return time, load, temperature


def get_data_spanish_weekly():
    """
    Weekly load-temperature slices for Spain
    —————————————————————————————————————————————————
    Returns
    -------
    times        : np.ndarray, dtype=object, shape (n_weeks,)
    energy       : np.ndarray,               shape (n_weeks, 168)
    temp         : np.ndarray,               shape (n_weeks, 168)
    workday      : np.ndarray,               shape (n_weeks, 168)
    season_feat  : np.ndarray,               shape (n_weeks, 168)
    """
    # ---------- raw files --------------------------------------------------
    p_energy  = "data/Spanish/energy_dataset.csv"
    p_weather = "data/Spanish/weather_features.csv"

    # ---------- pre-processing & merge ------------------------------------
    energy_df = pd.read_csv(p_energy)
    weather_df = pd.read_csv(p_weather)

    energy_df["time"] = pd.to_datetime(energy_df["time"], utc=True)
    weather_df["time"] = pd.to_datetime(weather_df["dt_iso"], utc=True)
    df = pd.merge(energy_df, weather_df, on="time", how="inner")
    df = df[::5]
    df = df[1:]

    df["time"] = df["time"].dt.tz_convert(None)  # or .dt.tz_localize(None)
    df = df[["time", "temp", "total load actual"]].dropna()
    df["Date"] = df["time"].dt.date  # now works
    df["day_of_week"] = df["time"].dt.dayofweek
    df["air_temperature"] = (df["temp"] - 273.15).astype(float)
    df["meter_reading"] = (df["total load actual"] / 1000).astype(float)

    # ---------- season helper ---------------------------------------------
    def get_season(month: int) -> int:
        return {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}[month]

    # ---------- daily grouping (24 samples each) ---------------------------
    meas_cols = ["meter_reading", "air_temperature", "Date", "time"]
    grouped   = df.groupby("Date")[meas_cols + ["day_of_week"]]

    array_3d, labels_3d, seasons_3d = [], [], []
    for date_val in sorted(grouped.groups.keys()):
        gdf = grouped.get_group(date_val)

        # make sure we have *exactly* 24 hourly rows
        if len(gdf) != 24:
            full_hours = pd.date_range(start=f"{date_val} 00:00:00",
                                       end=f"{date_val} 23:00:00",
                                       freq="h")
            tmpl = pd.DataFrame({"time": full_hours})
            gdf  = pd.merge(tmpl, gdf, on="time", how="left")
            gdf["Date"]        = gdf["time"].dt.date
            gdf["day_of_week"] = gdf["time"].dt.dayofweek
            for c in ["meter_reading", "air_temperature"]:
                gdf[c] = (gdf[c]
                          .interpolate("linear")
                          .ffill()
                          .bfill()
                          )
            gdf.fillna({"meter_reading": 0, "air_temperature": 15}, inplace=True)

        arr     = gdf[meas_cols].values
        w_label = 0 if gdf["day_of_week"].iloc[0] < 5 else 1
        season  = get_season(gdf["time"].iloc[0].month)

        array_3d.append(arr)
        labels_3d.append(np.full(len(arr), w_label))
        seasons_3d.append(np.full(len(arr), season))

    # ---------- pack consecutive days into full weeks ---------------------
    n_full_weeks = len(array_3d) // 7
    if n_full_weeks == 0:
        return (np.array([]),) * 5

    energy, temp, times, workday, season_feat = [], [], [], [], []
    for w in range(n_full_weeks):
        wk     = slice(w * 7, (w + 1) * 7)
        week_d = array_3d[wk]
        w_lbls = labels_3d[wk]
        w_seas = seasons_3d[wk]

        e  = np.concatenate([d[:, 0].astype(float) for d in week_d])
        t  = np.concatenate([d[:, 1].astype(float) for d in week_d])
        ts = np.concatenate([d[:, 3] for d in week_d])          # timestamps
        wl = np.concatenate([lbl.astype(int) for lbl in w_lbls])
        sl = np.concatenate([s.astype(int) for s in w_seas])

        if e.size != 168:     # incomplete week – skip
            continue

        energy.append(gaussian_filter1d(e, sigma=1))
        temp.append(gaussian_filter1d(t, sigma=1))
        times.append(ts)
        workday.append(wl)
        season_feat.append(sl)

    return (np.array(times, dtype=object),
            np.array(energy),
            np.array(temp),
            np.array(workday),
            np.array(season_feat))


def get_data_power_consumption():
    """
    https://www.kaggle.com/datasets/fedesoriano/electric-power-consumption
    Loads a CSV containing at least:
      ['Date Time', 'Temperature', 'Zone 1 Power Consumption']
    and does a simple time-series plot of Zone 1 vs. Temperature.
    """
    # 1) Load data
    file_path = "data/Consumption/powerconsumption.csv"  # <-- Adjust to your actual CSV
    df = pd.read_csv(file_path)

    # 2) Parse datetime and sort
    #    We assume a combined 'Date Time' column, like '2020-01-01 00:10:00'
    df['Date Time'] = pd.to_datetime(df['Datetime'])
    df.sort_values(by='Date Time', inplace=True)

    # 3) Select only needed columns
    #    We pick 'Zone 1 Power Consumption' & 'Temperature'
    df_filtered = df[['Date Time', 'Temperature', 'PowerConsumption_Zone1']].copy()

    # 4) Convert to numeric (in case CSV has strings)
    #    Coerce errors => NaN
    df_filtered['Temperature'] = pd.to_numeric(df_filtered['Temperature'], errors='coerce')
    df_filtered['Zone 1 Power Consumption'] = pd.to_numeric(df_filtered['PowerConsumption_Zone1'], errors='coerce')

    # 5) Drop rows with missing values if needed
    df_filtered.dropna(subset=['Temperature', 'Zone 1 Power Consumption'], inplace=True)
    scaler = MinMaxScaler()
    df_filtered[['Temperature', 'Zone 1 Power Consumption']] = scaler.fit_transform(
        df_filtered[['Temperature', 'Zone 1 Power Consumption']]
    )

    # 6) Simple Plot: Time series of Zone1 and Temperature
    fig, ax1 = plt.subplots(figsize=(10, 5))
    # Plot Zone 1 Power on ax1
    color1 = 'tab:blue'
    ax1.set_xlabel('Date Time')
    ax1.set_ylabel('Zone 1 Power Consumption', color=color1)
    ax1.plot(df_filtered['Date Time'], df_filtered['Zone 1 Power Consumption'], color=color1, label='Zone1 Power')
    ax1.tick_params(axis='y', labelcolor=color1)

    # Create a second y-axis for Temperature
    ax2 = ax1.twinx()  # shares x-axis
    color2 = 'tab:red'
    ax2.set_ylabel('Temperature', color=color2)
    ax2.plot(df_filtered['Date Time'], df_filtered['Temperature'], color=color2, label='Temperature')
    ax2.tick_params(axis='y', labelcolor=color2)
    plt.title('Zone 1 Power Consumption and Temperature Over Time')
    fig.tight_layout()


    # --------------------------------------------------------
    # 7) Reshape the data: separate by date
    #    => new shape: [#dates, #values_in_one_day]
    # --------------------------------------------------------
    # Extract the date and the time of day (as a string HH:MM:SS)
    df_filtered['Date'] = df_filtered['Date Time'].dt.date
    df_filtered['TimeOfDay'] = df_filtered['Date Time'].dt.strftime('%H:%M:%S')

    # Pivot so each row is one date, each column is a time of day
    pivot_time = df_filtered.pivot(index='Date', columns='TimeOfDay', values='Date Time')
    pivot_power = df_filtered.pivot(index='Date', columns='TimeOfDay', values='Zone 1 Power Consumption')
    pivot_temp = df_filtered.pivot(index='Date', columns='TimeOfDay', values='Temperature')

    # Sort the columns so time-of-day is in ascending order (00:00:00 < 00:10:00 < ...)
    pivot_time = pivot_time.reindex(sorted(pivot_time.columns), axis=1)
    pivot_power = pivot_power.reindex(sorted(pivot_power.columns), axis=1)
    pivot_temp = pivot_temp.reindex(sorted(pivot_temp.columns), axis=1)


    # 9) Create workday/weekend label
    workday_label = np.array([
        [1 if pd.Timestamp(date).weekday() >= 5 else 0] * pivot_power.shape[1]
        for date in pivot_power.index
    ])

    # --------------------------------------------------------
    # 8) Plot daily profiles (one line per date)
    # --------------------------------------------------------
    # Plot Zone 1 Power
    plt.figure(figsize=(10,4))
    for date_idx in pivot_power.index:
        plt.plot(pivot_power.columns, pivot_power.loc[date_idx, :], label=str(date_idx),  alpha=0.4, color="gray")
    plt.title("Daily Profile of Zone 1 Power Consumption")
    plt.xlabel("Time of Day (HH:MM:SS)")
    plt.ylabel("Scaled Power Consumption")
    # Uncomment to show legend with all dates
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Plot Temperature
    plt.figure(figsize=(10,4))
    for date_idx in pivot_temp.index:
        plt.plot(pivot_temp.columns, pivot_temp.loc[date_idx, :], label=str(date_idx),  alpha=0.4, color="green")
    plt.title("Daily Profile of Temperature")
    plt.xlabel("Time of Day (HH:MM:SS)")
    plt.ylabel("Scaled Temperature")
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()


    # 10) Visualize one week of Power, Temperature, and Workday Label
    week_index = 0  # Change this to shift the week (e.g., 7 for second week)
    days_to_plot = 7
    power_week = pivot_power.iloc[week_index:week_index+days_to_plot, :].to_numpy().flatten()
    temp_week = pivot_temp.iloc[week_index:week_index+days_to_plot, :].to_numpy().flatten()
    label_week = workday_label[week_index:week_index+days_to_plot, :].flatten()

    time_axis = np.arange(len(power_week))  # X-axis for plotting
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, power_week, label='Power', linewidth=1)
    plt.plot(time_axis, temp_week, label='Temperature', linewidth=1)
    plt.plot(time_axis, label_week, label='Workday Label', linewidth=2, linestyle='--')
    plt.title("One Week of Power, Temperature, and Workday Labels")
    plt.xlabel("10-minute Intervals over 7 Days")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig("results/one_week_data.pdf")
    plt.show()

    return np.array(pivot_time), np.array(pivot_power), np.array(pivot_temp)




def get_data_power_consumption_weekly():
    """
    Weekly load-temperature slices (Zone-1 household data)
    ------------------------------------------------------
    Returns
    -------
    times        : ndarray[object]  – n_weeks, each element len = points_per_day*7
    energy       : ndarray[float]   – n_weeks × (points_per_day*7)
    temp         : ndarray[float]   – idem
    workday      : ndarray[int]     – idem (0 weekday, 1 weekend)
    season_feat  : ndarray[int]     – idem (0-winter … 3-autumn)
    """
    csv_path = Path("data/Consumption/powerconsumption.csv")

    # ── 1.  Read & basic cleaning ──────────────────────────────────────────
    df = pd.read_csv(csv_path)
    # column names vary slightly across versions → be defensive
    time_col = next(c for c in df.columns if c.lower().startswith(("date time", "datetime")))
    temp_col = next(c for c in df.columns if "temp"  in c.lower())
    power_col= next(c for c in df.columns if "zone1" in c.lower())

    df["time"]            = pd.to_datetime(df[time_col])
    df["air_temperature"] = pd.to_numeric(df[temp_col],   errors="coerce")
    df["meter_reading"]   = pd.to_numeric(df[power_col],  errors="coerce")
    df = df[["time", "air_temperature", "meter_reading"]].dropna()
    df = df[::6]
    # print(df)
    df.sort_values("time", inplace=True)

    for c in ["air_temperature", "meter_reading"]:
        col_min, col_max = df[c].min(), df[c].max()
        df[c] = (df[c] - col_min) / (col_max - col_min)

    # ── 2.  Identify full days & points-per-day ────────────────────────────
    df["date"] = df["time"].dt.date
    day_counts = df.groupby("date").size()
    points_per_day = int(day_counts.mode().iloc[0])          # most common daily length

    full_dates = day_counts[day_counts == points_per_day].index
    df = df[df["date"].isin(full_dates)].copy()

    # ── 3.  Season & weekday helpers ───────────────────────────────────────
    def get_season(month):
        return {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}[month]

    # ── 4.  Daily arrays (guaranteed length = points_per_day) ──────────────
    meas_cols = ["meter_reading", "air_temperature", "date", "time"]
    grouped   = df.groupby("date")[meas_cols]

    array_3d, labels_3d, seasons_3d = [], [], []
    for d in sorted(grouped.groups.keys()):
        g = grouped.get_group(d).sort_values("time")
        # (No need to re-index; we already filtered to full days.)
        arr = g[meas_cols].values
        w_label = 0 if g["time"].dt.dayofweek.iloc[0] < 5 else 1
        season  = get_season(g["time"].iloc[0].month)

        array_3d.append(arr)
        labels_3d.append(np.full(points_per_day, w_label))
        seasons_3d.append(np.full(points_per_day, season))

    # ── 5.  Pack into complete weeks (7 consecutive full days) ─────────────
    n_full_weeks = len(array_3d) // 7
    if n_full_weeks == 0:
        return (np.array([]),) * 5

    sigma = max(1, points_per_day // 24)   # ≈ 1-hour smoothing
    energy, temp, times, workday, season_feat = [], [], [], [], []

    for w in range(n_full_weeks):

        wk = slice(w*7, (w+1)*7)
        week_d, w_lbls, w_seas = array_3d[wk], labels_3d[wk], seasons_3d[wk]

        e = np.asarray(np.concatenate([d[:, 0] for d in week_d]), dtype=float)
        t = np.asarray(np.concatenate([d[:, 1] for d in week_d]), dtype=float)
        ts = np.concatenate([d[:,3] for d in week_d])
        wl = np.concatenate(w_lbls)
        sl = np.concatenate(w_seas)

        energy.append(gaussian_filter1d(e, sigma=sigma))
        temp.append(gaussian_filter1d(t, sigma=sigma))
        times.append(ts)
        workday.append(wl)
        season_feat.append(sl)

    # plot_data(times, energy, temp, workday, season_feat)

    # --- Visualization Section ---
    # plot_week = 5
    # if len(energy) > 0:
    #     fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    #     fig.suptitle(f"Weekly Load and Temperature (Week {plot_week+1})", fontsize=13, fontweight='bold')
    #
    #     # Left subplot: Load
    #     axes[0].plot(times[plot_week], energy[plot_week], color='tab:blue')
    #     axes[0].set_title('Weekly Load', fontsize=12)
    #     axes[0].set_xlabel('Time')
    #     axes[0].set_ylabel('Load (kWh)')
    #     axes[0].tick_params(axis='x', rotation=30)
    #
    #     # Right subplot: Temperature
    #     axes[1].plot(times[plot_week], temp[plot_week], color='tab:orange')
    #     axes[1].set_title('Weekly Temperature', fontsize=12)
    #     axes[1].set_xlabel('Time')
    #     axes[1].set_ylabel('Temperature (°C)')
    #     axes[1].tick_params(axis='x', rotation=30)
    #
    #     plt.tight_layout()
    #     plt.show()

    return (np.array(times, dtype=object),
            np.array(energy),
            np.array(temp),
            np.array(workday),
            np.array(season_feat))



def get_data_kaggle_2():
    """
    https://www.kaggle.com/datasets/srinuti/residential-power-usage-3years-data-timeseries
    Loads the 'power_usage_2016_to_2020.csv' and 'weather_2016_2020_daily.csv' datasets,
    merges them by date, creates daily profiles, and plots a single week of data
    (Power, Temperature, Workday Label) in a flattened time series.
    """
    load_file = "data/Residential/power_usage_2016_to_2020.csv"
    df_load = pd.read_csv(load_file)
    df_load['DateTime'] = pd.to_datetime(df_load['StartDate'])
    df_load['Date'] = df_load['DateTime'].dt.date
    df_load.rename(columns={'Value (kWh)': 'Power'}, inplace=True)

    weather_file = "data/Residential/weather_2016_2020_daily.csv"
    df_weather = pd.read_csv(weather_file)

    df_weather['Date'] = pd.to_datetime(df_weather['Date']).dt.date
    df_weather.rename(columns={'Temp_avg': 'Temperature'}, inplace=True)
    df_weather = df_weather[['Date', 'Temperature']]


    df_merged = pd.merge(df_load, df_weather, on='Date', how='left')

    df_merged.sort_values(by='DateTime', inplace=True)
    df_merged.dropna(subset=['Power', 'Temperature'], inplace=True)

    scaler = MinMaxScaler()
    df_merged[['Power', 'Temperature']] = scaler.fit_transform(df_merged[['Power', 'Temperature']])
    df_merged['TimeOfDay'] = df_merged['DateTime'].dt.strftime('%H:%M:%S')

    pivot_power = df_merged.pivot(index='Date', columns='TimeOfDay', values='Power')
    pivot_temp = df_merged.pivot(index='Date', columns='TimeOfDay', values='Temperature')
    pivot_time = df_merged.pivot(index='Date', columns='TimeOfDay', values='DateTime')

    # Sort columns so time-of-day is in ascending order
    pivot_power = pivot_power.reindex(sorted(pivot_power.columns), axis=1)
    pivot_temp = pivot_temp.reindex(sorted(pivot_temp.columns), axis=1)
    pivot_time = pivot_time.reindex(sorted(pivot_time.columns), axis=1)


    pivot_dates = pivot_power.index  # these are datetime.date objects

    df_day = df_load.groupby('Date')['day_of_week'].first().reindex(pivot_dates)
    weekend_indicator = df_day.isin([5, 6]).astype(int).values  # 1 if day_of_week in [6,7], else 0

    workday_label_2D = np.array([
        [weekend_indicator[i]] * pivot_power.shape[1]
        for i in range(len(pivot_dates))
    ])
    print(workday_label_2D)
    plt.figure(figsize=(10, 4))
    for date_idx in pivot_power.index:
        plt.plot(
            pivot_power.columns,
            pivot_power.loc[date_idx, :],
            label=str(date_idx), alpha=0.4, color="gray"
        )
    plt.title("Daily Profile of Power")
    plt.xlabel("Time of Day")
    plt.ylabel("Scaled Power")
    plt.tight_layout()
    plt.show()

    # 7b) Plot daily temperature profiles
    plt.figure(figsize=(10, 4))
    for date_idx in pivot_temp.index:
        plt.plot(
            pivot_temp.columns,
            pivot_temp.loc[date_idx, :],
            label=str(date_idx), alpha=0.4, color="blue"
        )
    plt.title("Daily Profile of Temperature")
    plt.xlabel("Time of Day")
    plt.ylabel("Scaled Temperature")
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------
    # 8) Select ONE WEEK of data and flatten it into a single time-series
    # ------------------------------------------------
    # Let's say we pick the first 7 days in the pivot:
    week_index = 10  # which chunk of 7 days to pick
    days_to_plot = 7
    chosen_dates = pivot_power.index[week_index:week_index + days_to_plot]

    power_week = pivot_power.loc[chosen_dates, :].to_numpy().flatten()
    temp_week = pivot_temp.loc[chosen_dates, :].to_numpy().flatten()
    label_week = workday_label_2D[week_index:week_index + days_to_plot, :].flatten()

    # The X-axis will be one point per hour (or half-hour, etc.) times 7 days
    time_axis = np.arange(len(power_week))

    # ------------------------------------------------
    # 9) Plot one-week time series of Power, Temperature, Workday
    # ------------------------------------------------
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, power_week, label='Power', linewidth=1)
    plt.plot(time_axis, temp_week, label='Temperature', linewidth=1)
    plt.plot(time_axis, label_week, label='Workday Label',
             linewidth=2, linestyle='--')

    print(list(power_week))

    plt.title("One Week of Power, Temperature, and Workday Labels")
    plt.xlabel("Hourly Points over 7 Days")
    plt.ylabel("Scaled Value / Label")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return pivot_power, pivot_temp, workday_label_2D



def get_data_residential_weekly():
    """
    Residential power-usage data (2016-2020) → weekly slices.

    Returns
    -------
    times        : np.ndarray (dtype=object)  – shape (n_weeks,)
                   each element is a 1-D array of datetime stamps
    energy       : np.ndarray, shape (n_weeks, points_per_day*7)
    temp         : np.ndarray, same shape
    workday      : np.ndarray, same shape, int {0,1}
    season_feat  : np.ndarray, same shape, int {0,1,2,3}
    """

    # ── paths ──────────────────────────────────────────────────────────────
    p_load    = Path("data/Residential/power_usage_2016_to_2020.csv")
    p_weather = Path("data/Residential/weather_2016_2020_daily.csv")

    # ── 1. read & basic merge  (load = hourly, weather = daily) ───────────
    df_load             = pd.read_csv(p_load)
    df_load["time"]     = pd.to_datetime(df_load["StartDate"])
    df_load["date"]     = df_load["time"].dt.date
    df_load.rename(columns={"Value (kWh)": "meter_reading"}, inplace=True)

    df_weather          = pd.read_csv(p_weather)
    df_weather["date"]  = pd.to_datetime(df_weather["Date"]).dt.date
    df_weather.rename(columns={"Temp_avg": "air_temperature"}, inplace=True)

    df = pd.merge(df_load[["time", "date", "meter_reading", "day_of_week"]],
                  df_weather[["date", "air_temperature"]],
                  on="date", how="left")

    # ── 2. keep numeric & drop NaN ─────────────────────────────────────────
    df["meter_reading"]   = pd.to_numeric(df["meter_reading"],   errors="coerce")
    df["air_temperature"] = pd.to_numeric(df["air_temperature"], errors="coerce")
    df.dropna(subset=["meter_reading", "air_temperature"], inplace=True)
    df.sort_values("time", inplace=True)

    # min-max normalise both variables globally
    for c in ["meter_reading", "air_temperature"]:
        v_min, v_max = df[c].min(), df[c].max()
        df[c] = (df[c] - v_min) / (v_max - v_min)

    # ── 3. ensure full-day rows & discover points_per_day ─────────────────
    day_counts     = df.groupby("date").size()
    points_per_day = int(day_counts.mode().iloc[0])      # most common length
    full_dates     = day_counts[day_counts == points_per_day].index
    df             = df[df["date"].isin(full_dates)].copy()


    # ── 4. helpers ─────────────────────────────────────────────────────────
    def get_season(m):    # 0=winter … 3=autumn
        return {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}[m]

    meas_cols = ["meter_reading", "air_temperature", "date", "time"]
    grouped   = df.groupby("date")[meas_cols]

    # ── 5. daily arrays (guaranteed identical length) ─────────────────────
    daily, d_labels, d_seasons = [], [], []
    for d in sorted(grouped.groups.keys()):
        g  = grouped.get_group(d).sort_values("time")
        arr = g[meas_cols].values
        weekend = 1 if g["time"].dt.dayofweek.iloc[0] >= 5 else 0
        season  = get_season(g["time"].iloc[0].month)

        daily.append(arr)
        d_labels.append(np.full(points_per_day, weekend,  dtype=int))
        d_seasons.append(np.full(points_per_day, season, dtype=int))

    # ── 6. build consecutive 7-day blocks starting at 00:00 ───────────────
    n_full_weeks = len(daily) // 7
    if n_full_weeks == 0:
        return (np.array([]),) * 5

    # sigma = max(1, points_per_day // 24)      # ≈ 1-hour smoothing
    energy, temp, times, workday, season_feat = [], [], [], [], []

    for w in range(n_full_weeks):
        sl = slice(w*7, (w+1)*7)
        week_d, w_lbl, w_sea = daily[sl], d_labels[sl], d_seasons[sl]

        e  = np.asarray(np.concatenate([d[:,0] for d in week_d]), dtype=float)
        t  = np.asarray(np.concatenate([d[:,1] for d in week_d]), dtype=float)
        ts = np.concatenate([d[:,3] for d in week_d])
        wl = np.concatenate(w_lbl)
        sf = np.concatenate(w_sea)

        energy.append(gaussian_filter1d(e, sigma=1))
        temp.append(gaussian_filter1d(t, sigma=1))
        times.append(ts)
        workday.append(wl)
        season_feat.append(sf)

    # plot_data(times, energy, temp, workday, season_feat)

    return (np.array(times, dtype=object),
            np.array(energy),
            np.array(temp),
            np.array(workday),
            np.array(season_feat))



def get_data_solar_weather_weekly():
    """
    Returns
    -------
    times        : np.ndarray  (dtype=object)  shape (n_weeks,)
    energy       : np.ndarray  shape (n_weeks, points_per_day*7)
    temp         : np.ndarray  same shape
    workday      : np.ndarray  same shape, int {0,1}
    season_feat  : np.ndarray  same shape, int {0,1,2,3}
    """

    # ── 1.  read & basic cleaning ─────────────────────────────────────────
    p_csv = Path("data/Solar/solar_weather.csv")
    df    = pd.read_csv(p_csv, parse_dates=["Time"])

    # you sampled 1000:10000 in the draft – keep that if desired
    # df = df.iloc[1000:10000].copy()
    df = df.iloc[::4].copy()

    # keep two numeric columns & drop NaN
    df = df[["Time", "Energy delta[Wh]", "temp"]].rename(
            columns={"Energy delta[Wh]": "meter_reading",
                     "temp":            "air_temperature"})
    df["meter_reading"]   = pd.to_numeric(df["meter_reading"],   errors="coerce")
    df["air_temperature"] = pd.to_numeric(df["air_temperature"], errors="coerce")
    df.dropna(inplace=True)
    df.sort_values("Time", inplace=True)
    # print(df)

    # ── 2.  global min-max normalisation ─────────────────────────────────
    for c in ["meter_reading", "air_temperature"]:
        vmin, vmax = df[c].min(), df[c].max()
        df[c] = (df[c] - vmin) / (vmax - vmin)

    # ── 3.  identify full days / sample rate ─────────────────────────────
    df["date"]  = df["Time"].dt.date
    day_counts  = df.groupby("date").size()
    pts_per_day = int(day_counts.mode().iloc[0])        # modal length
    full_dates  = day_counts[day_counts == pts_per_day].index
    df          = df[df["date"].isin(full_dates)].copy()

    # ── 4.  helpers ──────────────────────────────────────────────────────
    def get_season(m):        # 0=winter,1=spring,2=summer,3=autumn
        return {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}[m]

    meas_cols = ["meter_reading", "air_temperature", "date", "Time"]
    grouped   = df.groupby("date")[meas_cols]

    daily, d_wd, d_sea = [], [], []
    for d in sorted(grouped.groups.keys()):
        g  = grouped.get_group(d).sort_values("Time")
        arr = g[meas_cols].values

        wd_flag = 1 if g["Time"].dt.dayofweek.iloc[0] >= 5 else 0
        season  = get_season(g["Time"].iloc[0].month)

        daily.append(arr)
        d_wd.append(np.full(pts_per_day, wd_flag, dtype=int))
        d_sea.append(np.full(pts_per_day, season,  dtype=int))

    # ── 5.  consecutive 7-day blocks, starting at 00:00 ──────────────────
    n_full_weeks = len(daily) // 7
    if n_full_weeks == 0:
        return (np.array([]),)*5

    sigma = max(1, pts_per_day // 24)       # ≈ one-hour smoothing
    energy, temp, times, workday, season_feat = [], [], [], [], []

    for w in range(n_full_weeks):
        sl = slice(w*7, (w+1)*7)
        wk_d, wk_wd, wk_sea = daily[sl], d_wd[sl], d_sea[sl]

        e  = np.asarray(np.concatenate([d[:,0] for d in wk_d]), dtype=float)
        t  = np.asarray(np.concatenate([d[:,1] for d in wk_d]), dtype=float)
        ts = np.concatenate([d[:,3] for d in wk_d])
        wl = np.concatenate(wk_wd)
        sf = np.concatenate(wk_sea)

        energy.append(gaussian_filter1d(e, sigma=sigma))
        temp.append(gaussian_filter1d(t, sigma=sigma))
        times.append(ts)
        workday.append(wl)
        season_feat.append(sf)

    # plot_data(times, energy, temp, workday, season_feat)

    return (np.array(times,   dtype=object),
            np.array(energy),
            np.array(temp),
            np.array(workday),
            np.array(season_feat))


def _oncor_load_weekly_utils():
    '''
    Run once. Creates one NPZ per XFMR with:
      - tensor:        (weeks, 168, F) float
      - features:      (F,) object array of feature names
      - time_index:    (weeks, 168) datetime64[h]
      - time_index_str:(weeks, 168) str (ISO 8601)
      - week_start:    (weeks,) datetime64[h]
      - week_end:      (weeks,) datetime64[h]
      - xfmr:          () str
    '''
    csv_path = "TransformerLoadData.csv"
    out_dir  = "processed_data"
    # os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    time_col  = "DATEHRLWT"
    group_col = "XFMR"

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col)

    if group_col not in df.columns:
        df[group_col] = "ALL"

    expected_features = [
        "KWH",
        "SURFACETEMPERATUREFAHRENHEIT",
        "SURDPOINTTEMPFAHRENHEIT",
        "PREPREVHOURINCHES",
        "RELATIVEHUMIDITY",
        "WINDSPEEDMPH",
        "SURFACEWINDGUSTSMPH",
        "WINDCHILLTEMPERATUREFAHRENHEIT",
        "HEATINDEXFAHRENHEIT",
        "SNOWFALLINCHES",
    ]
    present_features = [f for f in expected_features if f in df.columns]
    if not present_features:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        present_features = [c for c in numeric_cols if c != group_col]

    def build_weekly_tensor(gdf, features, week_hours=168):
        # Keep only relevant columns
        gdf = gdf[[time_col] + features].copy().sort_values(time_col).set_index(time_col)

        # Aggregate duplicate timestamps by mean (if any)
        gdf = gdf.groupby(level=0).mean(numeric_only=True)

        # Reindex to continuous hourly timeline
        start = gdf.index.min().floor("h")
        end   = gdf.index.max().ceil("h")
        full_index = pd.date_range(start=start, end=end, freq="h")
        gdf = gdf.reindex(full_index)

        X = gdf[features].copy()
        total_hours = len(X)
        num_weeks   = total_hours // week_hours

        week_arrays   = []
        week_times    = []
        week_spans_s  = []
        week_spans_e  = []

        for w in range(num_weeks):
            sl = X.iloc[w*week_hours:(w+1)*week_hours]
            # Only keep fully observed weeks
            if not sl.isna().any().any():
                week_arrays.append(sl.to_numpy(dtype=float))
                idx = sl.index.values.astype("datetime64[h]")  # (168,)
                week_times.append(idx)
                week_spans_s.append(idx[0])
                week_spans_e.append(idx[-1])

        if week_arrays:
            arr          = np.stack(week_arrays, axis=0)               # (W,168,F)
            time_index   = np.stack(week_times, axis=0)               # (W,168) datetime64[h]
            week_start   = np.array(week_spans_s, dtype="datetime64[h]")
            week_end     = np.array(week_spans_e, dtype="datetime64[h]")
        else:
            F = len(features)
            arr        = np.empty((0, week_hours, F), dtype=float)
            time_index = np.empty((0, week_hours), dtype="datetime64[h]")
            week_start = np.empty((0,), dtype="datetime64[h]")
            week_end   = np.empty((0,), dtype="datetime64[h]")

        return arr, time_index, week_start, week_end, num_weeks, total_hours

    out_paths = []
    for xfmr, g in df.groupby(group_col):
        tensor, time_index, week_start, week_end, possible_weeks, total_hours = build_weekly_tensor(
            g, present_features, week_hours=168
        )

        safe_xfmr = str(xfmr).replace("/", "_").replace("\\", "_").replace(" ", "_")
        out_file  = os.path.join(out_dir, f"weeks_tensor_{safe_xfmr}.npz")

        # Also keep ISO strings for portability
        time_index_str = np.array([[str(t) for t in row] for row in time_index], dtype=object)

        np.savez_compressed(
            out_file,
            tensor=tensor,
            features=np.array(present_features, dtype=object),
            time_index=time_index,
            time_index_str=time_index_str,
            week_start=week_start,
            week_end=week_end,
            xfmr=np.array(safe_xfmr, dtype=object),
        )
        out_paths.append(out_file)

    print("Saved files:")
    for p in out_paths:
        print(p)

def combine_oncor_transformers(transformer_ids, out_path="processed_data/weeks_tensor_all.npz"):
    """Combine weekly data from multiple transformers into one .npz file."""
    tensors = []
    time_idx = []
    time_idx_str = []
    common_features = None
    for tid in transformer_ids:
        data = np.load(f"processed_data/weeks_tensor_{tid}.npz", allow_pickle=True)
        tensors.append(data["tensor"])                     # weekly data array
        time_idx.append(data["time_index"])                # weekly time indices
        time_idx_str.append(data["time_index_str"])        # string time indices (if any)
        if common_features is None:
            common_features = data["features"]             # feature list (assume identical for all)
    # Concatenate all transformers' data along the week axis:
    tensor_all = np.concatenate(tensors, axis=0)
    time_index_all = np.concatenate(time_idx, axis=0)
    time_index_str_all = np.concatenate(time_idx_str, axis=0)
    # Save the combined dataset
    np.savez(out_path, tensor=tensor_all, features=common_features, 
             time_index=time_index_all, time_index_str=time_index_str_all)
    print(f"Combined data saved to {out_path}")

def get_data_oncor_load_weekly(XFMR="all"):
    """
    Load weekly ONCOR data for specified transformer ID.
    Parameters
    ----------
    XFMR : str
        Transformer ID string (e.g., "377136683"), or "all" for combined.
    """
    if XFMR == "all":
        fpath = "processed_data/weeks_tensor_all.npz"
    else:
        fpath = f"processed_data/weeks_tensor_{XFMR}.npz"
    
    data = np.load(fpath, allow_pickle=True)
    X = data["tensor"]
    features = data["features"].tolist()
    time_index = data["time_index"]
    time_index_str = data["time_index_str"]
    
    load     = X[:, :, features.index("KWH")]
    temp     = X[:, :, features.index("SURDPOINTTEMPFAHRENHEIT")]
    workday  = X[:, :, features.index("RELATIVEHUMIDITY")]
    season   = X[:, :, features.index("HEATINDEXFAHRENHEIT")]
    
    return time_index, load, temp, workday, season




#
#### Etth 1,2 ########################################################################
# ---------- ETTh1 helpers ----------
def _load_etth_dataframe(path_csv: str) -> pd.DataFrame:
    """
    Load ETTh1 CSV (expects columns like: date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT).
    Ensures an hourly-regular time index with missing timestamps filled and values interpolated.
    """
    if not os.path.exists(path_csv):
        raise FileNotFoundError(f"ETTh1 file not found at: {path_csv}")

    df = pd.read_csv(path_csv)
    # Standard ETTh1 uses 'date' as timestamp column
    if "date" not in df.columns:
        # try common alternatives
        for cand in ["time", "timestamp", "Date", "ds"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "date"})
                break
        if "date" not in df.columns:
            raise ValueError(f"Could not find a timestamp column in {path_csv} (looked for 'date').")

    # Parse datetime and set as index
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["date"] = df["date"].dt.tz_convert(None)  # make naive
    df = df.set_index("date").sort_index()

    # Build a complete hourly index and reindex
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")
    df = df.reindex(full_idx)

    # Interpolate numeric columns and fill edges
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].interpolate("time").ffill().bfill()

    df.index.name = "date"
    return df
#

def get_data_etth1_weekly(path_csv="data/datasets/ETTh1.csv"):
    """
    Slice ETTh1 into consecutive full weeks (7*24 = 168 hours) and return all features.

    Returns
    -------
    times       : np.ndarray (dtype=object) shape (n_weeks,)        # each entry: np.array of 168 timestamps
    HUFL        : np.ndarray              shape (n_weeks, 168)
    HULL        : np.ndarray              shape (n_weeks, 168)
    MUFL        : np.ndarray              shape (n_weeks, 168)
    MULL        : np.ndarray              shape (n_weeks, 168)
    LUFL        : np.ndarray              shape (n_weeks, 168)
    LULL        : np.ndarray              shape (n_weeks, 168)
    OT          : np.ndarray              shape (n_weeks, 168)
    workday     : np.ndarray              shape (n_weeks, 168)  # int {0,1}; 1 = weekend (follows your example)
    season_feat : np.ndarray              shape (n_weeks, 168)  # int {0=winter,1=spring,2=summer,3=autumn}
    """
    df = _load_etth_dataframe(path_csv)

    feature_cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in ETTh1. Found: {list(df.columns)}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # keep only needed columns
    df = df[feature_cols].copy()

    # ensure complete hourly coverage already handled by _load_etth1_dataframe
    # build per-day aligned blocks with exactly 24 hours each
    by_date = df.groupby(df.index.date)

    def _season_from_month(m):  # 0=winter,1=spring,2=summer,3=autumn
        return {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}[m]

    daily_feats = []   # list of DataFrames (24 x 7)
    daily_times = []   # list of DatetimeIndex (24)
    daily_wd    = []   # list of np.ndarray (24,) int
    daily_sea   = []   # list of np.ndarray (24,) int

    for day_key in sorted(by_date.groups.keys()):
        # build exact hourly index for the calendar day
        day_start = pd.Timestamp(day_key)
        hourly = pd.date_range(start=day_start, periods=24, freq="h")
        # slice and reindex to this day
        g = by_date.get_group(day_key).reindex(hourly)
        # fill/interpolate numeric (features)
        g[feature_cols] = g[feature_cols].interpolate("time").ffill().bfill()

        if len(g) != 24 or g[feature_cols].isna().any().any():
            # skip incomplete/ill-formed days
            continue

        daily_feats.append(g[feature_cols])
        daily_times.append(hourly)

        # workday flag (following your solar_weather example: 1 = weekend, 0 = weekday)
        # change to (1 if dayofweek < 5 else 0) if you want 1 = weekday
        wd_flag = 1 if hourly[0].dayofweek >= 5 else 0
        daily_wd.append(np.full(24, wd_flag, dtype=int))

        season = _season_from_month(hourly[0].month)
        daily_sea.append(np.full(24, season, dtype=int))

    if not daily_feats:
        # nothing usable
        return (np.array([]),)*10

    # pack consecutive 7-day blocks
    n_full_weeks = len(daily_feats) // 7
    if n_full_weeks == 0:
        return (np.array([]),)*10

    HUFL, HULL, MUFL, MULL, LUFL, LULL, OT = [], [], [], [], [], [], []
    workday, season_feat, times = [], [], []

    for w in range(n_full_weeks):
        sl = slice(w*7, (w+1)*7)

        week_df_list = daily_feats[sl]
        week_time    = np.concatenate(daily_times[sl])
        week_wd      = np.concatenate(daily_wd[sl])
        week_sea     = np.concatenate(daily_sea[sl])

        # concatenate each feature across the 7 days (24*7 = 168)
        HUFL.append(np.concatenate([d["HUFL"].values for d in week_df_list]).astype(float))
        HULL.append(np.concatenate([d["HULL"].values for d in week_df_list]).astype(float))
        MUFL.append(np.concatenate([d["MUFL"].values for d in week_df_list]).astype(float))
        MULL.append(np.concatenate([d["MULL"].values for d in week_df_list]).astype(float))
        LUFL.append(np.concatenate([d["LUFL"].values for d in week_df_list]).astype(float))
        LULL.append(np.concatenate([d["LULL"].values for d in week_df_list]).astype(float))
        OT.append(  np.concatenate([d["OT"].values   for d in week_df_list]).astype(float))

        workday.append(week_wd)
        season_feat.append(week_sea)
        times.append(week_time)

    return (np.array(times, dtype=object),
            np.array(HUFL),
            np.array(HULL),
            np.array(MUFL),
            np.array(MULL),
            np.array(LUFL),
            np.array(LULL),
            np.array(OT),
            np.array(workday),
            np.array(season_feat))


def get_data_etth2_weekly(path_csv="data/datasets/ETTh2.csv"):
    """
    Slice ETTh2 into consecutive full weeks (7*24 = 168 hours) and return all features.

    Returns
    -------
    times       : np.ndarray (dtype=object) shape (n_weeks,)        # each entry: np.array of 168 timestamps
    HUFL        : np.ndarray              shape (n_weeks, 168)
    HULL        : np.ndarray              shape (n_weeks, 168)
    MUFL        : np.ndarray              shape (n_weeks, 168)
    MULL        : np.ndarray              shape (n_weeks, 168)
    LUFL        : np.ndarray              shape (n_weeks, 168)
    LULL        : np.ndarray              shape (n_weeks, 168)
    OT          : np.ndarray              shape (n_weeks, 168)
    workday     : np.ndarray              shape (n_weeks, 168)  # int {0,1}; 1 = weekend (follows your example)
    season_feat : np.ndarray              shape (n_weeks, 168)  # int {0=winter,1=spring,2=summer,3=autumn}
    """
    df = _load_etth_dataframe(path_csv)

    feature_cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in ETTh1. Found: {list(df.columns)}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # keep only needed columns
    df = df[feature_cols].copy()

    # ensure complete hourly coverage already handled by _load_etth1_dataframe
    # build per-day aligned blocks with exactly 24 hours each
    by_date = df.groupby(df.index.date)

    def _season_from_month(m):  # 0=winter,1=spring,2=summer,3=autumn
        return {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}[m]

    daily_feats = []   # list of DataFrames (24 x 7)
    daily_times = []   # list of DatetimeIndex (24)
    daily_wd    = []   # list of np.ndarray (24,) int
    daily_sea   = []   # list of np.ndarray (24,) int

    for day_key in sorted(by_date.groups.keys()):
        # build exact hourly index for the calendar day
        day_start = pd.Timestamp(day_key)
        hourly = pd.date_range(start=day_start, periods=24, freq="h")
        # slice and reindex to this day
        g = by_date.get_group(day_key).reindex(hourly)
        # fill/interpolate numeric (features)
        g[feature_cols] = g[feature_cols].interpolate("time").ffill().bfill()

        if len(g) != 24 or g[feature_cols].isna().any().any():
            # skip incomplete/ill-formed days
            continue

        daily_feats.append(g[feature_cols])
        daily_times.append(hourly)

        # workday flag (following your solar_weather example: 1 = weekend, 0 = weekday)
        # change to (1 if dayofweek < 5 else 0) if you want 1 = weekday
        wd_flag = 1 if hourly[0].dayofweek >= 5 else 0
        daily_wd.append(np.full(24, wd_flag, dtype=int))

        season = _season_from_month(hourly[0].month)
        daily_sea.append(np.full(24, season, dtype=int))

    if not daily_feats:
        # nothing usable
        return (np.array([]),)*10

    # pack consecutive 7-day blocks
    n_full_weeks = len(daily_feats) // 7
    if n_full_weeks == 0:
        return (np.array([]),)*10

    HUFL, HULL, MUFL, MULL, LUFL, LULL, OT = [], [], [], [], [], [], []
    workday, season_feat, times = [], [], []

    for w in range(n_full_weeks):
        sl = slice(w*7, (w+1)*7)

        week_df_list = daily_feats[sl]
        week_time    = np.concatenate(daily_times[sl])
        week_wd      = np.concatenate(daily_wd[sl])
        week_sea     = np.concatenate(daily_sea[sl])

        # concatenate each feature across the 7 days (24*7 = 168)
        HUFL.append(np.concatenate([d["HUFL"].values for d in week_df_list]).astype(float))
        HULL.append(np.concatenate([d["HULL"].values for d in week_df_list]).astype(float))
        MUFL.append(np.concatenate([d["MUFL"].values for d in week_df_list]).astype(float))
        MULL.append(np.concatenate([d["MULL"].values for d in week_df_list]).astype(float))
        LUFL.append(np.concatenate([d["LUFL"].values for d in week_df_list]).astype(float))
        LULL.append(np.concatenate([d["LULL"].values for d in week_df_list]).astype(float))
        OT.append(  np.concatenate([d["OT"].values   for d in week_df_list]).astype(float))

        workday.append(week_wd)
        season_feat.append(week_sea)
        times.append(week_time)

    return (np.array(times, dtype=object),
            np.array(HUFL),
            np.array(HULL),
            np.array(MUFL),
            np.array(MULL),
            np.array(LUFL),
            np.array(LULL),
            np.array(OT),
            np.array(workday),
            np.array(season_feat))


#
# def _parse_compact_timestamp(s: str):
#     if not isinstance(s, str):
#         return pd.NaT
#     s = s.strip()
#     t = pd.to_datetime(s, errors="coerce", utc=False)
#     if pd.notna(t):
#         return t
#     m = re.match(r"^\s*(\d{1,2})(\d{1,2})(\d{4})\s+(\d{1,2}):(\d{2})\s*$", s)
#     if m:
#         mm, dd, yyyy, HH, MM = m.groups()
#         try:
#             return pd.Timestamp(year=int(yyyy), month=int(mm), day=int(dd), hour=int(HH), minute=int(MM))
#         except Exception:
#             return pd.NaT
#     return pd.NaT
#
# def safe_iso(x):
#     try:
#         return pd.Timestamp(x).isoformat()
#     except Exception:
#         return str(x)
#
# def get_data_GEFCom2014(timestamp_col: str = "TIMESTAMP"):
#     path_csv = "/Users/muhaoguo/Documents/study/Paper_Projects/PESGM/data/GEFCom2014 Data/GEFCom2014-L_V2/Load/Task 2/L2-train.csv"
#     df = pd.read_csv(path_csv)
#     parsed = df[timestamp_col].apply(_parse_compact_timestamp)
#     df = df.loc[~parsed.isna()].copy()
#     df.insert(0, "date", parsed.values)
#     df["date"] = pd.to_datetime(df["date"], errors="coerce")
#     df = df.dropna(subset=["date"])
#
#     num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
#     ordered = []
#     if "ZONEID" in num_cols_all:
#         ordered.append("ZONEID")
#     if "LOAD" in num_cols_all:
#         ordered.append("LOAD")
#     wnames = sorted([c for c in num_cols_all if re.fullmatch(r"w\d+", c)], key=lambda s: int(s[1:]))
#     ordered.extend(wnames)
#     remaining = [c for c in num_cols_all if c not in set(ordered)]
#     ordered.extend(remaining)
#     num_cols = ordered
#
#     df = df.drop(columns=["TIMESTAMP"])
#     df = df.groupby("date", as_index=True)[num_cols].mean().sort_index()
#
#     full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
#     df = df.reindex(full_idx)
#     df[num_cols] = df[num_cols].interpolate("time").ffill().bfill()
#     df.index.name = "date"
#
#     by_date = df.groupby(df.index.date)
#
#     day_feats, day_times, day_wd, day_sea = [], [], [], []
#     def _season_from_month(m: int) -> int:
#         return {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}[m]
#
#     for day in sorted(by_date.groups.keys()):
#         start = pd.Timestamp(day)
#         hidx = pd.date_range(start, periods=24, freq="h")
#         g = by_date.get_group(day).reindex(hidx)
#         g[num_cols] = g[num_cols].interpolate("time").ffill().bfill()
#         if len(g) != 24 or g[num_cols].isna().any().any():
#             continue
#         day_feats.append(g[num_cols])
#         day_times.append(hidx)
#         wd = 1 if hidx[0].dayofweek >= 5 else 0
#         day_wd.append(np.full(24, wd, dtype=int))
#         sea = _season_from_month(hidx[0].month)
#         day_sea.append(np.full(24, sea, dtype=int))
#
#     n_full_weeks = len(day_feats) // 7
#     times, workday, season_feat = [], [], []
#     feat_arrays_by_name = {c: [] for c in num_cols}
#     for w in range(n_full_weeks):
#         sl = slice(7*w, 7*(w+1))
#         week_list = day_feats[sl]
#         times.append(np.concatenate(day_times[sl]))
#         workday.append(np.concatenate(day_wd[sl]))
#         season_feat.append(np.concatenate(day_sea[sl]))
#         for c in num_cols:
#             feat_arrays_by_name[c].append(np.concatenate([d[c].values for d in week_list]).astype(float))
#
#     times = np.array(times, dtype=object)
#     workday = np.array(workday, dtype=int)
#     season_feat = np.array(season_feat, dtype=int)
#     feat_arrays_by_name = {k: np.array(v) for k, v in feat_arrays_by_name.items()}
#
#     # outputs
#     w_names = [f"w{i}" for i in range(1, 26) if f"w{i}" in feat_arrays_by_name]
#     feature_arrays = [feat_arrays_by_name[w] for w in w_names]
#     feature_names = w_names
#     load = feat_arrays_by_name["LOAD"]
#
#     return times, feature_arrays, workday, season_feat, load, feature_names



# ---------- helpers (reuse from your code) ----------
def _parse_compact_timestamp(s: str):
    if not isinstance(s, str):
        return pd.NaT
    s = s.strip()
    t = pd.to_datetime(s, errors="coerce", utc=False)
    if pd.notna(t):
        return t
    m = re.match(r"^\s*(\d{1,2})(\d{1,2})(\d{4})\s+(\d{1,2}):(\d{2})\s*$", s)
    if m:
        mm, dd, yyyy, HH, MM = m.groups()
        try:
            return pd.Timestamp(year=int(yyyy), month=int(mm), day=int(dd), hour=int(HH), minute=int(MM))
        except Exception:
            return pd.NaT
    return pd.NaT

def _season_from_month(m: int) -> int:
    return {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}[m]

def _process_single_csv_to_weeks(path_csv: str, timestamp_col: str = "TIMESTAMP"
) -> Tuple[np.ndarray, dict, np.ndarray, np.ndarray]:
    """
    Returns (times, feat_arrays_by_name, workday, season_feat) for one CSV.
    feat_arrays_by_name maps each numeric column name -> (n_weeks, 168)
    """
    df = pd.read_csv(path_csv)
    parsed = df[timestamp_col].apply(_parse_compact_timestamp)
    mask = ~parsed.isna()
    df = df.loc[mask].copy()
    parsed = parsed[mask]  # keep only the matching rows
    df.insert(0, "date", parsed.values)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).drop(columns=[timestamp_col])

    # numeric columns and ordering (LOAD + w1..w25 first)
    num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
    ordered = []
    if "ZONEID" in num_cols_all:
        ordered.append("ZONEID")
    if "LOAD" in num_cols_all:
        ordered.append("LOAD")
    wnames = sorted([c for c in num_cols_all if re.fullmatch(r"w\d+", c)], key=lambda s: int(s[1:]))
    ordered.extend(wnames)
    remaining = [c for c in num_cols_all if c not in set(ordered)]
    ordered.extend(remaining)
    num_cols = ordered

    # aggregate dups, regularize hourly
    df = df.groupby("date", as_index=True)[num_cols].mean().sort_index()
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_idx)
    # df[num_cols] = df[num_cols].interpolate("time").ffill().bfill()
    df.index.name = "date"

    # daily grouping
    by_date = df.groupby(df.index.date)
    day_feats, day_times, day_wd, day_sea = [], [], [], []

    for day in sorted(by_date.groups.keys()):
        start = pd.Timestamp(day)
        hidx = pd.date_range(start, periods=24, freq="h")
        g = by_date.get_group(day).reindex(hidx)
        g[num_cols] = g[num_cols].interpolate("time").ffill().bfill()
        if len(g) != 24 or g[num_cols].isna().any().any():
            continue
        day_feats.append(g[num_cols])
        day_times.append(hidx)
        wd = 1 if hidx[0].dayofweek >= 5 else 0  # 1=weekend
        day_wd.append(np.full(24, wd, dtype=int))
        sea = _season_from_month(hidx[0].month)
        day_sea.append(np.full(24, sea, dtype=int))

    # pack weekly
    n_full_weeks = len(day_feats) // 7
    times, workday, season_feat = [], [], []
    feat_arrays_by_name = {c: [] for c in num_cols}

    for w in range(n_full_weeks):
        sl = slice(7 * w, 7 * (w + 1))
        week_list = day_feats[sl]
        # Concatenate 7 days (should be 168 hours)
        week_df = pd.concat(week_list)

        # --- Skip incomplete or invalid weeks ---
        if len(week_df) != 168:
            continue
        if week_df.isna().any().any():
            continue
        # (optional) if LOAD present and week nearly constant, skip too
        if "LOAD" in week_df and week_df["LOAD"].std() < 1e-3:
            continue

        times.append(np.concatenate(day_times[sl]))
        workday.append(np.concatenate(day_wd[sl]))
        season_feat.append(np.concatenate(day_sea[sl]))

        for c in num_cols:
            feat_arrays_by_name[c].append(
                np.concatenate([d[c].values for d in week_list]).astype(float)
            )

    # finalize arrays
    times = np.array(times, dtype=object)
    workday = np.array(workday, dtype=int)
    season_feat = np.array(season_feat, dtype=int)
    feat_arrays_by_name = {k: np.array(v) for k, v in feat_arrays_by_name.items()}
    return times, feat_arrays_by_name, workday, season_feat

# ---------- multi-file combiner ----------
def get_data_GEFCom2014_multi(timestamp_col: str = "TIMESTAMP"):
    """
    Combine multiple GEFCom2014 L-track CSVs (e.g., L2..L15) into one dataset.
    Returns:
      times(object), feature_arrays(w1..w25), workday, season_feat, load, feature_names(w1..w25)
    """

    root = "data/GEFCom2014 Data/GEFCom2014-L_V2/Load"
    csv_paths = [f"{root}/Task {i}/L{i}-train.csv" for i in range(2, 16)]

    all_times, all_workday, all_season = [], [], []
    # We will only keep tasks that have all w1..w25 and LOAD
    required_ws = [f"w{i}" for i in range(1, 26)]
    kept_count = 0
    skipped = []

    # accumulate per-feature lists of weekly blocks
    w_buckets = {w: [] for w in required_ws}
    load_bucket = []

    for path in csv_paths:
        times, feats, wd, sea = _process_single_csv_to_weeks(path, timestamp_col=timestamp_col)

        # check required features
        if "LOAD" not in feats or any(w not in feats for w in required_ws):
            skipped.append((path, list(feats.keys())))
            continue  # skip tasks missing required features

        # append
        all_times.append(times)        # list of (n_weeks_task,) object
        all_workday.append(wd)         # (n_weeks_task, 168)
        all_season.append(sea)         # (n_weeks_task, 168)
        for w in required_ws:
            w_buckets[w].append(feats[w])   # each (n_weeks_task, 168)
        load_bucket.append(feats["LOAD"])   # (n_weeks_task, 168)
        kept_count += 1

    if kept_count == 0:
        raise ValueError("No tasks had complete features (LOAD + w1..w25). Skipped info: " + str(skipped))

    # concat across tasks along week axis
    times_out = np.concatenate(all_times, axis=0)                         # object array of weeks
    workday_out = np.concatenate(all_workday, axis=0)                     # (N,168)
    season_out = np.concatenate(all_season, axis=0)                       # (N,168)
    load_out = np.concatenate(load_bucket, axis=0)                        # (N,168)
    feature_names = required_ws
    feature_arrays = [np.concatenate(w_buckets[w], axis=0) for w in required_ws]  # each (N,168)

    # cast/pack to your signature
    return (np.array(times_out, dtype=object),
            feature_arrays,                  # list of 25 arrays, each (N,168)
            workday_out.astype(int),
            season_out.astype(int),
            load_out.astype(float),
            feature_names)




def get_flores():
    # Load the dataset
    path = "data/FloresLoad2008/Flores load 2008.csv"  # update path if needed
    df = pd.read_csv(path)

    # Convert to numeric (coerce invalid values)
    colname = df.columns[0]
    series = pd.to_numeric(df[colname], errors="coerce").interpolate().ffill().bfill()

    # Assume hourly readings starting from Jan 1, 2008, 00:00
    start = pd.Timestamp("2008-01-01 00:00:00")
    idx = pd.date_range(start=start, periods=len(series), freq="h")
    s = pd.Series(series.values, index=idx)

    # Keep only full weeks (168 hours per week)
    n_hours = len(s)
    n_weeks = n_hours // 168
    s = s.iloc[:n_weeks * 168]

    # Convert to shape (313, 168)
    load = s.values.reshape(n_weeks, 168)
    print("Weekly array shape:", load.shape)
    return load


if __name__ == "__main__":
    # _oncor_load_weekly_utils()
    # times, load, temp, workday, season = get_data_oncor_load_weekly()

    # times, energy, temp, workday, season_feat = get_data_building_weather_weekly()
    # print(times.shape, energy.shape, temp.shape, workday.shape, season_feat.shape)

    # times, energy, temp, workday, season_feat  = get_data_spanish_weekly()
    # print(times.shape, energy.shape, temp.shape, workday.shape, season_feat.shape)
    #
    # times, energy, temp, workday, season_feat  = get_data_power_consumption_weekly()
    # print(times.shape, energy.shape, temp.shape, workday.shape, season_feat.shape)
    #
    # times, energy, temp, workday, season_feat = get_data_residential_weekly()
    # print(times.shape, energy.shape, temp.shape, workday.shape, season_feat.shape)
    #
    # times, energy, temp, workday, season_feat = get_data_solar_weather_weekly()
    # print(times.shape, energy.shape, temp.shape, workday.shape, season_feat.shape)

    # times, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT, workday, season_feat = get_data_etth1_weekly()
    # plot_etth_weekly_features(times, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT, week_idx=0)

    # times, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT, workday, season_feat = get_data_etth2_weekly()
    # plot_etth_weekly_features(times, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT, week_idx=0)

    # times, feature_arrays, workday, season_feat, load, feature_names = get_data_GEFCom2014()
    # times, feature_arrays, workday, season_feat, load, feature_names = get_data_GEFCom2014()
    times, feature_arrays, workday, season_feat, load, feature_names = get_data_GEFCom2014_multi()
    # print(times, feature_arrays, workday.shape, season_feat.shape, load.shape, feature_names)
    # plot_data(times, load, load, workday, season_feat)
    # load = get_flores()





