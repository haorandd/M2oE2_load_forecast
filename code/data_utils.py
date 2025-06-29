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
        norm_features      = ('load', 'temp'),
        output_len         = 24,          # how many steps each decoder step predicts
        encoder_len_weeks  = 1,
        decoder_len_weeks  = 1,
        num_in_week        = 168,         # ← NEW: default parameter
        device             = None):

    # ----------------------------------------------------------
    # 1. flatten, scale, keep 1‑D per feature
    # ----------------------------------------------------------
    processed, scalers = {}, {}
    for k, arr in feature_dict.items():
        if arr.size == 0:
            raise ValueError(f"feature '{k}' is empty.")
        vec = arr.astype(float).flatten()          # weeks → long vector
        if k in norm_features:
            sc              = MinMaxScaler()
            processed[k]    = sc.fit_transform(vec.reshape(-1, 1)).flatten()
            scalers[k]      = sc
        else:
            processed[k]    = vec
            scalers[k]      = None

    n_weeks = feature_dict['load'].shape[0]
    need_weeks = encoder_len_weeks + decoder_len_weeks
    if n_weeks < need_weeks:
        raise ValueError(f"Need ≥{need_weeks} consecutive weeks, found {n_weeks}.")

    enc_seq_len = encoder_len_weeks * num_in_week
    dec_seq_len = decoder_len_weeks * num_in_week
    L           = dec_seq_len - output_len
    if L <= 0:
        raise ValueError("`output_len` must be smaller than decoder sequence length.")

    # ----------------------------------------------------------
    # 2. build samples (stride = 1 week)
    # ----------------------------------------------------------
    X_enc_l, X_enc_t, X_enc_w, X_enc_s = [], [], [], []
    X_dec_in_l, X_dec_in_t, X_dec_in_w, X_dec_in_s = [], [], [], []
    Y_dec_target_l = []

    last_start = n_weeks - need_weeks   # inclusive
    for w in range(last_start + 1):
        enc_start =  w * num_in_week
        enc_end   = (w + encoder_len_weeks) * num_in_week
        dec_start =  enc_end
        dec_end   =  dec_start + dec_seq_len   # exclusive

        # -- encoder --
        X_enc_l.append(processed['load'   ][enc_start:enc_end])
        X_enc_t.append(processed['temp'   ][enc_start:enc_end])
        X_enc_w.append(processed['workday'][enc_start:enc_end])
        X_enc_s.append(processed['season' ][enc_start:enc_end])

        # -- decoder input (teacher forcing) --
        X_dec_in_l.append(processed['load'   ][dec_start : dec_start + L])
        X_dec_in_t.append(processed['temp'   ][dec_start : dec_start + L])
        X_dec_in_w.append(processed['workday'][dec_start : dec_start + L])
        X_dec_in_s.append(processed['season' ][dec_start : dec_start + L])

        # -- decoder targets (sliding output_len window) --
        load_dec_full = processed['load'][dec_start: dec_end]
        targets = np.stack([
            load_dec_full[i: i + output_len] for i in range(L+1)],
            axis=0)
        Y_dec_target_l.append(targets)

    # ----------------------------------------------------------
    # 3. pack → tensors
    # ----------------------------------------------------------
    to_tensor = lambda lst: torch.tensor(lst, dtype=torch.float32).unsqueeze(-1).to(device)

    data_tensors = {
        'X_enc_l'      : to_tensor(X_enc_l),      # [B, enc_seq_len, 1]
        'X_enc_t'      : to_tensor(X_enc_t),
        'X_enc_w'      : to_tensor(X_enc_w),
        'X_enc_s'      : to_tensor(X_enc_s),

        'X_dec_in_l'   : to_tensor(X_dec_in_l),   # [B, L, 1]
        'X_dec_in_t'   : to_tensor(X_dec_in_t),
        'X_dec_in_w'   : to_tensor(X_dec_in_w),
        'X_dec_in_s'   : to_tensor(X_dec_in_s),

        'Y_dec_target_l': torch.tensor(
            Y_dec_target_l, dtype=torch.float32).unsqueeze(-1).to(device)  # [B, L, output_len, 1]
    }

    # quick check
    for k, v in data_tensors.items():
        print(f"{k:15s} {tuple(v.shape)}")

    # ----------------------------------------------------------
    # 4. train / test split
    # ----------------------------------------------------------
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
    energy_df = pd.read_csv('/Users/muhaoguo/Documents/study/Paper_Projects/PESGM/data/kaggle/energy_dataset.csv')
    weather_df = pd.read_csv('/Users/muhaoguo/Documents/study/Paper_Projects/PESGM/data/kaggle/weather_features.csv')

    # Convert timestamp columns to datetime format for easier merging and plotting
    energy_df['time'] = pd.to_datetime(energy_df['time'])
    weather_df['time'] = pd.to_datetime(weather_df['dt_iso'])

    # Merge datasets on the 'timestamp' column
    merged_df = pd.merge(energy_df, weather_df, on='time', how='inner')
    merged_df = merged_df[['time', 'temp', 'total load actual']].dropna()
    merged_df = merged_df[::5]

    time = merged_df["time"].values
    print(time)
    exit()
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
    p_energy  = "/Users/muhaoguo/Documents/study/Paper_Projects/PESGM/data/Spanish/energy_dataset.csv"
    p_weather = "/Users/muhaoguo/Documents/study/Paper_Projects/PESGM/data/Spanish/weather_features.csv"

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
                                       freq="H")
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
    file_path = "/Users/muhaoguo/Documents/study/Paper_Projects/PESGM/data/powerconsumption/powerconsumption.csv"  # <-- Adjust to your actual CSV
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


def plot_data(times, energy, temp, workday, season_feat,
                   alpha=0.5, lw=1.0, cmap="viridis"):
    """
    Overlay *all* weeks in four side-by-side sub-figures.

    Parameters
    ----------
    times, energy, temp, workday, season_feat : list/ndarray
        Output from your get_data_…_weekly routine.
    alpha : float
        Per-curve transparency (≤1).  Lower → less clutter.
    lw : float
        Line width.
    cmap : str or matplotlib Colormap
        Used to give each week a slightly different colour.
    """
    n_weeks = len(times)
    if n_weeks == 0:
        print("Nothing to plot.")
        return

    # colour map to distinguish weeks (wraps if >256)
    colours = plt.cm.get_cmap(cmap, n_weeks)

    fig, axes = plt.subplots(
        nrows=1, ncols=4, figsize=(22, 4),
        sharex=False, sharey=False,
        gridspec_kw={"wspace": 0.25})

    date_fmt = mdates.DateFormatter("%b\n%d")

    # -------------------------------------------------------------
    # iterate once, plotting the same week on all four axes
    # -------------------------------------------------------------
    for w in range(n_weeks):
        c = colours(w)

        axes[0].plot(times[w], energy[w],  color=c, alpha=alpha, lw=lw)
        axes[1].plot(times[w], temp[w],    color=c, alpha=alpha, lw=lw)
        axes[2].step(times[w], workday[w], where="mid",
                     color=c, alpha=alpha, lw=lw)
        axes[3].step(times[w], season_feat[w], where="mid",
                     color=c, alpha=alpha, lw=lw)

    # -------------------------------------------------------------
    # cosmetics
    # -------------------------------------------------------------
    axes[0].set_title("Energy (norm.)")
    axes[0].set_ylabel("0–1")
    axes[1].set_title("Temperature (norm.)")
    axes[2].set_title("Weekend flag")
    axes[2].set_ylim(-0.1, 1.1)
    axes[3].set_title("Season (0–3)")
    axes[3].set_ylim(-0.2, 3.2)

    for ax in axes:
        ax.xaxis.set_major_formatter(date_fmt)
        ax.tick_params(axis="x", rotation=45, labelsize=8)

    fig.suptitle(f"Overlay of {n_weeks} weeks", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.show()

    ##
    fig, axes = plt.subplots(  nrows=1, ncols=4, figsize=(22, 4),  sharex=False, sharey=False,  gridspec_kw={"wspace": 0.25})
    date_fmt = mdates.DateFormatter("%b\n%d")
    # -------------------------------------------------------------
    # iterate once, plotting the same week on all four axes
    # -------------------------------------------------------------
    for w in range(n_weeks):
        c = colours(w)
        axes[0].plot(energy[w], color=c, alpha=alpha, lw=lw)
        axes[1].plot(temp[w], color=c, alpha=alpha, lw=lw)
        axes[2].plot(workday[w], color=c, alpha=alpha, lw=lw)
        axes[3].plot(season_feat[w], color=c, alpha=alpha, lw=lw)

    # -------------------------------------------------------------
    # cosmetics
    # -------------------------------------------------------------
    axes[0].set_title("Energy (norm.)")
    axes[0].set_ylabel("0–1")
    axes[1].set_title("Temperature (norm.)")
    axes[2].set_title("Weekend flag")
    axes[2].set_ylim(-0.1, 1.1)
    axes[3].set_title("Season (0–3)")
    axes[3].set_ylim(-0.2, 3.2)

    for ax in axes:
        ax.xaxis.set_major_formatter(date_fmt)
        ax.tick_params(axis="x", rotation=45, labelsize=8)

    fig.suptitle(f"Overlay of {n_weeks} weeks", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.show()




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
    csv_path = Path("/Users/muhaoguo/Documents/study/Paper_Projects/PESGM/data/powerconsumption/powerconsumption.csv")

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
    load_file = "/Users/muhaoguo/Documents/study/Paper_Projects/PESGM/data/Kaggle_2/power_usage_2016_to_2020.csv"
    df_load = pd.read_csv(load_file)
    df_load['DateTime'] = pd.to_datetime(df_load['StartDate'])
    df_load['Date'] = df_load['DateTime'].dt.date
    df_load.rename(columns={'Value (kWh)': 'Power'}, inplace=True)

    weather_file = "/Users/muhaoguo/Documents/study/Paper_Projects/PESGM/data/Kaggle_2/weather_2016_2020_daily.csv"
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
    p_load    = Path("/Users/muhaoguo/Documents/study/Paper_Projects/PESGM/data/residential/power_usage_2016_to_2020.csv")
    p_weather = Path("/Users/muhaoguo/Documents/study/Paper_Projects/PESGM/data/residential/weather_2016_2020_daily.csv")

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
    p_csv = Path("/Users/muhaoguo/Documents/study/Paper_Projects/PESGM/data/solar_weather.csv")
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


if __name__ == "__main__":
    times, energy, temp, workday, season_feat = get_data_building_weather_weekly()
    print(times.shape, energy.shape, temp.shape, workday.shape, season_feat.shape)

    times, energy, temp, workday, season_feat  = get_data_spanish_weekly()
    print(times.shape, energy.shape, temp.shape, workday.shape, season_feat.shape)

    times, energy, temp, workday, season_feat  = get_data_power_consumption_weekly()
    print(times.shape, energy.shape, temp.shape, workday.shape, season_feat.shape)

    times, energy, temp, workday, season_feat = get_data_residential_weekly()
    print(times.shape, energy.shape, temp.shape, workday.shape, season_feat.shape)

    times, energy, temp, workday, season_feat = get_data_solar_weather_weekly()
    print(times.shape, energy.shape, temp.shape, workday.shape, season_feat.shape)




