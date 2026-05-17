"""
Signal processing over bioimpedance data (time and frequency domain)
"""

import pandas as pd
import numpy as np
from io import StringIO
from scipy.signal import savgol_filter
import os
import matplotlib.pyplot as plt
from typing import Dict, Any

from datasets_util.naming_conventions import DatasetConfig
from signal_processing.cross_features import safe_mean, safe_std

SHOULD_PLOT = False  # Set to True to enable plotting of raw vs filtered signals
APPLY_PHASE_UNWRAP = True  # Set to True to apply phase unwrapping before smoothing


def __plot_quality(df_raw, df_quality, modality, file_id=None):
    """
    Plots raw vs filtered magnitude and phase.

    Args:
        df_raw: original dataframe (must contain Zmag, Zphi and t or f)
        df_filtered: processed dataframe (must contain mag_filtered, phase_filtered)
        modality: 'bioimpedance_time' or 'bioimpedance_frequency'
        file_id: optional (for title)
    """

    # --------------------------------------------------
    # Select axis
    # --------------------------------------------------
    if "bioimpedance_time" in modality:
        x = df_raw["t"].values
        x_label = "Time (s)"
    elif "bioimpedance_frequency" in modality:
        x = df_raw["f"].values
        x_label = "Frequency (Hz)"
    else:
        raise ValueError(f"Unknown modality: {modality}")

    # --------------------------------------------------
    # Extract signals
    # --------------------------------------------------
    mag_raw = df_raw["Zmag"].values
    phase_raw = df_raw["Zphi"].values

    quality = df_quality["quality"].values

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    plt.figure(figsize=(10, 6))

    # Magnitude
    plt.subplot(2, 1, 1)
    plt.plot(x, mag_raw, label="Bioimpedance", alpha=0.6)
    if "bioimpedance_frequency" in modality:
        plt.xscale("log")
    plt.ylabel("Impedance Magnitude (Ohm)")
    plt.title(f"File: {file_id} | Magnitude")
    plt.legend()
    plt.grid(True)

    # Phase
    plt.subplot(2, 1, 2)
    plt.plot(x, quality, label="Quality", linewidth=2)
    if "bioimpedance_frequency" in modality:
        plt.xscale("log")
    plt.xlabel(x_label)
    plt.ylabel("Quality")
    plt.title("Quality (1=best)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def __plot_raw_vs_filtered(df_raw, df_filtered, modality, file_id=None):
    """
    Plots raw vs filtered magnitude and phase.

    Args:
        df_raw: original dataframe (must contain Zmag, Zphi and t or f)
        df_filtered: processed dataframe (must contain mag_filtered, phase_filtered)
        modality: 'bioimpedance_time' or 'bioimpedance_frequency'
        file_id: optional (for title)
    """

    # --------------------------------------------------
    # Select axis
    # --------------------------------------------------
    if "bioimpedance_time" in modality:
        x = df_raw["t"].values
        x_label = "Time (s)"
    elif "bioimpedance_frequency" in modality:
        x = df_raw["f"].values
        x_label = "Frequency (Hz)"
    else:
        raise ValueError(f"Unknown modality: {modality}")

    # --------------------------------------------------
    # Extract signals
    # --------------------------------------------------
    mag_raw = df_raw["Zmag"].values
    phase_raw = df_raw["Zphi"].values

    mag_f = df_filtered["mag_filtered"].values
    phase_f = df_filtered["phase_filtered"].values

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    plt.figure(figsize=(10, 6))

    # Magnitude
    plt.subplot(2, 1, 1)
    plt.plot(x, mag_raw, label="Raw", alpha=0.6)
    plt.plot(x, mag_f, label="Filtered", linewidth=2)
    if "bioimpedance_frequency" in modality:
        plt.xscale("log")
    plt.ylabel("Impedance Magnitude (Ohm)")
    plt.title(f"File: {file_id} | Magnitude")
    plt.legend()
    plt.grid(True)

    # Phase
    plt.subplot(2, 1, 2)
    plt.plot(x, phase_raw, label="Raw", alpha=0.6)
    plt.plot(x, phase_f, label="Filtered", linewidth=2)
    if "bioimpedance_frequency" in modality:
        plt.xscale("log")
    plt.xlabel(x_label)
    plt.ylabel("Phase (degrees)")
    plt.title("Phase")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# Loader for bioimpedance files (your format)
# --------------------------------------------------


def load_bioimp_file(path):
    header_lines = []
    data_lines = []
    found_table = False

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()

            if stripped.startswith("mnum"):
                found_table = True

            if found_table:
                data_lines.append(line)
            else:
                if stripped:
                    header_lines.append(stripped)

    if not data_lines:
        raise ValueError(f"No measurement table found in {path}")

    df = pd.read_csv(StringIO("".join(data_lines)), sep=None, engine="python")
    return header_lines, df


# --------------------------------------------------
# Smoothing function
# --------------------------------------------------
def __smooth_signal(x, window=11, poly=3):
    """
    Applies Savitzky-Golay smoothing with safety checks.
    """
    n = len(x)

    # Window must be odd and <= n
    window = min(window, n if n % 2 == 1 else n - 1)
    if window < 5:
        return x  # too short → skip

    if window % 2 == 0:
        window -= 1

    return savgol_filter(x, window_length=window, polyorder=poly)


# --------------------------------------------------
# Process a single file
# --------------------------------------------------
def bioimpedance_frequency_filtering_file_processing(path, modality, file_id):
    header, df = load_bioimp_file(path)

    # Detect domain
    if "bioimpedance_time" in modality:
        x = df["t"].values
        x_name = "time"
    elif "bioimpedance_frequency" in modality:
        x = df["f"].values
        x_name = "frequency"
    else:
        raise ValueError(f"Unknown modality: {modality}")

    mag = df["Zmag"].values
    phase = df["Zphi"].values

    if APPLY_PHASE_UNWRAP:
        phase = np.rad2deg(np.unwrap(np.deg2rad(phase)))

    # Smooth
    mag_f = __smooth_signal(mag)
    phase_f = __smooth_signal(phase)

    # Build output dataframe (WITH file_id)
    df_out = pd.DataFrame({
        "file_id": file_id,
        x_name: x,
        "mag_filtered": mag_f,
        "phase_filtered": phase_f
    })

    if SHOULD_PLOT:
        __plot_raw_vs_filtered(df, df_out, modality, file_id)
    return df_out


def bioimpedance_frequency_quality_file_processing(path, modality, file_id):
    header, df = load_bioimp_file(path)

    # Detect domain
    if "bioimpedance_time" in modality:
        x = df["t"].values
        x_name = "time"
    elif "bioimpedance_frequency" in modality:
        x = df["f"].values
        x_name = "frequency"
    else:
        raise ValueError(f"Unknown modality: {modality}")

    mag = np.array(df["Zmag"].values)

    # First normalize mag to [0,1] for quality calculation
    mag = (mag - np.min(mag)) / (np.max(mag) - np.min(mag))

    # Smooth
    mag_f = __smooth_signal(mag)

    # define quality based on mean-squared error (mse) between raw and smoothed signals
    rmse = np.array((np.array(mag) - np.array(mag_f))**2.0)
    empirical_factor = 10.0  # Adjust this factor to calibrate quality scores
    quality = 1 - empirical_factor*rmse  # quality in [0,1], higher is better
    # if quality is negative due to high rmse, set to 0
    quality[quality < 0] = 0
    # if quality is above 1 due to low rmse, set to 1
    quality[quality > 1] = 1

    # Build output dataframe (WITH file_id)
    df_out = pd.DataFrame({
        "file_id": file_id,
        x_name: x,
        "quality": quality
    })

    if SHOULD_PLOT:
        __plot_quality(df, df_out, modality, file_id=file_id)
    return df_out


# --------------------------------------------------
# Main pipelines
# --------------------------------------------------


def bioimpedance_frequency_filtering_pipeline(dataset_config_file: str):
    # for naming convention when saving processed files (e.g., "file_id_filtered.csv")
    waveform_id = "filtered"

    # Example: process a dataset
    datasetConfig = DatasetConfig(dataset_config_file)
    base_path = datasetConfig.get_dataset_root_path()

    df = datasetConfig.get_dataset_info_dataframe()

    # Filter only bioimpedance
    df_bio = df[df["modality"].str.contains("bioimpedance")].copy()

    counter = 0
    for _, row in df_bio.iterrows():
        file_id = row["file_id"]
        modality = row["modality"]

        raw_relative_path = datasetConfig.get_raw_relative_path(file_id)
        full_path = os.path.join(base_path, raw_relative_path)

        try:
            df_out = bioimpedance_frequency_filtering_file_processing(
                full_path, modality, file_id)
            # file_id is not needed in the output CSV, so we can drop it before saving
            df_out = df_out.drop(columns=["file_id"])
            print(f"\nProcessed {file_id}")
            print(df_out.head())

            output_filename = datasetConfig.get_gen_complete_path(
                file_id, waveform_id, new_extension="csv")

            # create output directory if it doesn't exist
            output_dir = os.path.dirname(output_filename)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # Save to proper location
            df_out.to_csv(output_filename, index=False)
            print(f"Saved processed data for {file_id} to {output_filename}")

            counter += 1
        except Exception as e:
            print(f"[ERROR] {file_id}: {e}")

    print(f"\nFinished processing {counter} files.")


def bioimpedance_frequency_quality_pipeline(dataset_config_file: str):
    # for naming convention when saving processed files (e.g., "file_id_filtered.csv")
    waveform_id = "quality"

    # Example: process a dataset
    datasetConfig = DatasetConfig(dataset_config_file)
    base_path = datasetConfig.get_dataset_root_path()

    df = datasetConfig.get_dataset_info_dataframe()

    # Filter only bioimpedance
    df_bio = df[df["modality"].str.contains("bioimpedance")].copy()

    counter = 0
    for _, row in df_bio.iterrows():
        file_id = row["file_id"]
        modality = row["modality"]

        raw_relative_path = datasetConfig.get_raw_relative_path(file_id)
        full_path = os.path.join(base_path, raw_relative_path)

        try:
            df_out = bioimpedance_frequency_quality_file_processing(
                full_path, modality, file_id)
            # file_id is not needed in the output CSV, so we can drop it before saving
            df_out = df_out.drop(columns=["file_id"])
            print(f"\nProcessed {file_id}")
            print(df_out.head())

            output_filename = datasetConfig.get_gen_complete_path(
                file_id, waveform_id, new_extension="csv")

            # create output directory if it doesn't exist
            output_dir = os.path.dirname(output_filename)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # Save to proper location
            df_out.to_csv(output_filename, index=False)
            print(f"Saved processed data for {file_id} to {output_filename}")

            counter += 1
        except Exception as e:
            print(f"[ERROR] {file_id}: {e}")

    print(f"\nFinished processing {counter} files.")


def extract_bioimp_features(bioimp: np.ndarray, fs: int) -> Dict[str, Any]:

    try:
        features = {}

        features["bioimp_mean"] = safe_mean(bioimp)
        features["bioimp_std"] = safe_std(bioimp)

        # slope
        slope = np.gradient(bioimp)
        features["bioimp_slope_mean"] = safe_mean(slope)
        features["bioimp_slope_std"] = safe_std(slope)

        features["bioimp_peak_to_peak"] = np.max(bioimp) - np.min(bioimp)

        return features

    except Exception as e:
        raise RuntimeError(
            f"Bioimpedance feature extraction failed: {e}") from e

    return features


if __name__ == "__main__":
    dataset_config_file = "multimodal_dataset_folders.json"
    bioimpedance_frequency_filtering_pipeline(dataset_config_file)
    bioimpedance_frequency_quality_pipeline(dataset_config_file)
