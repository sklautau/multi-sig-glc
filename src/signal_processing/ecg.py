"""
ECG signal processing"""

import numpy as np
import os
import matplotlib.pyplot as plt
import neurokit2 as nk
from typing import Dict, Any

from datasets_util.naming_conventions import DatasetConfig
from datasets_util.waveforms import read_sigmf_file
from datasets_util.waveforms import save_sigmf_signal_rf32_le
from signal_processing.cross_features import safe_mean, safe_std

# ======================================================
# Parameters
# ======================================================
SHOULD_PLOT = False  # False to disable plotting
# for naming convention when saving processed files (e.g., "file_id_filtered.csv")
REQUIRED_FS = 500  # assumed sampling frequency (Hz)

SUBJECT_IDS_TO_PLOT = {
    # "ieb_02", # several files
    "ieb_01",
    "ieb_02",
    "ieb_03",
    "ieb_10",
}

FILE_IDS_TO_PLOT = {
    "file_id20",
    # "file_id80", # bad signal
    "file_id90"
}

# ======================================================
# Visualization
# ======================================================


def __plot_input_output_waveforms(input_wav, output_wav, fs, title, title1="Raw signal", title2="Filtered signal") -> None:
    t = np.arange(len(input_wav)) / fs

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True)

    axes[0].plot(t, input_wav, lw=0.8, color="#79B1CE")
    axes[0].set_title(title1)
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, output_wav, lw=0.8, color="#332288")
    axes[1].set_title(title2)
    axes[1].grid(alpha=0.3)

    fig.suptitle(title)
    axes[0].set_xlabel("Time (s)")
    axes[1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()


def ecg_quality_file_processing(ecg_filename: str) -> tuple[np.ndarray, np.ndarray, dict]:
    if ecg_filename is None:
        raise Exception(f"File {ecg_filename} not found!")

    print(f"  Using ECG file: {ecg_filename}")

    input_signal, input_metadata = read_sigmf_file(ecg_filename)

    fs = input_metadata["global"]["core:sample_rate"]
    if fs != REQUIRED_FS:
        raise ValueError(
            f"Expected sampling frequency {REQUIRED_FS} Hz, but got {fs} Hz in file {ecg_filename}")

    try:
        # "dissimilarity" or "templatematch" or "ho2025" or others (see https://neuropsychology.github.io/NeuroKit/functions/ecg.html#ecg-quality)
        method = "templatematch"
        ecq_quality = nk.ecg.ecg_quality(input_signal, sampling_rate=REQUIRED_FS,
                                         method=method)
    except Exception as e:
        # `templatematch` and `dissimilarity` require at least one detected peak.
        # fallback to zeros if error (e.g., no detected peaks) occurs
        ecq_quality = np.zeros_like(input_signal)
        print(f"****Error**** processing file {ecg_filename}: {e}")

    return ecq_quality, input_signal, input_metadata


def ecg_quality_pipeline(dataset_config_file: str) -> None:
    """
    ECG processing pipeline for generating WAVEFORM_ID = quality
    """
    input_waveform_id = "filtered"
    waveform_id = "quality"

    # process the dataset
    datasetConfig = DatasetConfig(dataset_config_file)

    # make a deep copy to prevent external modifications
    df = datasetConfig.get_dataset_info_dataframe()

    # Filter only ecg signals. It is safe to modify df because df is a deep copy
    df = df[df["modality"].str.contains("ecg")]

    file_counter = 0
    for counter, row in df.iterrows():
        file_id = row["file_id"]
        participant_id = row["participant_id"]

        input_complete_path = datasetConfig.get_gen_complete_path(
            file_id, input_waveform_id)
        print(f"File ID: {file_id}, Complete path: {input_complete_path}")

        output_signal, input_signal, metadata = ecg_quality_file_processing(
            input_complete_path)

        output_filename = datasetConfig.get_gen_complete_path(
            file_id, waveform_id)
        # print("Relative path for file_id", file_id, ":", relative_path)

        # create output directory if it doesn't exist
        output_dir = os.path.dirname(output_filename)
        os.makedirs(output_dir, exist_ok=True)
        # Save to proper location
        save_sigmf_signal_rf32_le(output_signal, metadata, output_filename)
        print(f"Saved processed data for {file_id} to {output_filename}")

        file_counter += 1

        # --------------------------------------------------
        # Plot only selected subjects
        # --------------------------------------------------

        # --------------------------------------------------
        # Plot only selected subjects
        # --------------------------------------------------
        if SHOULD_PLOT and participant_id in SUBJECT_IDS_TO_PLOT:
            __plot_input_output_waveforms(
                input_signal,
                output_signal,
                REQUIRED_FS,
                title=f"Subject {participant_id}",
                title1="Filtered ECG signal",
                title2="ECG quality"
            )

        if SHOULD_PLOT and file_id in FILE_IDS_TO_PLOT:
            signals, info = nk.ecg_process(
                output_signal, sampling_rate=REQUIRED_FS, method='neurokit', show=False)

            nk.ecg_plot(signals, info)

            plt.show()

    print(f"\nFinished processing {file_counter} files.")


def ecg_filtering_file_processing(ecg_filename: str) -> tuple[np.ndarray, np.ndarray, dict]:
    if ecg_filename is None:
        raise Exception("  ⚠ No _wave file found, skipping")

    print(f"  Using ECG file: {ecg_filename}")

    raw, input_metadata = read_sigmf_file(ecg_filename)

    fs = input_metadata["global"]["core:sample_rate"]
    if fs != REQUIRED_FS:
        raise ValueError(
            f"Expected sampling frequency {REQUIRED_FS} Hz, but got {fs} Hz in file {ecg_filename}")

    #  If signals are reversed in acquisition, uncomment:
    # raw = np.flipud(raw)

    # see https://neuropsychology.github.io/NeuroKit/functions/ecg.html#ecg-process
    # extract the cleaned ECG signal from the DataFrame
    # note that ecg_process has a "cleaning" stage, and
    # we also applied a bandpass Butterworth filter
    # I will disable it because it's breaking for file_id80, but you can experiment with it if you want
    if False:
        signals_df, info = nk.ecg.ecg_process(raw, sampling_rate=REQUIRED_FS,
                                              method="neurokit", show=False)
        filt = signals_df["ECG_Clean"]
    else:
        filt = nk.ecg.ecg_clean(raw, sampling_rate=REQUIRED_FS,
                                method="neurokit")
    return filt, raw, input_metadata


def ecg_filtering_pipeline(dataset_config_file: str) -> None:
    """
    ECG processing pipeline for generating WAVEFORM_ID = filtered
    """
    waveform_id = "filtered"

    # process the dataset
    datasetConfig = DatasetConfig(dataset_config_file)

    # make a deep copy to prevent external modifications
    df = datasetConfig.get_dataset_info_dataframe()

    # Filter only ecg signals. It is safe to modify df because df is a deep copy
    df = df[df["modality"].str.contains("ecg")]

    file_counter = 0
    for counter, row in df.iterrows():
        file_id = row["file_id"]
        participant_id = row["participant_id"]

        raw_complete_path = datasetConfig.get_raw_complete_path(file_id)
        print(f"File ID: {file_id}, Complete path: {raw_complete_path}")

        filtered_signal, input_signal, metadata = ecg_filtering_file_processing(
            raw_complete_path)

        output_filename = datasetConfig.get_gen_complete_path(
            file_id, waveform_id)
        # print("Relative path for file_id", file_id, ":", relative_path)

        # create output directory if it doesn't exist
        output_dir = os.path.dirname(output_filename)
        os.makedirs(output_dir, exist_ok=True)
        # Save to proper location
        save_sigmf_signal_rf32_le(filtered_signal, metadata, output_filename)
        print(f"Saved processed data for {file_id} to {output_filename}")

        file_counter += 1

        # --------------------------------------------------
        # Plot only selected subjects
        # --------------------------------------------------
        if SHOULD_PLOT and participant_id in SUBJECT_IDS_TO_PLOT:
            __plot_input_output_waveforms(
                input_signal,
                filtered_signal,
                REQUIRED_FS,
                title=f"Subject {participant_id}"
            )

        if SHOULD_PLOT and file_id in FILE_IDS_TO_PLOT:
            signals, info = nk.ecg_process(
                filtered_signal, sampling_rate=REQUIRED_FS)

            nk.ecg_plot(signals, info)

            plt.show()

    print(f"\nFinished processing {file_counter} files.")


def extract_ecg_features(ecg: np.ndarray, fs: int) -> Dict[str, Any]:

    try:
        signals, info = nk.ecg_process(ecg, sampling_rate=fs)
        rpeaks = info["ECG_R_Peaks"]

        if False:  # you can enable this block to extract more features using neurokit2
            nk.hrv_frequency(rpeaks, sampling_rate=fs)

            # Nonlinear HRV features
            nk.hrv_nonlinear(rpeaks, sampling_rate=fs)

            # Morphological (wave-based features)
            signals, info = nk.ecg_process(ecg, sampling_rate=fs)
            _, waves = nk.ecg_delineate(
                ecg, info["ECG_R_Peaks"], sampling_rate=fs)

            signals, info = nk.ecg_process(ecg, sampling_rate=fs)

            rpeaks = info["ECG_R_Peaks"]

            features = nk.hrv(rpeaks, sampling_rate=fs)

            features["mean_hr"] = signals["ECG_Rate"].mean()
            features["std_hr"] = signals["ECG_Rate"].std()

        # --- HRV ---
        hrv = nk.hrv(rpeaks, sampling_rate=fs)

        features = {
            "ecg_hr_mean": safe_mean(signals["ECG_Rate"]),
            "ecg_hr_std": safe_std(signals["ECG_Rate"]),
        }

        # Frequency-domain (HRV spectral)
        # flatten HRV
        for col in hrv.columns:
            features[f"ecg_{col.lower()}"] = hrv.iloc[0][col]

        # --- RR ---
        rr = np.diff(rpeaks) / fs
        features["ecg_rr_mean"] = safe_mean(rr)
        features["ecg_rr_std"] = safe_std(rr)

        # --- SQI proxy ---
        features["ecg_sqi"] = np.mean(
            (signals["ECG_Rate"] > 40) & (signals["ECG_Rate"] < 180))

        return features

    except Exception as e:
        raise RuntimeError(f"ECG feature extraction failed: {e}") from e

    return features


if __name__ == "__main__":
    dataset_config_file = "multimodal_dataset_folders.json"
    ecg_filtering_pipeline(dataset_config_file)
    ecg_quality_pipeline(dataset_config_file)
