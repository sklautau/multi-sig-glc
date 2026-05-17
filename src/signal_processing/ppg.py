"""
PPG signal processing
Non-morphological - IMF intrinsic mode functions, energy-based features, entropy-based features, etc.

Morphological - notch-independent morphological features

Morphological - pulse-based, depends on notch


"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks
from scipy.stats import kurtosis
from scipy.signal import welch
from statistics import variance
import neurokit2 as nk
import matplotlib.pyplot as plt
from typing import Dict, Any
import pandas as pd

from datasets_util.naming_conventions import DatasetConfig
from datasets_util.waveforms import read_sigmf_file
from datasets_util.waveforms import save_sigmf_signal_rf32_le
from datasets_util.features import append_identification
from signal_processing.cross_features import safe_mean, safe_std
from signal_processing.ppg_quality import estimate_sqi_neurokit_tm
from signal_processing.ppg_quality import estimate_sqi_custom_version
from signal_processing.ppg_quality import estimate_single_sqi_value
# from signal_processing.ppg_features_by_ufsc import extract_features_from_average_pulse


# ======================================================
# Parameters
# ======================================================
SHOULD_PLOT = False  # False to disable plotting
SHOULD_USE_CUSTOM_SQI = False  # If False, use neurokit2 SQI
# for naming convention when saving processed files (e.g., "file_id_filtered.csv")
DEFAULT_REQUIRED_PPG_FS = 500  # fallback sampling frequency (Hz)

# filter parameters
HIGHPASS = 0.5  # bandpass from 0.5 to 4 Hz
LOWPASS = 4.0
ORDER = 4

# If False, errors will be logged but the pipeline will continue, returning NaN or empty values for features that failed to extract.
RAISE_EXCEPTION_ON_PPG_PROCESSING = True

SUBJECT_IDS_TO_PLOT = {
    "hu_01",  # IEB_1 subject 1
    "hu_02",
    "hu_03",
    "ieb_01",  # IEB_3 subject 1
    "ieb_02",
    "ieb_03",
    "ieb_04",
    "ieb_05",
    "ieb_06",
    "ieb_07",
    "ieb_08",
}


# ======================================================
# UTILITY METHODS
# ======================================================

def _energy_in_band(min_f, max_f, psd, f):
    m = (f >= min_f) & (f <= max_f)
    return np.trapezoid(psd[m], f[m]) if m.any() else 0


def __detect_beats_neurokit(ppg_signal, fs):
    """
    Detect peaks and onsets using NeuroKit WITHOUT filtering.
    """

    signal = np.asarray(ppg_signal).astype(float)

    # DO NOT call nk.ppg_clean
    info = nk.ppg_findpeaks(signal, sampling_rate=fs, method="elgendi")

    peaks = info.get("PPG_Peaks", [])

    # Some methods return onsets (Charlton)
    onsets = info.get("PPG_Onsets", None)

    # fallback: estimate onsets via local minima
    if onsets is None or len(onsets) == 0:
        inverted = -signal
        onsets = nk.signal_findpeaks(inverted)["Peaks"]

    beats = []

    for i in range(len(peaks)):
        p = peaks[i]

        prev_onsets = onsets[onsets < p]
        next_onsets = onsets[onsets > p]

        if len(prev_onsets) == 0 or len(next_onsets) == 0:
            continue

        start = prev_onsets[-1]
        end = next_onsets[0]

        if end > start:
            beats.append((start, p, end))

    return beats


def __butter_bandpass_filter(signal, fs):
    # ======================================================
    # Butterworth filter
    # ======================================================
    signal = np.asarray(signal, dtype=float)

    nyq = fs / 2.0  # Nyquist frequency
    low = HIGHPASS / nyq  # use normalized values
    high = LOWPASS / nyq

    if low > high:
        raise ValueError("lowcut is greater than highcut")

    b, a = butter(ORDER, [low, high], btype="band")
    # print("b, a", b, "\n", a)
    # print("abs(poles) = ", np.abs(np.roots(a)))
    filtered = filtfilt(b, a, signal)
    return filtered


def _debug_processing_files(dataset_config_file):
    waveform_id = "quality"
    input_waveform_id = "filtered"

    # Example: process a dataset
    datasetConfig = DatasetConfig(dataset_config_file)
    required_ppg_fs = datasetConfig.get_ppg_fs()

    # a deep copy is made below, to allow external modifications
    df = datasetConfig.get_dataset_info_dataframe()

    # Filter only ppg signals. It is safe to modify df because df is a deep copy
    df = df[df["modality"].str.contains("ppg")]

    file_counter = 0
    for counter, row in df.iterrows():
        file_id = row["file_id"]
        raw_complete_path = datasetConfig.get_raw_complete_path(file_id)
        # full_path = os.path.join(base_path, raw_relative_path)
        print(f"File ID: {file_id}, Complete path: {raw_complete_path}")

        filtered_signal, input_signal, metadata = ppg_filtering_file_processing(
            raw_complete_path, required_ppg_fs)

        # features = extract_features_group1_peak_based(
        #    filtered_signal, None, fs)
        # print("AAAAAAAAAAAAAA ", features)


def _normalize_peaks(peaks):
    '''Make sure we deal with np.array'''
    if peaks is None:
        return None

    # pandas Series => numpy
    peaks = np.asarray(peaks)

    if peaks.ndim == 1 and peaks.dtype == bool:
        # boolean mask → indices
        return np.where(peaks)[0]

    # already indices
    if np.issubdtype(peaks.dtype, np.integer):
        return peaks

    # fallback (e.g., float or weird types)
    try:
        return peaks.astype(int)
    except Exception:
        return None


def __plot_input_output_waveforms(input_wav, output_wav, fs, title, title1="Raw signal", title2="Filtered signal"):
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

# ======================================================
# METHODS TO EXECUTE PROCESSING PIPELINE
# ======================================================


def ppg_filtering_file_processing(ppg_filename: str, required_ppg_fs: int = DEFAULT_REQUIRED_PPG_FS) -> tuple[np.ndarray, np.ndarray, dict]:
    if ppg_filename is None:
        raise Exception("  ⚠ No _wave file found, skipping")

    print(f"  Using PPG file: {ppg_filename}")

    raw, input_metadata = read_sigmf_file(ppg_filename)

    fs = input_metadata["global"]["core:sample_rate"]
    if fs != required_ppg_fs:
        raise ValueError(
            f"Expected sampling frequency {required_ppg_fs} Hz, but got {fs} Hz in file {ppg_filename}")

    #  If signals are reversed in acquisition, uncomment:
    # raw = np.flipud(raw)

    filt = __butter_bandpass_filter(raw, fs)

    return filt, raw, input_metadata


def ppg_quality_file_processing(ppg_filename: str, required_ppg_fs: int = DEFAULT_REQUIRED_PPG_FS) -> tuple[np.ndarray, np.ndarray, dict]:
    '''
    Estimate Signal Quality Index (SQI) for PPG signals,
    generating one SQI value for each PPG sample.
    '''
    if ppg_filename is None:
        raise Exception(f"File {ppg_filename} not found!")

    print(f"  Using PPG file: {ppg_filename}")

    ppg_signal, input_metadata = read_sigmf_file(ppg_filename)

    fs = input_metadata["global"]["core:sample_rate"]
    if fs != required_ppg_fs:
        raise ValueError(
            f"Expected sampling frequency {required_ppg_fs} Hz, but got {fs} Hz in file {ppg_filename}")

    if SHOULD_USE_CUSTOM_SQI:
        sqi_signal = estimate_sqi_custom_version(ppg_signal, fs)
    else:
        sqi_signal = estimate_sqi_neurokit_tm(ppg_signal, fs)

    from signal_processing.ppg_quality import plot_sqi_comparison

    if False:  # SHOULD_PLOT:
        plot_sqi_comparison(ppg_signal, fs=fs, title=ppg_filename)

    if False:  # debug, compare
        # mean_sqi is often a smaller number:
        mean_sqi = estimate_single_sqi_value(ppg_signal, fs)
        mean_sqi_from_array = np.mean(sqi_signal)
        print("%%%%%%%%%%% DEBUG: mean_sqi, mean_sqi_from_array = ",
              mean_sqi, mean_sqi_from_array)

    return sqi_signal, ppg_signal, input_metadata


def ppg_filtering_pipeline(dataset_config_file: str) -> None:
    waveform_id = "filtered"

    # Example: process a dataset
    datasetConfig = DatasetConfig(dataset_config_file)
    required_ppg_fs = datasetConfig.get_ppg_fs()

    # make a deep copy to prevent external modifications
    df = datasetConfig.get_dataset_info_dataframe()

    print('df["modality"].value_counts()', df["modality"].value_counts())

    # Filter only ppg signals. It is safe to modify df because df is a deep copy
    df = df[df["modality"].str.contains("ppg")]

    file_counter = 0
    for counter, row in df.iterrows():
        file_id = row["file_id"]
        participant_id = row["participant_id"]

        raw_complete_path = datasetConfig.get_raw_complete_path(file_id)
        # full_path = os.path.join(base_path, raw_relative_path)
        print(f"File ID: {file_id}, Complete path: {raw_complete_path}")

        filtered_signal, input_signal, metadata = ppg_filtering_file_processing(
            raw_complete_path, required_ppg_fs)

        output_filename = datasetConfig.get_gen_complete_path(
            file_id, waveform_id)
        # print("Relative path for file_id", file_id, ":", relative_path)

        # create output directory if it doesn't exist
        output_dir = os.path.dirname(output_filename)
        os.makedirs(output_dir, exist_ok=True)
        # Save to proper location
        save_sigmf_signal_rf32_le(filtered_signal, metadata, output_filename)
        print(
            f"Saved processed data for file_id={file_id} to {output_filename}")

        file_counter += 1
        # --------------------------------------------------
        # Plot only selected subjects
        # --------------------------------------------------
        if SHOULD_PLOT and participant_id in SUBJECT_IDS_TO_PLOT:
            __plot_input_output_waveforms(
                input_signal,
                filtered_signal,
                required_ppg_fs,
                title=f"Subject {participant_id}"
            )

        if SHOULD_PLOT:
            signals, info = nk.ppg_process(
                filtered_signal, sampling_rate=required_ppg_fs)

            nk.ppg_plot(signals, info)

            plt.show()

    print(f"\nFinished processing {file_counter} files.")


def ppg_quality_pipeline(dataset_config_file: str) -> None:
    waveform_id = "quality"
    input_waveform_id = "filtered"

    # Example: process a dataset
    datasetConfig = DatasetConfig(dataset_config_file)
    required_ppg_fs = datasetConfig.get_ppg_fs()

    # a deep copy is made below, to allow external modifications
    df = datasetConfig.get_dataset_info_dataframe()

    # Filter only ppg signals. It is safe to modify df because df is a deep copy
    df = df[df["modality"].str.contains("ppg")]

    file_counter = 0
    for counter, row in df.iterrows():
        file_id = row["file_id"]
        participant_id = row["participant_id"]

        filtered_complete_path = datasetConfig.get_gen_complete_path(
            file_id, input_waveform_id)
        print(
            f"File ID: {file_id}, Patient ID: {participant_id}, Complete path: {filtered_complete_path}")

        sqi_signal, input_signal, metadata = ppg_quality_file_processing(
            filtered_complete_path, required_ppg_fs)

        output_filename = datasetConfig.get_gen_complete_path(
            file_id, waveform_id)
        # print("Relative path for file_id", file_id, ":", relative_path)

        # create output directory if it doesn't exist
        output_dir = os.path.dirname(output_filename)
        os.makedirs(output_dir, exist_ok=True)
        # Save to proper location
        save_sigmf_signal_rf32_le(sqi_signal, metadata, output_filename)
        print(
            f"Saved processed data for file_id={file_id} to {output_filename}")

        file_counter += 1

        if SHOULD_PLOT and participant_id in SUBJECT_IDS_TO_PLOT:
            __plot_input_output_waveforms(
                input_signal,
                sqi_signal,
                required_ppg_fs,
                title=f"Subject {participant_id}",
                title1="Filtered PPG signal",
                title2="SQI"
            )

        if SHOULD_PLOT:
            # try to plot and catch exceptions (e.g., if sqi_signal is empty or has issues)
            try:
                signals, info = nk.ppg_process(
                    sqi_signal, sampling_rate=required_ppg_fs)
                nk.ppg_plot(signals, info)

                plt.show()
            except Exception as e:
                print(
                    f"   ########## Error occurred while estimating SQI: {e}")
                continue

    print(f"\nFinished processing {file_counter} files.")


# ======================================================
# FEATURE EXTRACTION
# ======================================================

def extract_ppg_features(ppg: np.ndarray, fs: int) -> Dict[str, Any]:

    try:
        # https://neuropsychology.github.io/NeuroKit/examples/ecg_hrv/ecg_hrv.html
        # Use Neurokit2 to find PPG events
        signals, info = nk.ppg_process(ppg, sampling_rate=fs)
        # Get peaks and normalize to numpy array (after nk.ppg_process)
        peaks = _normalize_peaks(info.get("PPG_Peaks", None))
        if peaks is not None:
            peaks = np.sort(peaks)
    except Exception as e:
        if RAISE_EXCEPTION_ON_PPG_PROCESSING:
            raise Exception(e)
        else:
            print("WARNING: Neurokit2 ppg_process failed!!!")

    # Calculate each group of features and rename the features such that
    # all of them have the preamble ppg followed by a group identification

    # Compute pulse features as Luis
    # pf = pulseFeatures(ppg, fs)
    # pf.compute(show=False)

    # Peaks-based features
    features1_peaks = extract_features_group1_peaks(signals, peaks, fs)
    features1_peaks = append_identification(features1_peaks, "ppg", "p1")

    # Beats-based features
    features2_beats = extract_features_group2_beats(ppg, fs=fs)
    features2_beats = append_identification(features2_beats, "ppg", "b2")

    # Power spectral density (PSD)-based features
    features3_spectral = extract_features_group3_spectral(ppg, fs)
    features3_spectral = append_identification(
        features3_spectral, "ppg", "s3")

    # HRV features
    features4_hrv = extract_features_group4_hrv(ppg, peaks, fs)
    features4_hrv = append_identification(features4_hrv, "ppg", "h4")

    concatenated_dictionary = {
        **features1_peaks,
        **features2_beats,
        **features3_spectral,
        **features4_hrv
    }

    if False:
        # Features from legacy UFSC code, which is not robust to low quality PPG:
        features5_ufsc = extract_features_from_average_pulse(ppg, fs)
        features5_ufsc = append_identification(features5_ufsc, "ppg", "u5")

        concatenated_dictionary = {
            **concatenated_dictionary,
            **features5_ufsc
        }

    return concatenated_dictionary


def extract_features_group1_peaks(signals, peaks, fs: int) -> Dict[str, Any]:
    features = {
        "ppg_hr_mean": safe_mean(signals["PPG_Rate"]),
        "ppg_hr_std": safe_std(signals["PPG_Rate"]),
    }

    # Amplitude
    clean = signals["PPG_Clean"]
    features["ppg_amp_mean"] = safe_mean(clean)
    features["ppg_amp_std"] = safe_std(clean)

    # Peak intervals
    if peaks is not None and len(peaks) >= 2:
        rr = np.diff(peaks) / fs
        features["ppg_peak_interval_std"] = safe_std(rr)
    else:
        features["ppg_peak_interval_std"] = np.nan

    # Derivatives
    d1 = np.gradient(clean)
    d2 = np.gradient(d1)

    features["ppg_d1_max_mean"] = safe_mean(np.abs(d1))
    features["ppg_d2_max_mean"] = safe_mean(np.abs(d2))

    # SQI (robust)
    if peaks is not None and len(peaks) >= 3:
        try:
            sqi_values = nk.ppg_quality(clean)
            features["ppg_sqi"] = safe_mean(
                sqi_values) if len(sqi_values) > 0 else np.nan
        except Exception:
            features["ppg_sqi"] = np.nan
    else:
        features["ppg_sqi"] = np.nan

    return features


def compute_metrics_for_all_beats(ppg, beats, fs) -> pd.DataFrame:
    # ======================================================
    # BEAT Metrics
    # ======================================================
    rows = []

    for start, peak, end in beats:
        beat = ppg[start:end]

        if len(beat) < 3:
            continue

        baseline = np.min(beat)
        amp = ppg[peak] - baseline

        T = (end - start) / fs
        Tr = (peak - start) / fs
        Tr_norm = Tr / T if T > 0 else np.nan

        auc = np.trapezoid(beat - baseline, dx=1/fs)

        half = baseline + amp / 2
        above = np.where(beat >= half)[0]
        width = (above[-1] - above[0]) / fs if len(above) >= 2 else np.nan

        rise_slope = amp / Tr if Tr > 0 else np.nan

        rows.append({
            "amplitude": amp,
            "T": T,
            "Tr": Tr,
            "Tr_norm": Tr_norm,
            "AUC": auc,
            "width": width,
            "rise_slope": rise_slope
        })

    return pd.DataFrame(rows)


def extract_features_group3_spectral(ppg, fs, show=False):
    # ======================================================
    # SPECTRAL FEATURES
    # ======================================================

    min_cycles = 5
    f_min = 0.7
    nperseg_min = int(min_cycles * fs / f_min)
    nperseg_target = int(fs / 0.05)
    nperseg = min(len(ppg), max(nperseg_min, nperseg_target))

    psd_freqs, psd = welch(ppg, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)

    pulse_band = (psd_freqs >= 0.6) & (psd_freqs <= 3.0)
    if not pulse_band.any():
        raise ValueError("No pulse band found")

    # Fundamental frequency
    idx_f0 = np.argmax(psd[pulse_band])
    f0 = psd_freqs[pulse_band][idx_f0]

    # Energies
    E1 = _energy_in_band(f0 * 0.9, f0 * 1.1, psd, psd_freqs)
    E2 = _energy_in_band(2 * f0 * 0.9, 2 * f0 * 1.1, psd, psd_freqs)

    eps = 1e-12
    harmonic_ratio = E2 / (E1 + eps)

    # Spectral stats
    psd_sum = np.sum(psd) + eps
    psd_norm = psd / psd_sum

    spectral_centroid = np.sum(psd_freqs * psd) / psd_sum
    spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + eps))

    mean_psd = np.mean(psd)
    std_psd = np.std(psd)
    var_psd = np.var(psd)
    kurtosis_psd = kurtosis(psd, fisher=True, bias=False)

    # Fundamental peak (restricted to band)
    max_power_1st = psd[pulse_band][idx_f0]
    f_max_1st = f0

    # Second harmonic (search near 2*f0)
    band_2nd = (psd_freqs >= 1.8*f0) & (psd_freqs <= 2.2*f0)
    if band_2nd.any():
        idx_2nd = np.argmax(psd[band_2nd])
        f_max_2nd = psd_freqs[band_2nd][idx_2nd]
        max_power_2nd = psd[band_2nd][idx_2nd]
    else:
        f_max_2nd = 0
        max_power_2nd = 0

    return {
        "f0": f0,
        "harmonic_ratio": harmonic_ratio,
        "spectral_centroid": spectral_centroid,
        "spectral_entropy": spectral_entropy,
        "MAX_POWER_1st": max_power_1st,
        "F_MAX_1st": f_max_1st,
        "MAX_POWER_2nd": max_power_2nd,
        "F_MAX_2nd": f_max_2nd,
        "MEAN_PSD": mean_psd,
        "STD_PSD": std_psd,
        "VAR_PSD": var_psd,
        "KUR_PSD": kurtosis_psd,
    }


def extract_features_group2_beats(segment, fs=500):

    beats = __detect_beats_neurokit(segment, fs)

    # feature aggregation block operating on beat-level PPG features (likely produced by NeuroKit2). It converts a per-beat DataFrame (beat_df) into summary statistics stored in a single dictionary row.
    beat_df = compute_metrics_for_all_beats(segment, beats, fs)

    # You have:
    # beat_df: a DataFrame where each row = one detected beat columns like:
    # amplitude, T, Tr, Tr_norm, AUC, width, rise_slope
    # You want:
    # one feature vector per signal segment so you compute:
    # median (robust central tendency)
    # CV (coefficient of variation) = variability
    row = {
        "N_BEATS": len(beat_df)
    }
    try:
        # -------------------------
        # STATISTICS
        # -------------------------
        if len(beat_df) >= 3:
            for c in beat_df.columns:
                row[f"{c}_median"] = beat_df[c].median()
                mean = beat_df[c].mean()
                row[f"{c}_cv"] = beat_df[c].std() / mean if mean != 0 else -1.0
        else:
            # assign -1 instead of np.nan
            for c in ["amplitude", "T", "Tr", "Tr_norm", "AUC", "width", "rise_slope"]:
                row[f"{c}_median"] = -1  # np.nan
                row[f"{c}_cv"] = -1  # np.nan

    except Exception as e:
        if RAISE_EXCEPTION_ON_PPG_PROCESSING:
            raise Exception(e)
        else:
            print(f"########## Error: {e}")

    return row


def extract_features_group4_hrv(ppg_frame, peaks, fs, show=False):
    # Compute HRV features as dataframe with 1 row and 25 features
    hrv_features_df = nk.hrv_time(peaks, sampling_rate=fs, show=False)

    if (show):
        print(peaks)
        print(hrv_features_df)
        plt.plot(ppg_frame)
        plt.scatter(peaks, ppg_frame[peaks], c='r')
        plt.show()

    # Convert to dictionary
    if hrv_features_df is not None and not hrv_features_df.empty:
        hrv_dict = hrv_features_df.iloc[0].to_dict()

    return hrv_dict


if __name__ == "__main__":
    dataset_config_file = "multimodal_dataset_folders.json"

    ppg_filtering_pipeline(dataset_config_file)
    ppg_quality_pipeline(dataset_config_file)

    # debug
    # _debug_processing_files(dataset_config_file)
