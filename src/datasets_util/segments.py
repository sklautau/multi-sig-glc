'''
Utilities to read and write Dataframe files with segments.
The segments are defined as contiguous time intervals of good quality data, based on a quality indicator (e.g., signal quality index) and a minimum duration threshold.
The output is a CSV file with the following columns:

segment_id,file_id,modality,start_sample,duration,quality_indicator
seg_id0,file_id2,ppg,0,80840,0.9932449460029602
seg_id1,file_id2,ppg,81183,14800,0.9863582849502563
...

where segment_id is a unique identifier for each segment across all files in the dataset, file_id is the identifier of the file from which the segment was extracted, modality is the type of signal (e.g., ppg, ecg, bioimpedance_frequency), start_sample is the index of the first sample of
the segment (inclusive), duration is the length of the segment in samples, and
quality_indicator is the median signal quality index for the segment
in the given modality.
'''

from pathlib import Path
from typing import Any, Dict, Union, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import neurokit2 as nk
import os

from datasets_util.naming_conventions import DatasetConfig
from datasets_util.waveforms import read_sigmf_file

SegmentRow = Dict[str, Any]
SegmentRows = List[SegmentRow]


MIN_THRESHOLD_PPG = 0.95  # quality threshold for good quality segments
MIN_THRESHOLD_ECG = 0.5  # quality threshold for good quality segments
MIN_DURATION = 10  # minimum duration in seconds for good quality segments
REQUIRED_FS = 60  # assumed sampling frequency (Hz)

# In segments.py - new SegmentExtractor class


class SegmentManager:
    '''
    Class to extract segments of good quality data from the dataset.
    For instance, it can be based on a quality indicator and a minimum duration threshold.
    '''

    def __init__(self, datasetConfig: DatasetConfig,
                 segments: Union[pd.DataFrame, str, Path]):
        self._datasetConfig = datasetConfig
        # now initialize the segments dataframe based on the input type
        if isinstance(segments, pd.DataFrame):
            self._segments_dataframe = segments
        elif isinstance(segments, (str, Path)):
            self._segments_dataframe = self.load_segments(str(segments))
        else:
            raise ValueError(
                "Invalid segments type. Expected DataFrame, str, or Path.")

    def get_segments_dataframe(self) -> pd.DataFrame:
        """Return the segments info as a pandas DataFrame."""
        # make a deep copy to prevent external modifications
        return self._segments_dataframe.copy()

    def get_segments_of_file_id(self, file_id: str) -> pd.DataFrame:
        return self._segments_dataframe.loc[self._segments_dataframe['file_id'] == file_id]

    def old_get_segments(self, file_id: str) -> pd.DataFrame:
        # AK-TODO
        df = self.get_segments_of_file_id(file_id)
        # loop over the records of this file_id and find the one with the given segment_id
        for _, row in df.iterrows():
            # open the file and extract the segment based on the start and end indices
            segment_path = self.get_gen_complete_path(
                file_id, waveform_id="segments", new_extension="csv")
            if not os.path.exists(segment_path):
                raise FileNotFoundError(
                    f"Segment file not found: {segment_path}")
            return pd.read_csv(segment_path)

    def load_segments(self, segments_file: str) -> pd.DataFrame:
        # Load segments from file
        segments_path = Path(segments_file)
        segments = None
        if segments_path.exists():
            segments = pd.read_csv(segments_file)
        else:
            complete_path = Path(
                self._datasetConfig.get_dataset_segments_path()) / segments_file
            if complete_path.exists():
                segments = pd.read_csv(complete_path)
            else:
                raise FileNotFoundError(
                    f"Segments file not found as {segments_file} nor {complete_path}")
        return segments

    def best_segment(self, file_id: str, modality: str) -> Optional[SegmentRow]:
        print(f"Finding best segment for file_id {file_id}...")
        df = self.get_segments_of_file_id(file_id)
        return best_segment(df, file_id, modality)


@staticmethod
def best_segment(all_segments_dataframe, file_id: str, modality: str) -> (Dict[str, Any], float):
    print(f"Finding best segment for file_id {file_id}...")
    segments_dataframe = all_segments_dataframe.loc[all_segments_dataframe['file_id'] == file_id]
    # filter by modality
    segments_dataframe = segments_dataframe[segments_dataframe["modality"] == modality]
    if segments_dataframe.empty:
        raise ValueError(
            f"No segments found for file_id {file_id} and modality {modality}")
    best_row = segments_dataframe.loc[segments_dataframe['quality_indicator'].idxmax(
    )]
    largest_quality = best_row['quality_indicator']
    # print(
    #    f"AAA Best segment for file_id {file_id}: segment_id {best_row['segment_id']}, modality {best_row['modality']}, quality_indicator {best_row['quality_indicator']}")
    return best_row.to_dict(), largest_quality


def format_segment_row(
    segments_data: SegmentRows,
    valid_segment_index: int,
    file_id: str,
    start: int,
    end: int,
    quality: np.ndarray,
    modality: str
) -> SegmentRows:
    segments_data.append({
        # label in NeuroKit2 convention (unique for each segment across all files in given dataframe/dataset)
        "segment_id": "seg_id" + str(valid_segment_index),
        "file_id": file_id,
        "modality": modality,
        "start_sample": start,  # onset in NeuroKit2 convention (inclusive)
        # duration in samples (so end index is exclusive in NeuroKit2 convention)
        "duration": end - start,
        "quality_indicator": np.median(quality[start:end])
    })

    print(f"  quality_indicator: {np.median(quality[start:end])}")
    return segments_data


def extract_bioimpedance_segments(
    quality_indicator: np.ndarray,
    file_id: str,
    current_index: int = 0,
) -> pd.DataFrame:
    modality = "bioimpedance_frequency"
    # for bioimpedance, we only have one segment for the whole file, so we can just return a single row with the quality indicator for the whole file
    start = 0
    end = len(quality_indicator)
    # in general, it stores all segments for this file, but in the bioimpedance case, we only have one segment for the whole file
    this_file_segments_data: SegmentRows = []
    segments_data = format_segment_row(
        this_file_segments_data, current_index, file_id, start, end, quality_indicator, modality)
    return pd.DataFrame(segments_data)


def extract_waveform_segments(
    quality_indicator: np.ndarray,
    file_id: str,
    fs: int,
    modality: str,
    current_index: int = 0,
    min_quality: float = 0.5,
    min_len_seconds: int = 5
) -> pd.DataFrame:
    quality_indicator = np.asarray(quality_indicator)

    # Convert threshold from seconds → samples
    min_len = int(min_len_seconds * fs)

    # boolean mask of valid quality samples
    good = quality_indicator >= min_quality

    this_file_segments_data = list()  # all segments for this file
    valid_segment_index = current_index
    start = None

    for i in range(len(good)):
        if good[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                end = i
                if (end - start) >= min_len:
                    this_file_segments_data = format_segment_row(
                        this_file_segments_data, valid_segment_index, file_id, start, end, quality_indicator, modality)
                    valid_segment_index += 1
                start = None

    # If segment continues until the last sample
    if start is not None:
        end = len(good)
        if (end - start) >= min_len:
            this_file_segments_data = format_segment_row(
                this_file_segments_data, valid_segment_index, file_id, start, end, quality_indicator, modality)
        valid_segment_index += 1

    # return a dataframe with all segments for this file
    df_segments = pd.DataFrame(this_file_segments_data)
    return df_segments


def quality_event_pipeline(datasetConfig: DatasetConfig) -> pd.DataFrame:
    # for naming convention when saving processed files (e.g., "file_id_filtered.csv")
    waveform_id = "quality"

    df = datasetConfig.get_dataset_info_dataframe()

    # Filter only bioimpedance
    # df_bio = df[df["modality"].str.contains("bioimpedance")].copy()

    counter = 0
    output_dataframe = None  # initialize output dataframe to store segments from all files
    for _, row in df.iterrows():
        file_id = row["file_id"]
        modality = row["modality"]

        complete_path = datasetConfig.get_gen_complete_path(
            file_id, waveform_id)

        print(
            f"Processing file {file_id} with modality {modality}")

        # switch depending on modality
        if modality == "ppg" or modality == "ecg":
            # read the signal and metadata
            signal, metadata = read_sigmf_file(complete_path)
            fs = metadata["global"]["core:sample_rate"]

            # identify good quality segments based on the threshold
            if modality == "ppg":
                min_quality = MIN_THRESHOLD_PPG
            elif modality == "ecg":
                min_quality = MIN_THRESHOLD_ECG
            else:
                raise ValueError(f"Modality {modality} is not supported")
            this_dataframe = extract_waveform_segments(signal, file_id,
                                                       fs, modality,
                                                       current_index=counter,
                                                       min_quality=min_quality,
                                                       min_len_seconds=MIN_DURATION,
                                                       )
            num_segments = len(this_dataframe)
            print(f"  Found {num_segments} good segments in file {file_id}")

            counter += num_segments
        elif modality == "bioimpedance_frequency":
            # read the csv with header frequency,quality
            quality_indicator = pd.read_csv(complete_path)["quality"].values
            quality_indicator = np.asarray(quality_indicator)
            this_dataframe = extract_bioimpedance_segments(
                quality_indicator, file_id, current_index=counter)
        else:
            raise ValueError(f"Modality {modality} is not supported")
        # create a new dataframe or update one to store the segments for this file
        if output_dataframe is None:
            output_dataframe = this_dataframe
        else:
            output_dataframe = pd.concat(
                [output_dataframe, this_dataframe], ignore_index=True)
    return output_dataframe


def create_segments_dataframe(dataset_config_file: str, output_file_name: str) -> str:

    datasetConfig = DatasetConfig(dataset_config_file)

    events_dataframe = quality_event_pipeline(datasetConfig)

    # save the combined dataframe with all segments from all files
    output_combined_path = os.path.join(
        datasetConfig.get_dataset_segments_path(), output_file_name)
    events_dataframe.to_csv(output_combined_path, index=False)
    print(
        f"Wrote combined segments dataframe to CSV file {output_combined_path}")
    return output_file_name


def number_of_windows(segment_duration_seconds: float, window_size_seconds: float, window_shift_seconds: float,
                      fs: float) -> int:
    '''
    Calculate the number of windows of size L seconds with shift S seconds that can be extracted from a segment of duration D seconds.
    The formula is: M = floor((N - L) / S) + 1
    See: https://ai6g.org/books/dsp/BlockorWindowProcessing.html#-the-top-representation-shows-nonoverlapping-windows-of-l-samples-with-both-nonwindowed-indexing-xn-and-windowed-indexing-xkm-the-bottom-representation-shows-overlapping-windows-with-l-and-shift-s-sample-using-nonwindowed-indexing
    '''
    N = int(window_size_seconds * fs)  # segment duration in samples
    L = int(window_size_seconds * fs)  # window size in samples
    S = int(window_shift_seconds * fs)  # window shift in samples
    M = np.floor((N - L) / S).astype(int) + 1  # number of windows per segment
    return M


@staticmethod
def segments_to_fixed_duration_windows(segments_dataframe: pd.DataFrame, window_size_seconds: float,
                                       window_shift_seconds: float, fs: float) -> pd.DataFrame:
    '''
    Split segments into fixed-duration windows with given shift.
    Returns a new dataframe with each row representing a window.
    '''
    # Convert time to samples
    window_size_samples = int(window_size_seconds * fs)
    window_shift_samples = int(window_shift_seconds * fs)

    # Calculate number of windows per segment
    number_of_windows_per_segment = segments_dataframe.apply(lambda row: number_of_windows(
        row["duration"] / fs, window_size_seconds, window_shift_seconds, fs), axis=1)

    windows_rows = []
    window_id_counter = 0

    for idx, (_, segment_row) in enumerate(segments_dataframe.iterrows()):
        num_windows = number_of_windows_per_segment.iloc[idx]
        segment_start_sample = segment_row["start_sample"]
        segment_duration = segment_row["duration"]

        # Check if segment is too short for even one window
        if num_windows < 1:
            print(f"Warning: Segment {segment_row['segment_id']} (file_id={segment_row['file_id']}, "
                  f"modality={segment_row['modality']}) is too short for a window. "
                  f"Duration: {segment_duration / fs:.2f}s, Window size: {window_size_seconds}s")
            continue

        # Create one row per window
        for window_idx in range(num_windows):
            window_start_sample = segment_start_sample + window_idx * window_shift_samples

            window_row = {
                "window_id": f"win_id{window_id_counter}",
                "segment_id": segment_row["segment_id"],
                "file_id": segment_row["file_id"],
                "modality": segment_row["modality"],
                "start_sample": window_start_sample,
                "duration": window_size_samples,
                "quality_indicator": segment_row["quality_indicator"]
            }
            windows_rows.append(window_row)
            window_id_counter += 1

    windows_dataframe = pd.DataFrame(windows_rows)
    print(
        f"Created {len(windows_dataframe)} windows from {len(segments_dataframe)} segments")
    return windows_dataframe


if __name__ == "__main__":
    dataset_config_file = "multimodal_dataset_folders.json"
    output_file_name = create_segments_dataframe(dataset_config_file)

    # now create a SegmentExtractor
    dataset_config = DatasetConfig(dataset_config_file)
    segmentExtractor = SegmentManager(dataset_config, output_file_name)

    print(segmentExtractor.get_segments_dataframe().head())

    file_id = "file_id1"
    modality = "ecg"
    segmentExtractor.best_segment(file_id, modality)

    # Test the new segments_to_fixed_duration_windows method
    print("\n" + "="*60)
    print("Testing segments_to_fixed_duration_windows method")
    print("="*60)

    segments_df = segmentExtractor.get_segments_dataframe()
    print(f"\nOriginal segments shape: {segments_df.shape}")
    print(f"Original segments:\n{segments_df.head()}")

    # Convert segments to fixed-duration windows (5 seconds windows with 2 second shift)
    window_size_seconds = 5.0
    window_shift_seconds = 2.0
    fs = 500  # sampling frequency in Hz

    windows_df = segments_to_fixed_duration_windows(
        segments_df,
        window_size_seconds=window_size_seconds,
        window_shift_seconds=window_shift_seconds,
        fs=fs
    )

    print(f"\nWindows dataframe shape: {windows_df.shape}")
    print(f"Windows dataframe:\n{windows_df.head(10)}")
    print(f"\nTotal windows created: {len(windows_df)}")
