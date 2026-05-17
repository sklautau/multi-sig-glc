''' 
Write file with segments of good quality
'''
import numpy as np

from datasets_util.naming_conventions import DatasetConfig
from datasets_util.segments import create_segments_dataframe

OUTPUT_ALL_SEGMENTS_FILE = "all_segments_with_quality.csv"

if __name__ == "__main__":
    fs_dict = {
        "ecg": 500,
        "ppg": 500,
        "bioimp": np.nan  # set to NaN if not available or not applicable
    }

    # Create a DatasetConfig instance to access dataset paths and info
    dataset_config_file = "multimodal_dataset_folders.json"
    datasetConfig = DatasetConfig(dataset_config_file)

    # generate segments file
    dataset_segments = OUTPUT_ALL_SEGMENTS_FILE
    output_file_name = create_segments_dataframe(
        dataset_config_file, dataset_segments)
