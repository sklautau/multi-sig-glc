'''
Utilities to read and write SigMF files.
'''

from pathlib import Path
import os
import pandas as pd
import json
from typing import Dict, Any, Union, Optional


class DatasetConfig:
    def __init__(self, config: Union[Dict[str, Any], str, Path]) -> None:
        """
        Initialize DatasetConfig from either:
        - a dictionary
        - a JSON file path (str or Path)

        Obs: this is a classic case for a single constructor with polymorphic input, keeping the API clean while supporting both
        dict and JSON file paths. The key idea is: accept a union type and internally normalize to a dictionary.
        """
        # Normalize input → dict
        if isinstance(config, (str, Path)):
            config_path = Path(config)
            if not config_path.exists():
                raise FileNotFoundError(
                    f"Config file not found: {config_path}")

            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)

        elif isinstance(config, dict):
            config_dict = config

        else:
            raise TypeError(
                "config must be either a dict or a path to a JSON file"
            )

        # Store raw dict
        self.config = config_dict

        # Normalize paths using pathlib
        self.dataset_file = Path(config_dict["DATASET_FILE"])
        self.generated_waveforms_path = Path(
            config_dict["GENERATED_WAVEFORMS_PATH"]
        )
        self.segments_path = Path(config_dict["SEGMENTS_PATH"])
        self.events_path = Path(config_dict["EVENTS_PATH"])
        self.machine_learning_path = Path(config_dict["MACHINE_LEARNING_PATH"])
        self.ppg_fs = int(config_dict.get("PPG_FS", 500))

        # Load the dataset info for speeding up access to file_id related info, e.g. relative paths
        self._dataset_info = self.__load_dataset_info()

        # Create folders if they do not exist
        self._pre_create_directories()

        # Validate configuration
        self._validate()

    def get_dataset_info_dataframe(self) -> pd.DataFrame:
        """Return the dataset info as a pandas DataFrame."""
        # make a deep copy to prevent external modifications
        return self._dataset_info.copy()

    def get_relative_path(self, file_id: str) -> str:
        # does not include "raw" subfolder as prefix, e.g. "S1_2025-07-29_14-17/ecg/ecg_125mgdL.sigmf-data"
        row = self._dataset_info.loc[self._dataset_info['file_id'] == file_id]
        if row.empty:
            raise ValueError(f"file_id '{file_id}' not found in dataset info")
        relative_path = Path(row['relative_path'].values[0])
        # ensure consistent path format with forward slashes
        # convert to string with forward slashes for consistency across platforms
        return relative_path.as_posix()

    def get_raw_relative_path(self, file_id: str) -> str:
        # includes "raw" subfolder as prefix, e.g. "raw/ieb_01/S1_2025-07-29_14-17/ecg/ecg_125mgdL.sigmf-data"
        relative_path = self.get_relative_path(file_id)
        # Ensure consistent path format
        output = Path("raw") / Path(relative_path)
        # convert to string with forward slashes for consistency across platforms
        return output.as_posix()

    def get_gen_relative_path(self, file_id: str, waveform_id: str, new_extension: Optional[str] = None) -> str:
        # gen reminds generated waveforms, e.g. "filtered_waveform/S1_2025-07-29_14-17/ecg/ecg_125mgdL_filtered_waveform.sigmf-data"
        relative_path = self.get_relative_path(file_id)
        output = Path(waveform_id) / _path_replace_file_name(relative_path,
                                                             waveform_id, new_extension=new_extension)
        # convert to string with forward slashes for consistency across platforms
        return output.as_posix()

    def get_raw_complete_path(self, file_id: str) -> str:
        """
        Returns the path to the dataset file.
        Call it "complete" path instead of "full" path because
        the output can be a relative path.
        """
        relative_path = self.get_raw_relative_path(file_id)
        complete_path = self.dataset_file.parent / relative_path
        # convert to string with forward slashes for consistency across platforms
        return complete_path.as_posix()

    def get_gen_complete_path(self, file_id: str, waveform_id: str, new_extension: Optional[str] = None) -> str:
        """
        Returns the path to the dataset file.
        Call it "complete" path instead of "full" path because
        the output can be a relative path.
        """
        relative_path = self.get_gen_relative_path(
            file_id, waveform_id, new_extension=new_extension)
        complete_path = self.generated_waveforms_path / relative_path
        # convert to string with forward slashes for consistency across platforms
        return complete_path.as_posix()

    def get_file_id(self, relative_path: str) -> str:
        """
        Return file_id for a given relative_path.
        Raises KeyError if not found or not unique.
        """
        matches = self._dataset_info.loc[self._dataset_info["relative_path"]
                                         == relative_path, "file_id"]

        if matches.empty:
            raise KeyError(f"relative_path not found: {relative_path}")

        if len(matches) > 1:
            raise ValueError(f"Multiple file_id entries for: {relative_path}")

        return matches.iloc[0]

    def get_dataset_root_path(self) -> str:
        """
        Returns the directory containing the dataset file,
        which corresponds to the DATASET_ROOT_PATH.
        """
        return Path(self.dataset_file.parent).as_posix()  # convert to string with forward slashes for consistency across platforms

    def get_dataset_events_path(self) -> str:
        # convert to string with forward slashes for consistency across platforms
        return Path(self.events_path).as_posix()

    def get_dataset_segments_path(self) -> str:
        # convert to string with forward slashes for consistency across platforms
        return Path(self.segments_path).as_posix()

    def get_dataset_machine_learning_path(self) -> str:
        # convert to string with forward slashes for consistency across platforms
        return Path(self.machine_learning_path).as_posix()

    def get_ppg_fs(self) -> int:
        return self.ppg_fs

    def as_init_dict(self) -> Dict[str, str]:
        """Return config as plain dict (string paths)
        to eventually initialize other classes."""
        return {
            "DATASET_FILE": str(self.dataset_file),
            "GENERATED_WAVEFORMS_PATH": str(self.generated_waveforms_path),
            "SEGMENTS_PATH": str(self.segments_path),
            "EVENTS_PATH": str(self.events_path),
            "MACHINE_LEARNING_PATH": str(self.machine_learning_path),
            "PPG_FS": self.ppg_fs
        }

    def _pre_create_directories(self) -> None:
        """Create folders if they do not exist."""
        for folder in [
            self.generated_waveforms_path,
            self.segments_path,
            self.events_path,
            self.machine_learning_path
        ]:
            folder.mkdir(parents=True, exist_ok=True)

    def __load_dataset_info(self) -> pd.DataFrame:
        df = pd.read_csv(self.dataset_file)
        return df

    def _validate(self) -> None:
        """
        Validate that the dataset file exists and is a _dataset.csv file.
        """
        if not self.dataset_file.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.dataset_file}")
        if not self.dataset_file.name.endswith("_dataset.csv"):
            raise ValueError(
                f"Dataset file must have _dataset.csv extension: {self.dataset_file}")

        if self.ppg_fs <= 0:
            raise ValueError(
                f"PPG_FS must be a positive integer, got: {self.ppg_fs}")

        dataset_raw_folder = Path(self.get_dataset_root_path()) / "raw"
        if not dataset_raw_folder.exists():
            raise FileNotFoundError(
                f"Subfolder raw in dataset root folder not found: {dataset_raw_folder}. A subfolder called raw is mandatory!")


def create_dataset_info(
    dataset_file: str = "../original_data/raw/mysignals_dataset.csv",
    base_generated_path: str = "../output_folder/mydataset/",
    ppg_fs: int = 500
) -> Dict[str, str]:
    """
    Builds a standardized dataset configuration dictionary from folders.
    """

    base_path = Path(base_generated_path)

    config = {
        "DATASET_FILE": str(Path(dataset_file)),
        "GENERATED_WAVEFORMS_PATH": str(base_generated_path),
        "SEGMENTS_PATH": str(base_path / "segments"),
        "EVENTS_PATH": str(base_path / "events"),
        "MACHINE_LEARNING_PATH": str(base_path / "ml"),
        "PPG_FS": int(ppg_fs)
    }

    return config


def save_dataset_init_dic(filename: str, my_dict: Dict[str, Any]) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(my_dict, f, indent=2)


def load_dataset_init_dic(filename: str) -> Dict[str, Any]:
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def _path_replace_raw_folder_and_filename(input_filename: str, root_dir: str, output_folder: str, output_suffix: str, new_extension: Optional[str] = None) -> str:
    temp_filename = _path_replace_raw_folder(
        input_filename, root_dir, output_folder)
    return _path_replace_file_name(temp_filename, output_suffix, new_extension=new_extension)


def _path_replace_extension(input_filename: str, new_extension: str) -> str:
    # New file extension (without leading dot), by default "sigmf-data".
    # --- Replace extension ---
    directory = os.path.dirname(input_filename)
    base_name = os.path.basename(input_filename)
    name_without_extension, _ = os.path.splitext(base_name)
    new_name = name_without_extension + "." + new_extension
    return os.path.join(directory, new_name)


def _path_replace_file_name(input_filename: str, output_suffix: str, new_extension: Optional[str] = None) -> str:
    '''
    Replace the file name of a given path with a new suffix and optional extension.

    Parameters
    ----------
    input_filename : str
        The path to the file to be modified.
    output_suffix : str
        The suffix to be appended to the file name.
    new_extension : str, optional
        The new extension of the file (without leading dot). If None, keep the same extension.

    Returns
    -------
    str
        The modified path with the new suffix and optional extension.
    '''
    directory = os.path.dirname(input_filename)
    filename = os.path.basename(input_filename)
    base_filename, input_extension = os.path.splitext(filename)
    # print("base_filename:", base_filename, "input_extension:", input_extension)
    if new_extension is None:
        # keep the same extension but skip the dot '.ext'
        new_extension = input_extension[1:]
    out_filename = base_filename + "_" + output_suffix + "." + new_extension
    return os.path.join(directory, out_filename)


def _path_replace_raw_folder(input_filename: str, root_dir: str, output_folder: str) -> str:
    """
    Construct an output file path by preserving the relative directory structure
    of an input file with respect to a given root directory, and
    redirecting the result to a target output folder.

    This version fully supports relative paths and does NOT convert them to absolute
    paths. All outputs remain consistent with the input path type (relative in,
    relative out).

    Processing steps:
    1) Normalize paths using os.path.normpath (no absolute conversion)
    2) Compute the relative path from root_dir to input_filename
    3) Validate that the result does not escape root_dir (no ".." prefix)
    4) Recreate the relative directory structure inside output_folder
    5) Return the final output path

    Parameters
    ----------
    input_filename : str
        Path to the input file (relative or absolute).
    root_dir : str
        Root directory used as reference for relative path computation.
        Must be consistent (both relative or both absolute) with input_filename.
    output_folder : str
        Base directory where output files will be stored.
        Will preserve relative/absolute nature.

    Returns
    -------
    str
        Output file path with preserved structure and updated extension.

    Raises
    ------
    ValueError
        If input_filename is not inside root_dir.

    Example
    -------
    Inputs:

        input_filename = "./data/raw/session1/file1.wav"
        root_dir      = "./data/raw"
        output_folder = "./data/processed"

    Step-by-step:

        1) Normalize (no absolute conversion):
           input_filename -> "data/raw/session1/file1.wav"
           root_dir      -> "data/raw"

        2) Compute relative path:
           rel_path = "session1/file1.wav"

        3) Validate containment:
           OK (does not start with "..")

        4) Extract directory:
           rel_dir = "session1"

        5) Construct output directory:
           out_dir = "data/processed/session1"

        6) Final output:
           "data/processed/session1/file1.sigmf-data"

    Notes
    -----
    - No conversion to absolute paths is performed.
    - The function assumes input_filename and root_dir are both relative
      or both absolute; mixing them may lead to incorrect results.
    - Containment is enforced by checking that the relative path does not
      start with "..".
    """

    # --- Step 1: Normalize (no abspath) ---
    input_filename = os.path.normpath(input_filename)
    root_dir = os.path.normpath(root_dir)
    output_folder = os.path.normpath(output_folder)

    # --- Step 2: Relative path ---
    rel_path = os.path.relpath(input_filename, root_dir)

    # --- Step 3: Validate containment (robust) ---
    # use absolute paths for validation but do not use them for output construction
    input_abs = os.path.abspath(input_filename)
    root_abs = os.path.abspath(root_dir)

    if os.path.commonpath([input_abs, root_abs]) != root_abs:
        print("input_abs:", input_abs)
        print("root_abs:", root_abs)
        print(
            f"ERROR: root_dir '{root_dir}' is not inside input_filename '{input_filename}'")
        raise ValueError("input_filename must be inside root_dir")
    rel_dir = os.path.dirname(rel_path)

    # --- Step 4: Recreate directory ---
    out_dir = os.path.join(output_folder, rel_dir)
    os.makedirs(out_dir, exist_ok=True)

    # --- Step 5: Final path ---
    base_name = os.path.basename(input_filename)
    output_filename = os.path.join(out_dir, base_name)

    return output_filename


if __name__ == "__main__":
    # Example usage
    dataset_folders = create_dataset_info(dataset_file="../original_data/multimodal_dataset_example/multimodal_dataset.csv",
                                          base_generated_path="../generated_waveforms/multimodal_dataset_example/")

    file_path = "temp_multimodal_dataset.json"
    save_dataset_init_dic(file_path, dataset_folders)
    dataset_folders = load_dataset_init_dic(file_path)
    print("Loaded dataset folders:", dataset_folders)

    datasetConfig = DatasetConfig(dataset_folders)
    file_id = "file_id3"
    relative_path = datasetConfig.get_relative_path(file_id)
    print("Relative path for file_id", file_id, ":", relative_path)

    relative_raw_path = datasetConfig.get_raw_relative_path(file_id)
    print("Relative raw path for file_id", file_id, ":", relative_raw_path)

    waveform_id = "filtered_waveform"
    relative_gen_path = datasetConfig.get_gen_relative_path(
        file_id, waveform_id, new_extension="test_ext")
    print("Relative generated path for file_id", file_id,
          "and waveform_id", waveform_id, ":", relative_gen_path)

    output_filename = datasetConfig.get_gen_complete_path(
        file_id, waveform_id, new_extension="test_ext")
    print("Input relative_path:", relative_path)
    print("Output filename for waveform_id:", output_filename)
