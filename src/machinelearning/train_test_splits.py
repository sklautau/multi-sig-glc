'''
Manually create a pandas dataframe with the
following column name:
- participant_id
in order to split the data into disjoint sets:
 training, validation and test sets.
'''

from sklearn.model_selection import train_test_split
import pandas as pd
import os
from pathlib import Path

from datasets_util.naming_conventions import DatasetConfig

DATASET_CONFIG_FILE = "multimodal_dataset_folders.json"
INPUT_FEATURES_FILE = "multimodal_features_with_metadata.csv"
PREAMBLE_PARTICIPANTS_FILE_NAMES = "participants"
PREAMBLE_TRAIN_TEST_SPLITS_FILE_NAMES = "split_id1"


def extract_participants_from_features_file(dataframe, list_of_participant_ids) -> pd.DataFrame:
    return dataframe[dataframe["participant_id"].isin(list_of_participant_ids)]


def train_test_splits_from_features_file(output_folder) -> None:
    features_file = os.path.join(
        output_folder, INPUT_FEATURES_FILE)
    df = pd.read_csv(features_file)
    # read the participants_train.txt file, taking in account that the first row is the header
    training_set_participants = pd.read_csv(os.path.join(
        output_folder, PREAMBLE_PARTICIPANTS_FILE_NAMES + "_train.txt"), header=0)
    test_set_participants = pd.read_csv(os.path.join(
        output_folder, PREAMBLE_PARTICIPANTS_FILE_NAMES + "_test.txt"), header=0)
    validation_set_participants = pd.read_csv(os.path.join(
        output_folder, PREAMBLE_PARTICIPANTS_FILE_NAMES + "_validation.txt"), header=0)
    training_set = extract_participants_from_features_file(
        df, training_set_participants["participant_id"])
    test_set = extract_participants_from_features_file(
        df, test_set_participants["participant_id"])
    validation_set = extract_participants_from_features_file(
        df, validation_set_participants["participant_id"])

    # save the sets to csv files:
    training_set.to_csv(os.path.join(
        output_folder, PREAMBLE_TRAIN_TEST_SPLITS_FILE_NAMES + "_train.csv"), index=False)
    test_set.to_csv(os.path.join(
        output_folder, PREAMBLE_TRAIN_TEST_SPLITS_FILE_NAMES + "_test.csv"), index=False)
    validation_set.to_csv(os.path.join(
        output_folder, PREAMBLE_TRAIN_TEST_SPLITS_FILE_NAMES + "_validation.csv"), index=False)

    print(
        f"Wrote training set to {os.path.join(output_folder, PREAMBLE_TRAIN_TEST_SPLITS_FILE_NAMES + '_train.csv')}")
    print(
        f"Wrote test set to {os.path.join(output_folder, PREAMBLE_TRAIN_TEST_SPLITS_FILE_NAMES + '_test.csv')}")
    print(
        f"Wrote validation set to {os.path.join(output_folder, PREAMBLE_TRAIN_TEST_SPLITS_FILE_NAMES + '_validation.csv')}")


def create_splits_of_participants(datasetConfig: DatasetConfig) -> None:
    training_set = pd.DataFrame(
        [
            {"participant_id": "ieb_01"},
            {"participant_id": "ieb_02"},
            {"participant_id": "ieb_03"},
            {"participant_id": "ieb_04"},
            {"participant_id": "ieb_08"},
        ]
    )
    test_set = pd.DataFrame(
        [
            {"participant_id": "ieb_05"},
            {"participant_id": "ieb_07"},
        ]
    )
    validation_set = pd.DataFrame([{"participant_id": "ieb_06"}])

    # Save the sets to csv files:
    output_folder = datasetConfig.get_dataset_machine_learning_path()

    output_train_file = os.path.join(
        output_folder, PREAMBLE_PARTICIPANTS_FILE_NAMES + "_train.txt")
    training_set.to_csv(output_train_file, index=False)

    output_test_file = os.path.join(
        output_folder, PREAMBLE_PARTICIPANTS_FILE_NAMES + "_test.txt")
    test_set.to_csv(output_test_file, index=False)

    output_validation_file = os.path.join(
        output_folder, PREAMBLE_PARTICIPANTS_FILE_NAMES + "_validation.txt")
    validation_set.to_csv(output_validation_file, index=False)

    print(f"Wrote list of participants in training set to {output_train_file}")
    print(f"Wrote list of participants in test set to {output_test_file}")
    print(
        f"Wrote list of participants in validation set to {output_validation_file}")
    print(f"Wrote validation set to {output_validation_file}")


def create_random_splits_of_participants(
    output_folder: str,
    df: pd.DataFrame,
    train_pct: float = 0.7,
    val_pct: float = 0.15,
    test_pct: float = 0.15,
    random_state: int = 42,
) -> None:
    """
    This method is adapted for IEB_1.

    Splits participants into train/validation/test sets based on percentages.

    Args:
        datasetConfig: config object with output path
        df: dataframe containing a 'participant_id' column
        train_pct, val_pct, test_pct: must sum to 1.0
        random_state: for reproducibility
    """

    assert abs(train_pct + val_pct + test_pct -
               1.0) < 1e-6, "Percentages must sum to 1.0"

    # Get unique participants
    participants = df["participant_id"].dropna().unique()
    participants = pd.DataFrame({"participant_id": participants})

    # First split: train vs temp (val+test)
    train_df, temp_df = train_test_split(
        participants,
        test_size=(1 - train_pct),
        random_state=random_state,
        shuffle=True,
    )

    # Second split: validation vs test
    val_relative = val_pct / (val_pct + test_pct)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_relative),
        random_state=random_state,
        shuffle=True,
    )

    # Save
    # output_folder = datasetConfig.get_dataset_machine_learning_path()

    train_file = os.path.join(
        output_folder, PREAMBLE_PARTICIPANTS_FILE_NAMES + "_train.txt")
    val_file = os.path.join(
        output_folder, PREAMBLE_PARTICIPANTS_FILE_NAMES + "_validation.txt")
    test_file = os.path.join(
        output_folder, PREAMBLE_PARTICIPANTS_FILE_NAMES + "_test.txt")

    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"Train: {len(train_df)} participants → {train_file}")
    print(f"Validation: {len(val_df)} participants → {val_file}")
    print(f"Test: {len(test_df)} participants → {test_file}")


if __name__ == "__main__":
    datasetConfig = DatasetConfig(DATASET_CONFIG_FILE)
    create_splits_of_participants(datasetConfig)
    output_folder = datasetConfig.get_dataset_machine_learning_path()
    train_test_splits_from_features_file(output_folder)
